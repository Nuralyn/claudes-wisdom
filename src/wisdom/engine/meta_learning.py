"""Meta-learning engine — the system learns how it fails.

When wisdom is deprecated or cascades contamination, that data is logged.
This engine analyzes those logs to identify patterns: which types of wisdom
fail most often, which creation methods produce unreliable entries, and
which domains are contamination hotspots.

The output is actionable: risk scores that the adversarial engine can use
to adjust its challenge thresholds. High-risk profiles get scrutinized harder.

All analysis is computed on-demand from existing tables. No new tables.
No stored computed values. This follows the system's principle:
"compute at read time, not write time."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from wisdom.config import WisdomConfig
from wisdom.logging_config import get_logger
from wisdom.storage.sqlite_store import SQLiteStore

logger = get_logger("engine.meta_learning")


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO-ish timestamp from SQLite into a datetime."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


# ── Data structures (output-only, matching ChallengeReport pattern) ───


@dataclass
class FailureProfile:
    """Failure rates for a category of wisdom."""

    category: str  # e.g. "type:heuristic", "method:pipeline", "domain:databases"
    total_count: int
    deprecated_count: int
    contamination_events: int
    failure_rate: float  # deprecated_count / total_count
    contamination_rate: float  # contamination_events / total_count


@dataclass
class RiskScore:
    """Computed risk for a specific wisdom entry based on its historical profile."""

    wisdom_id: str
    base_risk: float  # 0.0 (safe) to 1.0 (high risk)
    risk_factors: list[dict] = field(default_factory=list)
    recommended_challenge_level: str = "standard"  # standard | elevated | maximum


@dataclass
class ContaminationPattern:
    """Analysis of how contamination spreads from a single source."""

    source_wisdom_id: str
    total_affected: int
    avg_penalty: float
    affected_types: dict = field(default_factory=dict)  # {"wisdom": N, "knowledge": N, ...}


@dataclass
class VelocityProfile:
    """How quickly a category of wisdom matures (gains confidence over time)."""

    category: str  # e.g. "type:heuristic", "domain:databases"
    entry_count: int
    avg_velocity: float  # avg confidence delta per day (positive = maturing)
    avg_events_per_day: float  # how actively confidence changes
    fastest_id: str | None = None
    slowest_id: str | None = None


@dataclass
class VolatilityEntry:
    """A single wisdom entry with erratic confidence history."""

    wisdom_id: str
    statement_preview: str
    total_events: int
    direction_changes: int
    avg_delta_magnitude: float
    max_swing: float
    volatility_score: float  # 0.0 (stable) to 1.0 (erratic)


class MetaLearningEngine:
    """Analyzes historical failure data to close the meta-learning loop.

    Constructor: MetaLearningEngine(sqlite, config)
    - sqlite: for querying contamination_log, confidence_log, wisdom, validation_events
    - config: for threshold reference
    - No vector store needed — all analysis is over structured log data
    """

    def __init__(self, sqlite: SQLiteStore, config: WisdomConfig):
        self.sqlite = sqlite
        self.config = config

    def failure_profiles(self) -> list[FailureProfile]:
        """Compute failure rates by wisdom_type and creation_method.

        Returns FailureProfiles sorted by failure_rate descending.
        Profiles with zero wisdom entries are excluded.
        """
        type_lifecycle = self.sqlite.count_wisdom_by_type_and_lifecycle()
        deprecated_profiles = self.sqlite.get_deprecated_wisdom_profiles()

        # Aggregate by type
        type_totals: dict[str, int] = {}
        type_deprecated: dict[str, int] = {}
        method_totals: dict[str, int] = {}
        method_deprecated: dict[str, int] = {}

        for row in type_lifecycle:
            wtype = row["type"]
            method = row["creation_method"]
            count = row["count"]

            type_totals[wtype] = type_totals.get(wtype, 0) + count
            method_totals[method] = method_totals.get(method, 0) + count

            if row["lifecycle"] == "deprecated":
                type_deprecated[wtype] = type_deprecated.get(wtype, 0) + count
                method_deprecated[method] = method_deprecated.get(method, 0) + count

        # Count contamination events per type/method
        contam_by_source = self.sqlite.count_contamination_by_source(limit=1000)
        type_contam: dict[str, int] = {}
        method_contam: dict[str, int] = {}
        for cs in contam_by_source:
            w = self.sqlite.get_wisdom(cs["source_wisdom_id"])
            if w:
                wtype = w.type.value
                method = w.creation_method.value
                type_contam[wtype] = type_contam.get(wtype, 0) + cs["total_events"]
                method_contam[method] = method_contam.get(method, 0) + cs["total_events"]

        profiles: list[FailureProfile] = []

        for wtype, total in type_totals.items():
            dep = type_deprecated.get(wtype, 0)
            contam = type_contam.get(wtype, 0)
            profiles.append(FailureProfile(
                category=f"type:{wtype}",
                total_count=total,
                deprecated_count=dep,
                contamination_events=contam,
                failure_rate=dep / total if total > 0 else 0.0,
                contamination_rate=contam / total if total > 0 else 0.0,
            ))

        for method, total in method_totals.items():
            dep = method_deprecated.get(method, 0)
            contam = method_contam.get(method, 0)
            profiles.append(FailureProfile(
                category=f"method:{method}",
                total_count=total,
                deprecated_count=dep,
                contamination_events=contam,
                failure_rate=dep / total if total > 0 else 0.0,
                contamination_rate=contam / total if total > 0 else 0.0,
            ))

        profiles.sort(key=lambda p: p.failure_rate, reverse=True)
        return profiles

    def domain_risk_assessment(self) -> list[dict]:
        """Which domains have the most contamination and deprecation?

        Returns list of dicts sorted by risk_score descending.
        """
        deprecated = self.sqlite.get_deprecated_wisdom_profiles()
        contam_sources = self.sqlite.count_contamination_by_source(limit=1000)

        # Count deprecations per domain
        domain_dep: dict[str, int] = {}
        for dp in deprecated:
            for d in dp["domains"]:
                domain_dep[d] = domain_dep.get(d, 0) + 1

        # Count contamination events per domain
        domain_contam: dict[str, int] = {}
        for cs in contam_sources:
            w = self.sqlite.get_wisdom(cs["source_wisdom_id"])
            if w:
                for d in w.applicable_domains:
                    domain_contam[d] = domain_contam.get(d, 0) + cs["total_events"]

        # Get active wisdom counts per domain
        all_domains = self.sqlite.get_all_domains()
        results = []
        for domain in all_domains:
            active_count = self.sqlite.count_wisdom(domain=domain)
            dep_count = domain_dep.get(domain, 0)
            contam_count = domain_contam.get(domain, 0)
            total_count = active_count + dep_count

            # Risk score: weighted combination of deprecation rate and contamination density
            dep_rate = dep_count / total_count if total_count > 0 else 0.0
            contam_density = contam_count / total_count if total_count > 0 else 0.0
            risk_score = 0.6 * dep_rate + 0.4 * min(1.0, contam_density)

            results.append({
                "domain": domain,
                "active_wisdom": active_count,
                "deprecated_count": dep_count,
                "contamination_events": contam_count,
                "risk_score": risk_score,
            })

        results.sort(key=lambda r: r["risk_score"], reverse=True)
        return results

    def compute_risk_score(self, wisdom_id: str) -> RiskScore:
        """Compute historical risk for a specific wisdom entry.

        Factors:
        1. Type risk: failure rate for this wisdom_type
        2. Method risk: failure rate for this creation_method
        3. Domain risk: contamination density in its domains
        4. Validation risk: unvalidated entries historically fail more
        5. Application risk: untested wisdom has higher risk
        6. Volatility risk: erratic confidence history signals instability
        """
        w = self.sqlite.get_wisdom(wisdom_id)
        if not w:
            return RiskScore(
                wisdom_id=wisdom_id,
                base_risk=0.5,
                risk_factors=[{"name": "unknown", "value": 0.5, "reason": "Wisdom not found"}],
                recommended_challenge_level="elevated",
            )

        profiles = self.failure_profiles()
        risk_factors = []

        # Factor 1: Type risk
        type_key = f"type:{w.type.value}"
        type_risk = 0.0
        for p in profiles:
            if p.category == type_key:
                type_risk = p.failure_rate
                break
        risk_factors.append({
            "name": "type_risk",
            "value": type_risk,
            "reason": f"{w.type.value} has {type_risk:.0%} historical failure rate",
        })

        # Factor 2: Method risk
        method_key = f"method:{w.creation_method.value}"
        method_risk = 0.0
        for p in profiles:
            if p.category == method_key:
                method_risk = p.failure_rate
                break
        risk_factors.append({
            "name": "method_risk",
            "value": method_risk,
            "reason": f"{w.creation_method.value} has {method_risk:.0%} historical failure rate",
        })

        # Factor 3: Domain risk
        domain_risks = self.domain_risk_assessment()
        domain_risk = 0.0
        if w.applicable_domains:
            domain_risk_values = []
            for d in w.applicable_domains:
                for dr in domain_risks:
                    if dr["domain"] == d:
                        domain_risk_values.append(dr["risk_score"])
                        break
            if domain_risk_values:
                domain_risk = max(domain_risk_values)
        risk_factors.append({
            "name": "domain_risk",
            "value": domain_risk,
            "reason": f"Highest domain risk: {domain_risk:.2f}",
        })

        # Factor 4: Validation risk
        events = self.sqlite.get_validation_events(wisdom_id)
        has_external = any(
            e["source"] in ("external", "peer", "adversarial")
            and e["verdict"] in ("confirmed", "confirmed_with_caveats")
            for e in events
        )
        validation_risk = 0.0 if has_external else 0.4
        risk_factors.append({
            "name": "validation_risk",
            "value": validation_risk,
            "reason": "Validated externally" if has_external else "No external validation",
        })

        # Factor 5: Application risk (untested = higher risk)
        if w.application_count == 0:
            app_risk = 0.5
        elif w.application_count < 3:
            app_risk = 0.3
        else:
            app_risk = max(0.0, 0.2 - w.application_count * 0.01)
        risk_factors.append({
            "name": "application_risk",
            "value": app_risk,
            "reason": f"{w.application_count} applications",
        })

        # Factor 6: Volatility risk (erratic confidence = unreliable)
        volatility_risk = self._compute_volatility_risk(wisdom_id)
        risk_factors.append({
            "name": "volatility_risk",
            "value": volatility_risk,
            "reason": f"Confidence volatility: {volatility_risk:.2f}",
        })

        # Weighted combination
        weights = {
            "type_risk": 0.20,
            "method_risk": 0.15,
            "domain_risk": 0.15,
            "validation_risk": 0.20,
            "application_risk": 0.15,
            "volatility_risk": 0.15,
        }
        base_risk = sum(
            weights.get(f["name"], 0.0) * f["value"]
            for f in risk_factors
        )
        base_risk = max(0.0, min(1.0, base_risk))

        # Determine challenge level
        if base_risk > 0.6:
            level = "maximum"
        elif base_risk > 0.3:
            level = "elevated"
        else:
            level = "standard"

        return RiskScore(
            wisdom_id=wisdom_id,
            base_risk=base_risk,
            risk_factors=risk_factors,
            recommended_challenge_level=level,
        )

    def _compute_volatility_risk(self, wisdom_id: str) -> float:
        """Compute volatility risk for a single wisdom entry.

        Returns 0.0 (stable) to 1.0 (highly erratic). Entries with
        fewer than 2 confidence events get 0.0 (no evidence of instability).
        """
        history = self.sqlite.get_confidence_history("wisdom", wisdom_id)
        if len(history) < 2:
            return 0.0

        deltas = [e["new_confidence"] - e["old_confidence"] for e in history]
        direction_changes = sum(
            1 for i in range(1, len(deltas))
            if deltas[i] * deltas[i - 1] < 0
        )
        reversal_rate = direction_changes / (len(deltas) - 1)
        avg_magnitude = sum(abs(d) for d in deltas) / len(deltas)
        magnitude_factor = min(1.0, avg_magnitude / 0.1)

        return 0.6 * reversal_rate + 0.4 * magnitude_factor

    def contamination_patterns(self, limit: int = 10) -> list[ContaminationPattern]:
        """Find the worst 'super-spreader' contamination events.

        Returns sources sorted by total_affected descending.
        """
        sources = self.sqlite.count_contamination_by_source(limit=limit)
        patterns = []
        for s in sources:
            patterns.append(ContaminationPattern(
                source_wisdom_id=s["source_wisdom_id"],
                total_affected=s["total_events"],
                avg_penalty=s["avg_penalty"],
                affected_types={
                    "wisdom": s["wisdom_affected"],
                    "knowledge": s["knowledge_affected"],
                    "experience": s["experience_affected"],
                },
            ))
        return patterns

    def confidence_trajectory(self) -> dict:
        """Aggregate confidence trend for all wisdom entries.

        Returns net direction, common decrease reasons, and overall health.
        """
        stats = self.sqlite.get_confidence_change_stats()
        decrease_reasons = self.sqlite.get_most_common_confidence_decrease_reasons(limit=5)

        total = stats["total_events"]
        if total == 0:
            return {
                "total_events": 0,
                "avg_delta": 0.0,
                "positive_changes": 0,
                "negative_changes": 0,
                "net_direction": "no_data",
                "top_decrease_reasons": [],
            }

        net_direction = "improving" if stats["avg_delta"] > 0 else "declining"
        if abs(stats["avg_delta"]) < 0.001:
            net_direction = "stable"

        return {
            "total_events": total,
            "avg_delta": stats["avg_delta"],
            "positive_changes": stats["positive_changes"],
            "negative_changes": stats["negative_changes"],
            "net_direction": net_direction,
            "top_decrease_reasons": decrease_reasons,
        }

    def learning_velocity(self) -> list[VelocityProfile]:
        """Compute how quickly different categories of wisdom mature.

        Groups by type and domain, computing average confidence velocity
        (delta per day) and activity rate (events per day). Categories with
        no confidence events are excluded.

        A positive velocity means entries in that category tend to gain
        confidence over time. Negative means they tend to decline.
        """
        histories = self.sqlite.get_wisdom_confidence_histories()
        creation_dates = self.sqlite.get_wisdom_creation_dates()

        if not histories:
            return []

        # Compute per-entry velocity
        entry_velocities: dict[str, dict] = {}  # wisdom_id -> {velocity, events_per_day}
        for wid, events in histories.items():
            created_str = creation_dates.get(wid)
            if not created_str or len(events) < 1:
                continue

            created = _parse_timestamp(created_str)
            last_ts = _parse_timestamp(events[-1]["timestamp"])
            span_days = max((last_ts - created).total_seconds() / 86400, 0.01)

            net_delta = events[-1]["new_confidence"] - events[0]["old_confidence"]
            entry_velocities[wid] = {
                "velocity": net_delta / span_days,
                "events_per_day": len(events) / span_days,
            }

        if not entry_velocities:
            return []

        # Group by type and domain
        all_wisdom_entries = {
            wid: self.sqlite.get_wisdom(wid) for wid in entry_velocities
        }

        category_entries: dict[str, list[str]] = {}
        for wid, w in all_wisdom_entries.items():
            if not w:
                continue
            type_key = f"type:{w.type.value}"
            category_entries.setdefault(type_key, []).append(wid)
            for d in w.applicable_domains:
                domain_key = f"domain:{d}"
                category_entries.setdefault(domain_key, []).append(wid)

        profiles: list[VelocityProfile] = []
        for cat, wids in category_entries.items():
            velocities = [entry_velocities[wid] for wid in wids if wid in entry_velocities]
            if not velocities:
                continue

            avg_vel = sum(v["velocity"] for v in velocities) / len(velocities)
            avg_epd = sum(v["events_per_day"] for v in velocities) / len(velocities)

            fastest = max(wids, key=lambda w: entry_velocities.get(w, {}).get("velocity", 0))
            slowest = min(wids, key=lambda w: entry_velocities.get(w, {}).get("velocity", 0))

            profiles.append(VelocityProfile(
                category=cat,
                entry_count=len(velocities),
                avg_velocity=avg_vel,
                avg_events_per_day=avg_epd,
                fastest_id=fastest if fastest != slowest else None,
                slowest_id=slowest if fastest != slowest else None,
            ))

        profiles.sort(key=lambda p: p.avg_velocity, reverse=True)
        return profiles

    def confidence_volatility(self, limit: int = 10) -> list[VolatilityEntry]:
        """Find wisdom entries with the most erratic confidence histories.

        Volatility measures instability: entries where confidence bounces
        up and down rather than moving steadily in one direction. High
        volatility means the system can't make up its mind — a sign the
        wisdom is contextually ambiguous or sitting at the boundary of
        correctness.

        Returns top N entries sorted by volatility_score descending.
        Entries with fewer than 2 confidence events are excluded.
        """
        histories = self.sqlite.get_wisdom_confidence_histories()
        if not histories:
            return []

        entries: list[VolatilityEntry] = []
        for wid, events in histories.items():
            if len(events) < 2:
                continue

            deltas = [e["new_confidence"] - e["old_confidence"] for e in events]

            # Direction changes: how many times the sign flips
            direction_changes = 0
            for i in range(1, len(deltas)):
                if deltas[i] * deltas[i - 1] < 0:  # sign change
                    direction_changes += 1

            avg_magnitude = sum(abs(d) for d in deltas) / len(deltas)
            max_swing = max(abs(d) for d in deltas)

            # Volatility score: combines reversal frequency with magnitude
            # Reversals normalized by opportunity (events - 1)
            reversal_rate = direction_changes / (len(deltas) - 1) if len(deltas) > 1 else 0
            # Scale magnitude to 0-1 (0.1 delta is significant for confidence)
            magnitude_factor = min(1.0, avg_magnitude / 0.1)
            volatility_score = 0.6 * reversal_rate + 0.4 * magnitude_factor

            w = self.sqlite.get_wisdom(wid)
            preview = w.statement[:60] if w else "(deleted)"

            entries.append(VolatilityEntry(
                wisdom_id=wid,
                statement_preview=preview,
                total_events=len(events),
                direction_changes=direction_changes,
                avg_delta_magnitude=avg_magnitude,
                max_swing=max_swing,
                volatility_score=volatility_score,
            ))

        entries.sort(key=lambda e: e.volatility_score, reverse=True)
        return entries[:limit]

    def risk_profile_for_adversarial(self, wisdom_id: str) -> dict | None:
        """Build a risk profile dict suitable for passing to AdversarialEngine.challenge().

        Returns None if the risk is standard (no adjustment needed).
        The adversarial engine has zero knowledge of this engine — it just
        accepts an optional dict of threshold overrides.
        """
        risk = self.compute_risk_score(wisdom_id)
        if risk.recommended_challenge_level == "standard":
            return None

        profile: dict = {"risk_level": risk.recommended_challenge_level}

        if risk.recommended_challenge_level == "maximum":
            profile["counterexample_threshold"] = 0.5  # lower = search harder
            profile["blind_spot_frequency"] = 0.2  # lower = more suspicious
        elif risk.recommended_challenge_level == "elevated":
            profile["counterexample_threshold"] = 0.6
            profile["blind_spot_frequency"] = 0.25

        return profile

    def summary(self) -> dict:
        """Complete meta-learning summary for CLI display.

        Returns dict combining:
        - Top riskiest profiles
        - Top riskiest domains
        - Top super-spreaders
        - Overall confidence trajectory
        - Learning velocity by category
        - Most volatile entries
        """
        profiles = self.failure_profiles()
        domain_risks = self.domain_risk_assessment()
        patterns = self.contamination_patterns(limit=5)
        trajectory = self.confidence_trajectory()
        velocity = self.learning_velocity()
        volatility = self.confidence_volatility(limit=5)

        # Top risky profiles (failure_rate > 0)
        risky_profiles = [p for p in profiles if p.failure_rate > 0][:5]

        # Top risky domains (risk_score > 0)
        risky_domains = [d for d in domain_risks if d["risk_score"] > 0][:5]

        return {
            "failure_profiles": risky_profiles,
            "risky_domains": risky_domains,
            "super_spreaders": patterns,
            "trajectory": trajectory,
            "velocity": velocity,
            "volatility": volatility,
            "total_profiles_analyzed": len(profiles),
            "total_domains_analyzed": len(domain_risks),
        }
