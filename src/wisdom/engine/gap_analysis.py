"""Gap analysis engine — blind spot detection and learning priorities."""

from __future__ import annotations

from wisdom.logging_config import get_logger
from wisdom.models.common import LifecycleState
from wisdom.storage.sqlite_store import SQLiteStore

logger = get_logger("engine.gaps")


class GapAnalysisEngine:
    """Detect wisdom gaps and suggest learning priorities."""

    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    def find_wisdom_gaps(self, domain: str | None = None) -> list[dict]:
        """Find domains with many experiences but little distilled wisdom."""
        domains = [domain] if domain else self.sqlite.get_all_domains()
        gaps = []

        for d in domains:
            exp_count = self.sqlite.count_experiences(domain=d)
            know_count = self.sqlite.count_knowledge(domain=d)
            wis_count = self.sqlite.count_wisdom(domain=d)

            if exp_count == 0:
                continue

            # Ratio of wisdom to experiences — low ratio = gap
            wisdom_ratio = wis_count / exp_count if exp_count > 0 else 0
            knowledge_ratio = know_count / exp_count if exp_count > 0 else 0

            if wisdom_ratio < 0.1 and exp_count >= 3:
                gaps.append({
                    "domain": d,
                    "experiences": exp_count,
                    "knowledge": know_count,
                    "wisdom": wis_count,
                    "wisdom_ratio": round(wisdom_ratio, 3),
                    "knowledge_ratio": round(knowledge_ratio, 3),
                    "severity": "high" if exp_count >= 10 and wis_count == 0 else "medium",
                    "suggestion": self._suggest_action(exp_count, know_count, wis_count),
                })

        # Sort by severity then experience count
        gaps.sort(key=lambda g: (0 if g["severity"] == "high" else 1, -g["experiences"]))
        return gaps

    def find_low_coverage_tasks(self) -> list[dict]:
        """Find task types frequently seen in experiences but with no applicable wisdom."""
        # Get all distinct task types from experiences
        rows = self.sqlite.conn.execute(
            "SELECT task_type, COUNT(*) as cnt FROM experiences WHERE task_type != '' GROUP BY task_type ORDER BY cnt DESC"
        ).fetchall()

        low_coverage = []
        for row in rows:
            task_type = row[0]
            count = row[1]
            # Check if any wisdom mentions this task type
            wis_count = self.sqlite.conn.execute(
                "SELECT COUNT(*) FROM wisdom WHERE statement LIKE ? OR applicability_conditions LIKE ?",
                (f"%{task_type}%", f"%{task_type}%"),
            ).fetchone()[0]

            if wis_count == 0 and count >= 2:
                low_coverage.append({
                    "task_type": task_type,
                    "experience_count": count,
                    "wisdom_count": wis_count,
                })

        return low_coverage

    def find_stale_domains(self, stale_days: int = 90) -> list[dict]:
        """Find domains where all wisdom is old and unreinforced."""
        domains = self.sqlite.get_all_domains()
        stale = []

        for d in domains:
            wisdom_entries = self.sqlite.list_wisdom(domain=d, limit=1000)
            if not wisdom_entries:
                continue

            active_wisdom = [
                w for w in wisdom_entries
                if w.lifecycle != LifecycleState.DEPRECATED
            ]
            if not active_wisdom:
                continue

            # Check if all active wisdom is old
            all_stale = all(w.age_days > stale_days for w in active_wisdom)
            if all_stale:
                avg_age = sum(w.age_days for w in active_wisdom) / len(active_wisdom)
                avg_confidence = sum(w.confidence.overall for w in active_wisdom) / len(active_wisdom)
                stale.append({
                    "domain": d,
                    "active_wisdom_count": len(active_wisdom),
                    "avg_age_days": round(avg_age, 1),
                    "avg_confidence": round(avg_confidence, 3),
                    "oldest_days": round(max(w.age_days for w in active_wisdom), 1),
                })

        stale.sort(key=lambda s: -s["avg_age_days"])
        return stale

    def suggest_next_extraction(self, limit: int = 5) -> list[dict]:
        """Recommend which unprocessed experiences to extract knowledge from next."""
        # Prioritize domains with many unprocessed experiences
        domains = self.sqlite.get_all_domains()
        suggestions = []

        for d in domains:
            unprocessed = self.sqlite.count_experiences(domain=d, unprocessed=True)
            if unprocessed == 0:
                continue
            wis_count = self.sqlite.count_wisdom(domain=d)

            # Priority: more unprocessed + fewer wisdom = higher priority
            priority = unprocessed * (1.0 / (1.0 + wis_count))
            suggestions.append({
                "domain": d,
                "unprocessed_experiences": unprocessed,
                "existing_wisdom": wis_count,
                "priority_score": round(priority, 2),
            })

        # Also check domain-less
        unprocessed_general = self.sqlite.count_experiences(unprocessed=True)
        if unprocessed_general > 0:
            suggestions.append({
                "domain": "(general)",
                "unprocessed_experiences": unprocessed_general,
                "existing_wisdom": 0,
                "priority_score": float(unprocessed_general),
            })

        suggestions.sort(key=lambda s: -s["priority_score"])
        return suggestions[:limit]

    def _suggest_action(self, exp_count: int, know_count: int, wis_count: int) -> str:
        if know_count == 0:
            return "Run knowledge extraction first"
        if wis_count == 0 and know_count > 0:
            return "Run wisdom synthesis"
        if wis_count > 0:
            return "Consider adding more experiences or reinforcing existing wisdom"
        return "Add more experiences"

    def summary(self, domain: str | None = None) -> dict:
        """Get a complete gap analysis summary."""
        return {
            "gaps": self.find_wisdom_gaps(domain),
            "low_coverage_tasks": self.find_low_coverage_tasks(),
            "stale_domains": self.find_stale_domains(),
            "extraction_suggestions": self.suggest_next_extraction(),
        }
