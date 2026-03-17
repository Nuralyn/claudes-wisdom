"""SQLite storage backend for all DIKW layers."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from wisdom.logging_config import get_logger
from wisdom.models.common import (
    ConfidenceScore,
    Relationship,
    RelationshipType,
)
from wisdom.models.experience import Experience
from wisdom.models.knowledge import Knowledge
from wisdom.models.wisdom import Wisdom

logger = get_logger("storage.sqlite")

SCHEMA_VERSION = 2

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiences (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    type TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT '',
    subdomain TEXT NOT NULL DEFAULT '',
    task_type TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL,
    input_text TEXT NOT NULL DEFAULT '',
    output_text TEXT NOT NULL DEFAULT '',
    result TEXT NOT NULL DEFAULT 'success',
    quality_score REAL NOT NULL DEFAULT 0.5,
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    processed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_exp_domain ON experiences(domain);
CREATE INDEX IF NOT EXISTS idx_exp_type ON experiences(type);
CREATE INDEX IF NOT EXISTS idx_exp_processed ON experiences(processed);
CREATE INDEX IF NOT EXISTS idx_exp_timestamp ON experiences(timestamp);

CREATE TABLE IF NOT EXISTS knowledge (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'pattern',
    statement TEXT NOT NULL,
    explanation TEXT NOT NULL DEFAULT '',
    preconditions TEXT NOT NULL DEFAULT '[]',
    postconditions TEXT NOT NULL DEFAULT '[]',
    domain TEXT NOT NULL DEFAULT '',
    subdomain TEXT NOT NULL DEFAULT '',
    specificity REAL NOT NULL DEFAULT 0.5,
    confidence TEXT NOT NULL DEFAULT '{}',
    supporting_count INTEGER NOT NULL DEFAULT 0,
    contradicting_count INTEGER NOT NULL DEFAULT 0,
    source_experience_ids TEXT NOT NULL DEFAULT '[]',
    validation_status TEXT NOT NULL DEFAULT 'unvalidated',
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    synthesized INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_know_domain ON knowledge(domain);
CREATE INDEX IF NOT EXISTS idx_know_type ON knowledge(type);
CREATE INDEX IF NOT EXISTS idx_know_synthesized ON knowledge(synthesized);

CREATE TABLE IF NOT EXISTS wisdom (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'principle',
    statement TEXT NOT NULL,
    reasoning TEXT NOT NULL DEFAULT '',
    implications TEXT NOT NULL DEFAULT '[]',
    counterexamples TEXT NOT NULL DEFAULT '[]',
    applicable_domains TEXT NOT NULL DEFAULT '[]',
    applicability_conditions TEXT NOT NULL DEFAULT '[]',
    inapplicability_conditions TEXT NOT NULL DEFAULT '[]',
    trade_offs TEXT NOT NULL DEFAULT '[]',
    confidence TEXT NOT NULL DEFAULT '{}',
    lifecycle TEXT NOT NULL DEFAULT 'emerging',
    version INTEGER NOT NULL DEFAULT 1,
    source_knowledge_ids TEXT NOT NULL DEFAULT '[]',
    relationships TEXT NOT NULL DEFAULT '[]',
    application_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    deprecation_reason TEXT NOT NULL DEFAULT '',
    creation_method TEXT NOT NULL DEFAULT 'pipeline',
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_wis_lifecycle ON wisdom(lifecycle);
CREATE INDEX IF NOT EXISTS idx_wis_domains ON wisdom(applicable_domains);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    relationship TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, relationship)
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id, source_type);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id, target_type);

CREATE TABLE IF NOT EXISTS confidence_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    old_confidence REAL NOT NULL,
    new_confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_clog_entity ON confidence_log(entity_type, entity_id);

CREATE TABLE IF NOT EXISTS validation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wisdom_id TEXT NOT NULL,
    source TEXT NOT NULL,
    verdict TEXT NOT NULL,
    evidence TEXT NOT NULL DEFAULT '',
    validator TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_val_wisdom ON validation_events(wisdom_id);

CREATE TABLE IF NOT EXISTS contamination_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_wisdom_id TEXT NOT NULL,
    affected_entity_id TEXT NOT NULL,
    affected_entity_type TEXT NOT NULL,
    penalty_applied REAL NOT NULL,
    reason TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_contam_source ON contamination_log(source_wisdom_id);
CREATE INDEX IF NOT EXISTS idx_contam_affected ON contamination_log(affected_entity_id);
"""


class SQLiteStore:
    """SQLite storage for experiences, knowledge, wisdom, and relationships."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cursor = self.conn.cursor()
        # Check if schema_version table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        if cursor.fetchone() is None:
            cursor.executescript(SCHEMA_SQL)
            cursor.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            self.conn.commit()
            logger.info("Initialized database schema v%d", SCHEMA_VERSION)
        else:
            self._run_migrations()

    def _run_migrations(self) -> None:
        """Apply any pending schema migrations."""
        row = self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current = row[0] if row else 0

        if current < 2:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS validation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wisdom_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    evidence TEXT NOT NULL DEFAULT '',
                    validator TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_val_wisdom ON validation_events(wisdom_id);

                CREATE TABLE IF NOT EXISTS contamination_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_wisdom_id TEXT NOT NULL,
                    affected_entity_id TEXT NOT NULL,
                    affected_entity_type TEXT NOT NULL,
                    penalty_applied REAL NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_contam_source ON contamination_log(source_wisdom_id);
                CREATE INDEX IF NOT EXISTS idx_contam_affected ON contamination_log(affected_entity_id);
            """)
            self.conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (2,)
            )
            self.conn.commit()
            logger.info("Migrated schema to v2 (validation_events, contamination_log)")

    def close(self) -> None:
        self.conn.close()

    # ── Experience CRUD ────────────────────────────────────────────────────

    def save_experience(self, exp: Experience) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO experiences
            (id, timestamp, type, domain, subdomain, task_type, description,
             input_text, output_text, result, quality_score, tags, metadata, processed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                exp.id, exp.timestamp, exp.type.value, exp.domain, exp.subdomain,
                exp.task_type, exp.description, exp.input_text, exp.output_text,
                exp.result.value, exp.quality_score, json.dumps(exp.tags),
                json.dumps(exp.metadata), int(exp.processed),
            ),
        )
        self.conn.commit()

    def get_experience(self, id: str) -> Experience | None:
        row = self.conn.execute("SELECT * FROM experiences WHERE id = ?", (id,)).fetchone()
        if row is None:
            return None
        return self._row_to_experience(row)

    def list_experiences(
        self, domain: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[Experience]:
        if domain:
            rows = self.conn.execute(
                "SELECT * FROM experiences WHERE domain = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (domain, limit, offset),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM experiences ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_experience(r) for r in rows]

    def delete_experience(self, id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM experiences WHERE id = ?", (id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def count_experiences(self, domain: str | None = None, unprocessed: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM experiences WHERE 1=1"
        params: list[str | int] = []
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        if unprocessed:
            sql += " AND processed = 0"
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else 0

    def mark_processed(self, ids: list[str]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(
            f"UPDATE experiences SET processed = 1 WHERE id IN ({placeholders})", ids
        )
        self.conn.commit()

    def get_unprocessed(self, domain: str | None = None, limit: int = 50) -> list[Experience]:
        if domain:
            rows = self.conn.execute(
                "SELECT * FROM experiences WHERE processed = 0 AND domain = ? ORDER BY timestamp ASC LIMIT ?",
                (domain, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM experiences WHERE processed = 0 ORDER BY timestamp ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_experience(r) for r in rows]

    def _row_to_experience(self, row: sqlite3.Row) -> Experience:
        return Experience(
            id=row["id"],
            timestamp=row["timestamp"],
            type=row["type"],
            domain=row["domain"],
            subdomain=row["subdomain"],
            task_type=row["task_type"],
            description=row["description"],
            input_text=row["input_text"],
            output_text=row["output_text"],
            result=row["result"],
            quality_score=row["quality_score"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            processed=bool(row["processed"]),
        )

    # ── Knowledge CRUD ─────────────────────────────────────────────────────

    def save_knowledge(self, k: Knowledge) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO knowledge
            (id, created_at, updated_at, type, statement, explanation,
             preconditions, postconditions, domain, subdomain, specificity,
             confidence, supporting_count, contradicting_count,
             source_experience_ids, validation_status, tags, metadata, synthesized)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                k.id, k.created_at, k.updated_at, k.type.value,
                k.statement, k.explanation,
                json.dumps(k.preconditions), json.dumps(k.postconditions),
                k.domain, k.subdomain, k.specificity,
                k.confidence.model_dump_json(),
                k.supporting_count, k.contradicting_count,
                json.dumps(k.source_experience_ids),
                k.validation_status.value,
                json.dumps(k.tags), json.dumps(k.metadata),
                int(k.synthesized),
            ),
        )
        self.conn.commit()

    def get_knowledge(self, id: str) -> Knowledge | None:
        row = self.conn.execute("SELECT * FROM knowledge WHERE id = ?", (id,)).fetchone()
        if row is None:
            return None
        return self._row_to_knowledge(row)

    def list_knowledge(
        self, domain: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[Knowledge]:
        if domain:
            rows = self.conn.execute(
                "SELECT * FROM knowledge WHERE domain = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (domain, limit, offset),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM knowledge ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_knowledge(r) for r in rows]

    def delete_knowledge(self, id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM knowledge WHERE id = ?", (id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def count_knowledge(self, domain: str | None = None, unsynthesized: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM knowledge WHERE 1=1"
        params: list[str | int] = []
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        if unsynthesized:
            sql += " AND synthesized = 0"
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else 0

    def mark_synthesized(self, ids: list[str]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(
            f"UPDATE knowledge SET synthesized = 1 WHERE id IN ({placeholders})", ids
        )
        self.conn.commit()

    def get_unsynthesized(self, domain: str | None = None, limit: int = 50) -> list[Knowledge]:
        if domain:
            rows = self.conn.execute(
                "SELECT * FROM knowledge WHERE synthesized = 0 AND domain = ? ORDER BY created_at ASC LIMIT ?",
                (domain, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM knowledge WHERE synthesized = 0 ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_knowledge(r) for r in rows]

    def _row_to_knowledge(self, row: sqlite3.Row) -> Knowledge:
        conf_data = json.loads(row["confidence"])
        return Knowledge(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            type=row["type"],
            statement=row["statement"],
            explanation=row["explanation"],
            preconditions=json.loads(row["preconditions"]),
            postconditions=json.loads(row["postconditions"]),
            domain=row["domain"],
            subdomain=row["subdomain"],
            specificity=row["specificity"],
            confidence=ConfidenceScore(**conf_data),
            supporting_count=row["supporting_count"],
            contradicting_count=row["contradicting_count"],
            source_experience_ids=json.loads(row["source_experience_ids"]),
            validation_status=row["validation_status"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            synthesized=bool(row["synthesized"]),
        )

    # ── Wisdom CRUD ────────────────────────────────────────────────────────

    def save_wisdom(self, w: Wisdom) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO wisdom
            (id, created_at, updated_at, type, statement, reasoning,
             implications, counterexamples, applicable_domains,
             applicability_conditions, inapplicability_conditions,
             trade_offs, confidence, lifecycle, version,
             source_knowledge_ids, relationships,
             application_count, success_count, failure_count,
             deprecation_reason, creation_method, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                w.id, w.created_at, w.updated_at, w.type.value,
                w.statement, w.reasoning,
                json.dumps(w.implications), json.dumps(w.counterexamples),
                json.dumps(w.applicable_domains),
                json.dumps(w.applicability_conditions),
                json.dumps(w.inapplicability_conditions),
                json.dumps([t.model_dump() for t in w.trade_offs]),
                w.confidence.model_dump_json(),
                w.lifecycle.value, w.version,
                json.dumps(w.source_knowledge_ids),
                json.dumps([r.model_dump() for r in w.relationships]),
                w.application_count, w.success_count, w.failure_count,
                w.deprecation_reason, w.creation_method.value,
                json.dumps(w.tags), json.dumps(w.metadata),
            ),
        )
        self.conn.commit()

    def update_wisdom(self, w: Wisdom) -> None:
        self.save_wisdom(w)

    def get_wisdom(self, id: str) -> Wisdom | None:
        row = self.conn.execute("SELECT * FROM wisdom WHERE id = ?", (id,)).fetchone()
        if row is None:
            return None
        return self._row_to_wisdom(row)

    def list_wisdom(
        self,
        domain: str | None = None,
        lifecycle: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Wisdom]:
        sql = "SELECT * FROM wisdom WHERE 1=1"
        params: list[str | int] = []
        if lifecycle:
            sql += " AND lifecycle = ?"
            params.append(lifecycle)
        if domain:
            sql += " AND applicable_domains LIKE ?"
            params.append(f'%"{domain}"%')
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_wisdom(r) for r in rows]

    def delete_wisdom(self, id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM wisdom WHERE id = ?", (id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def count_wisdom(self, domain: str | None = None) -> int:
        if domain:
            row = self.conn.execute(
                'SELECT COUNT(*) FROM wisdom WHERE applicable_domains LIKE ?',
                (f'%"{domain}"%',),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM wisdom").fetchone()
        return row[0] if row else 0

    def _row_to_wisdom(self, row: sqlite3.Row) -> Wisdom:
        from wisdom.models.common import TradeOff as TradeOffModel

        conf_data = json.loads(row["confidence"])
        trade_offs_data = json.loads(row["trade_offs"])
        relationships_data = json.loads(row["relationships"])
        return Wisdom(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            type=row["type"],
            statement=row["statement"],
            reasoning=row["reasoning"],
            implications=json.loads(row["implications"]),
            counterexamples=json.loads(row["counterexamples"]),
            applicable_domains=json.loads(row["applicable_domains"]),
            applicability_conditions=json.loads(row["applicability_conditions"]),
            inapplicability_conditions=json.loads(row["inapplicability_conditions"]),
            trade_offs=[TradeOffModel(**t) for t in trade_offs_data],
            confidence=ConfidenceScore(**conf_data),
            lifecycle=row["lifecycle"],
            version=row["version"],
            source_knowledge_ids=json.loads(row["source_knowledge_ids"]),
            relationships=[Relationship(**r) for r in relationships_data],
            application_count=row["application_count"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            deprecation_reason=row["deprecation_reason"],
            creation_method=row["creation_method"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
        )

    # ── Relationship CRUD ──────────────────────────────────────────────────

    def save_relationship(self, r: Relationship) -> None:
        rel_id = r.id or uuid.uuid4().hex[:16]
        ts = r.created_at or datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO relationships
            (id, source_id, source_type, target_id, target_type, relationship, strength, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (rel_id, r.source_id, r.source_type, r.target_id, r.target_type,
             r.relationship.value, r.strength, ts),
        )
        self.conn.commit()

    def get_relationships(
        self, entity_id: str, entity_type: str | None = None
    ) -> list[Relationship]:
        if entity_type:
            rows = self.conn.execute(
                """SELECT * FROM relationships
                WHERE (source_id = ? AND source_type = ?)
                   OR (target_id = ? AND target_type = ?)""",
                (entity_id, entity_type, entity_id, entity_type),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE source_id = ? OR target_id = ?",
                (entity_id, entity_id),
            ).fetchall()
        return [self._row_to_relationship(r) for r in rows]

    def find_conflicts(self, entity_id: str) -> list[Relationship]:
        rows = self.conn.execute(
            """SELECT * FROM relationships
            WHERE (source_id = ? OR target_id = ?) AND relationship = 'conflicts'""",
            (entity_id, entity_id),
        ).fetchall()
        return [self._row_to_relationship(r) for r in rows]

    def delete_relationship(self, id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM relationships WHERE id = ?", (id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        return Relationship(
            id=row["id"],
            source_id=row["source_id"],
            source_type=row["source_type"],
            target_id=row["target_id"],
            target_type=row["target_type"],
            relationship=RelationshipType(row["relationship"]),
            strength=row["strength"],
            created_at=row["created_at"],
        )

    # ── Confidence Log ─────────────────────────────────────────────────────

    def log_confidence_change(
        self,
        entity_type: str,
        entity_id: str,
        old_confidence: float,
        new_confidence: float,
        reason: str,
        details: str = "",
    ) -> None:
        self.conn.execute(
            """INSERT INTO confidence_log
            (entity_type, entity_id, old_confidence, new_confidence, reason, details)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (entity_type, entity_id, old_confidence, new_confidence, reason, details),
        )
        self.conn.commit()

    def get_confidence_history(
        self, entity_type: str, entity_id: str, limit: int = 50
    ) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM confidence_log
            WHERE entity_type = ? AND entity_id = ?
            ORDER BY id DESC LIMIT ?""",
            (entity_type, entity_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_events(self, since: str | None = None, limit: int = 50) -> list[dict]:
        if since:
            rows = self.conn.execute(
                "SELECT * FROM confidence_log WHERE timestamp >= ? ORDER BY id DESC LIMIT ?",
                (since, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM confidence_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Validation Events ────────────────────────────────────────────────

    def save_validation_event(
        self,
        wisdom_id: str,
        source: str,
        verdict: str,
        evidence: str = "",
        validator: str = "",
    ) -> None:
        """Record a validation event.

        source: 'self_report', 'peer', 'external', 'adversarial'
        verdict: 'confirmed', 'confirmed_with_caveats', 'challenged', 'refuted'
        """
        self.conn.execute(
            """INSERT INTO validation_events
            (wisdom_id, source, verdict, evidence, validator)
            VALUES (?, ?, ?, ?, ?)""",
            (wisdom_id, source, verdict, evidence, validator),
        )
        self.conn.commit()

    def get_validation_events(self, wisdom_id: str, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM validation_events WHERE wisdom_id = ? ORDER BY id DESC LIMIT ?",
            (wisdom_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_validations(self, wisdom_id: str, source: str | None = None) -> int:
        if source:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM validation_events WHERE wisdom_id = ? AND source = ?",
                (wisdom_id, source),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM validation_events WHERE wisdom_id = ?",
                (wisdom_id,),
            ).fetchone()
        return row[0] if row else 0

    # ── Contamination Log ──────────────────────────────────────────────────

    def log_contamination(
        self,
        source_wisdom_id: str,
        affected_entity_id: str,
        affected_entity_type: str,
        penalty_applied: float,
        reason: str,
    ) -> None:
        self.conn.execute(
            """INSERT INTO contamination_log
            (source_wisdom_id, affected_entity_id, affected_entity_type, penalty_applied, reason)
            VALUES (?, ?, ?, ?, ?)""",
            (source_wisdom_id, affected_entity_id, affected_entity_type, penalty_applied, reason),
        )
        self.conn.commit()

    def get_contamination_history(
        self, entity_id: str, limit: int = 50
    ) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM contamination_log
            WHERE affected_entity_id = ? OR source_wisdom_id = ?
            ORDER BY id DESC LIMIT ?""",
            (entity_id, entity_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def find_wisdom_sharing_knowledge(self, knowledge_ids: list[str]) -> list[Wisdom]:
        """Find all wisdom entries whose source_knowledge_ids overlap with the given set."""
        if not knowledge_ids:
            return []
        all_wisdom = self.list_wisdom(limit=100000)
        matching = []
        target_set = set(knowledge_ids)
        for w in all_wisdom:
            if set(w.source_knowledge_ids) & target_set:
                matching.append(w)
        return matching

    # ── Stats ──────────────────────────────────────────────────────────────

    def get_all_domains(self) -> list[str]:
        """Return all unique domains across all layers."""
        domains: set[str] = set()
        for row in self.conn.execute(
            "SELECT DISTINCT domain FROM experiences WHERE domain != ''"
        ).fetchall():
            domains.add(row[0])
        for row in self.conn.execute(
            "SELECT DISTINCT domain FROM knowledge WHERE domain != ''"
        ).fetchall():
            domains.add(row[0])
        # Wisdom stores domains as JSON arrays
        for row in self.conn.execute(
            "SELECT applicable_domains FROM wisdom"
        ).fetchall():
            for d in json.loads(row[0]):
                if d:
                    domains.add(d)
        return sorted(domains)

    # ── Meta-Learning Aggregates ─────────────────────────────────────────

    def count_wisdom_by_type_and_lifecycle(self) -> list[dict]:
        """Count wisdom grouped by (type, lifecycle, creation_method)."""
        rows = self.conn.execute(
            "SELECT type, lifecycle, creation_method, COUNT(*) as cnt "
            "FROM wisdom GROUP BY type, lifecycle, creation_method"
        ).fetchall()
        return [
            {"type": r[0], "lifecycle": r[1], "creation_method": r[2], "count": r[3]}
            for r in rows
        ]

    def count_contamination_by_source(self, limit: int = 20) -> list[dict]:
        """Top contamination sources by event count."""
        rows = self.conn.execute(
            "SELECT source_wisdom_id, COUNT(*) as cnt, "
            "AVG(penalty_applied) as avg_penalty, "
            "SUM(CASE WHEN affected_entity_type = 'wisdom' THEN 1 ELSE 0 END) as wisdom_cnt, "
            "SUM(CASE WHEN affected_entity_type = 'knowledge' THEN 1 ELSE 0 END) as knowledge_cnt, "
            "SUM(CASE WHEN affected_entity_type = 'experience' THEN 1 ELSE 0 END) as exp_cnt "
            "FROM contamination_log GROUP BY source_wisdom_id "
            "ORDER BY cnt DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "source_wisdom_id": r[0], "total_events": r[1],
                "avg_penalty": r[2], "wisdom_affected": r[3],
                "knowledge_affected": r[4], "experience_affected": r[5],
            }
            for r in rows
        ]

    def get_confidence_change_stats(self) -> dict:
        """Aggregate statistics over wisdom confidence changes."""
        row = self.conn.execute(
            "SELECT COUNT(*), "
            "AVG(new_confidence - old_confidence), "
            "SUM(CASE WHEN new_confidence > old_confidence THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN new_confidence < old_confidence THEN 1 ELSE 0 END) "
            "FROM confidence_log WHERE entity_type = 'wisdom'"
        ).fetchone()
        return {
            "total_events": row[0] or 0,
            "avg_delta": row[1] or 0.0,
            "positive_changes": row[2] or 0,
            "negative_changes": row[3] or 0,
        }

    def get_deprecated_wisdom_profiles(self) -> list[dict]:
        """Fetch type, creation_method, and domains for all deprecated wisdom."""
        rows = self.conn.execute(
            "SELECT id, type, creation_method, applicable_domains, confidence "
            "FROM wisdom WHERE lifecycle = 'deprecated'"
        ).fetchall()
        return [
            {
                "id": r[0], "type": r[1], "creation_method": r[2],
                "domains": json.loads(r[3]),
                "confidence": json.loads(r[4]).get("overall", 0.0),
            }
            for r in rows
        ]

    def get_contamination_count_for_wisdom(self, wisdom_id: str) -> int:
        """Count contamination events sourced from a specific wisdom entry."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM contamination_log WHERE source_wisdom_id = ?",
            (wisdom_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_most_common_confidence_decrease_reasons(self, limit: int = 5) -> list[dict]:
        """Find the most common reasons for confidence decreases."""
        rows = self.conn.execute(
            "SELECT reason, COUNT(*) as cnt, AVG(old_confidence - new_confidence) as avg_drop "
            "FROM confidence_log "
            "WHERE entity_type = 'wisdom' AND new_confidence < old_confidence "
            "GROUP BY reason ORDER BY cnt DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"reason": r[0], "count": r[1], "avg_drop": r[2]}
            for r in rows
        ]

    # ── Stats ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        exp_count = self.conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        know_count = self.conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
        wis_count = self.conn.execute("SELECT COUNT(*) FROM wisdom").fetchone()[0]
        rel_count = self.conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        return {
            "experiences": exp_count,
            "knowledge": know_count,
            "wisdom": wis_count,
            "relationships": rel_count,
            "domains": self.get_all_domains(),
        }
