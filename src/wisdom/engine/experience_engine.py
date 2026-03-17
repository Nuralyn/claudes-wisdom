"""Experience engine — manages the raw experience layer."""

from __future__ import annotations

from wisdom.logging_config import get_logger
from wisdom.models.common import ExperienceResult, ExperienceType
from wisdom.models.experience import Experience
from wisdom.storage.sqlite_store import SQLiteStore
from wisdom.storage.vector_store import VectorStore

logger = get_logger("engine.experience")


class ExperienceEngine:
    """Add, search, and manage experiences."""

    def __init__(self, sqlite: SQLiteStore, vector: VectorStore):
        self.sqlite = sqlite
        self.vector = vector

    def add(
        self,
        description: str,
        domain: str = "",
        subdomain: str = "",
        task_type: str = "",
        input_text: str = "",
        output_text: str = "",
        result: ExperienceResult = ExperienceResult.SUCCESS,
        quality_score: float = 0.5,
        exp_type: ExperienceType = ExperienceType.TASK,
        tags: list[str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Experience:
        """Record a new experience."""
        exp = Experience(
            type=exp_type,
            domain=domain,
            subdomain=subdomain,
            task_type=task_type,
            description=description,
            input_text=input_text,
            output_text=output_text,
            result=result,
            quality_score=quality_score,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.sqlite.save_experience(exp)
        self.vector.add(
            layer="experience",
            id=exp.id,
            text=exp.embedding_text,
            metadata={"domain": domain, "type": exp_type.value, "result": result.value},
        )
        logger.info("Added experience %s [%s] in domain '%s'", exp.id, exp_type.value, domain)
        return exp

    def get(self, id: str) -> Experience | None:
        return self.sqlite.get_experience(id)

    def list(self, domain: str | None = None, limit: int = 100, offset: int = 0) -> list[Experience]:
        return self.sqlite.list_experiences(domain=domain, limit=limit, offset=offset)

    def search(self, query: str, top_k: int = 10, domain: str | None = None) -> list[dict]:
        """Semantic search over experiences. Returns list of {experience, similarity}."""
        where = {"domain": domain} if domain else None
        results = self.vector.search(layer="experience", query=query, top_k=top_k, where=where)
        enriched = []
        for r in results:
            exp = self.sqlite.get_experience(r["id"])
            if exp:
                enriched.append({"experience": exp, "similarity": r["similarity"]})
        return enriched

    def delete(self, id: str) -> bool:
        self.vector.delete(layer="experience", id=id)
        return self.sqlite.delete_experience(id)

    def count(self, domain: str | None = None, unprocessed: bool = False) -> int:
        return self.sqlite.count_experiences(domain=domain, unprocessed=unprocessed)

    def get_unprocessed(self, domain: str | None = None, limit: int = 50) -> list[Experience]:
        return self.sqlite.get_unprocessed(domain=domain, limit=limit)

    def stats(self) -> dict:
        """Get experience statistics."""
        total = self.sqlite.count_experiences()
        unprocessed = self.sqlite.count_experiences(unprocessed=True)
        domains = {}
        for domain in self.sqlite.get_all_domains():
            domains[domain] = self.sqlite.count_experiences(domain=domain)
        return {
            "total": total,
            "unprocessed": unprocessed,
            "processed": total - unprocessed,
            "domains": domains,
        }
