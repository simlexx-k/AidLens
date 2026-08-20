from functools import lru_cache

from app.services.ranker.serving import AidRankerService


@lru_cache(maxsize=8)
def get_aidranker_service(
    model_name_or_path: str,
    candidate_k: int,
    batch_size: int,
    device: str,
    fail_open: bool,
) -> AidRankerService:
    """Return the process-local AidRanker singleton for one serving configuration."""

    return AidRankerService(
        model_name_or_path,
        candidate_k=candidate_k,
        batch_size=batch_size,
        device=device,
        fail_open=fail_open,
    )
