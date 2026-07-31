from nimbus_cache import CacheClient


cache = CacheClient()


def load_profile(user_id: str) -> dict | None:
    return cache.get(f"profile:{user_id}")
