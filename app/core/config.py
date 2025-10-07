import os
from functools import lru_cache


class AppConfig:
    def __init__(self) -> None:
        self.app_name: str = os.getenv("APP_NAME", "Tree Evaluator API")
        self.version: str = os.getenv("APP_VERSION", "0.1.0")
        self.environment: str = os.getenv("APP_ENV", "development")


@lru_cache
def get_app_config() -> AppConfig:
    return AppConfig()


