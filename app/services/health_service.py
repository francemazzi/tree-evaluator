from datetime import datetime, timezone

from app.core.config import AppConfig
from app.models.response import HealthCheckResponse


class HealthService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def get_health(self) -> HealthCheckResponse:
        return HealthCheckResponse(
            status="ok",
            app=self._config.app_name,
            version=self._config.version,
            timestamp=datetime.now(timezone.utc),
            environment=self._config.environment,
        )


