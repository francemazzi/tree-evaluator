from datetime import datetime

from pydantic import BaseModel, ConfigDict


class HealthCheckResponse(BaseModel):
    status: str
    app: str
    version: str
    timestamp: datetime
    environment: str

    model_config = ConfigDict(from_attributes=True)


