from fastapi import FastAPI

from app.core.config import AppConfig, get_app_config
from app.api.v1.router import api_v1_router


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or get_app_config()
    application = FastAPI(title=cfg.app_name, version=cfg.version)

    # Routers
    application.include_router(api_v1_router)

    @application.get("/")
    async def root():
        return {"message": f"{cfg.app_name} is running", "version": cfg.version}

    return application


app = create_app()


