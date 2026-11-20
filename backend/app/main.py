"""ModelDoctor API.

Run with:  uvicorn app.main:create_app --factory --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker

from app.api.routes import datasets, health, projects
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.db.session import make_engine

API_PREFIX = "/api/v1"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()

    engine = make_engine(settings.resolved_database_url)

    app = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.version,
        description="Diagnose, explain and optimise machine-learning models.",
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)

    for module in (health, projects, datasets):
        app.include_router(module.router, prefix=API_PREFIX)

    return app
