"""ModelDoctor API.

Run with:  uvicorn app.main:create_app --factory --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker

from app.api.routes import (
    assistant, datasets, experiments, explain, health, models, monitoring, projects, registry, reports,
)
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.security import GatewayMiddleware
from app.db.session import make_engine
from app.services.jobs import recover_interrupted

API_PREFIX = "/api/v1"
ROUTERS = (health, projects, datasets, reports, models, explain, experiments, registry, monitoring, assistant)

log = logging.getLogger("modeldoctor")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()

    engine = make_engine(settings.resolved_database_url)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            recovered = recover_interrupted(session_factory)
            if recovered:
                log.warning("marked %d interrupted job(s) as failed", recovered)
        except Exception:  # tables may not exist yet (before `alembic upgrade head`)
            log.warning("could not check for interrupted jobs; has `alembic upgrade head` been run?")
        if not settings.api_keys:
            log.warning("MD_API_KEYS is empty: the API is open (fine for local development only)")
        yield

    app = FastAPI(
        lifespan=lifespan,
        title=f"{settings.app_name} API",
        version=settings.version,
        description="Diagnose, explain and optimise machine-learning models.",
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory

    # Starlette runs the last-added middleware first: CORS wraps the gateway so
    # preflight requests and error responses still carry CORS headers.
    app.add_middleware(GatewayMiddleware, settings=settings, session_factory=session_factory)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)

    for module in ROUTERS:
        app.include_router(module.router, prefix=API_PREFIX)
    app.include_router(monitoring.metrics_router)

    return app
