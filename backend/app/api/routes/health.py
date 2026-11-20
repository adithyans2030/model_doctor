from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import DbDep, SettingsDep
from app.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(db: DbDep, settings: SettingsDep):
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        database = "error"
    return HealthOut(
        status="ok" if database == "ok" else "degraded",
        version=settings.version,
        database=database,
    )
