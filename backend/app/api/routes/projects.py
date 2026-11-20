from fastapi import APIRouter, status
from sqlalchemy import func, select

from app.api.deps import DbDep, SettingsDep
from app.db.models import Dataset, Project
from app.schemas import ProjectCreate, ProjectOut
from app.services import datasets as svc

router = APIRouter(prefix="/projects", tags=["projects"])


def _out(project: Project, count: int) -> ProjectOut:
    return ProjectOut.model_validate(project).model_copy(update={"dataset_count": count})


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreate, db: DbDep):
    project = Project(name=body.name, description=body.description)
    db.add(project)
    db.commit()
    db.refresh(project)
    return _out(project, 0)


@router.get("", response_model=list[ProjectOut])
def list_projects(db: DbDep):
    counts = (
        select(Dataset.project_id, func.count(Dataset.id).label("n"))
        .group_by(Dataset.project_id)
        .subquery()
    )
    rows = db.execute(
        select(Project, func.coalesce(counts.c.n, 0))
        .outerjoin(counts, counts.c.project_id == Project.id)
        .order_by(Project.created_at.desc())
    ).all()
    return [_out(p, n) for p, n in rows]


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, db: DbDep):
    project = svc.get_project(db, project_id)
    return _out(project, len(project.datasets))


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str, db: DbDep, settings: SettingsDep):
    svc.delete_project(db, settings, svc.get_project(db, project_id))
