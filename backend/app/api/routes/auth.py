"""User accounts: register, sign in, current user, and admin user management."""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select

from app.api.deps import DbDep, SettingsDep
from app.core.auth import create_token, hash_password, needs_rehash, verify_password
from app.core.errors import DomainError, NotFoundError
from app.core.errors import ForbiddenError as Forbidden
from app.core.errors import UnauthorizedError as Unauthorized
from app.db.models import User, utcnow

router = APIRouter(prefix="/auth", tags=["auth"])

Role = Literal["viewer", "member", "admin"]


class RegisterIn(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=200)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        local, _, domain = v.partition("@")
        if not local or "." not in domain:
            raise ValueError("enter a valid email address")
        return v


class LoginIn(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    name: str
    role: Role
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: UserOut


class UserUpdate(BaseModel):
    role: Role | None = None
    is_active: bool | None = None


class AuthStatus(BaseModel):
    auth_required: bool
    open_registration: bool
    has_users: bool


def current_user(request: Request, db) -> User | None:
    uid = getattr(request.state, "user_id", None)
    return db.get(User, uid) if uid else None


def require_admin(request: Request, db) -> User:
    user = current_user(request, db)
    if user is None:
        raise Unauthorized("Sign in as an admin.")
    if user.role != "admin":
        raise Forbidden("Only admins can do this.")
    return user


@router.get("/status", response_model=AuthStatus)
def auth_status(db: DbDep, settings: SettingsDep):
    """Whether sign-in is required, so the dashboard knows to show the login page."""
    has_users = (db.scalar(select(func.count()).select_from(User)) or 0) > 0
    return {"auth_required": settings.auth_required, "open_registration": settings.open_registration,
            "has_users": has_users}


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, request: Request, db: DbDep, settings: SettingsDep):
    """Create an account. The first account becomes **admin**; later ones are members
    (or need an admin when MD_OPEN_REGISTRATION=false)."""
    first = (db.scalar(select(func.count()).select_from(User)) or 0) == 0
    if not first and not settings.open_registration:
        require_admin(request, db)
    if db.scalar(select(User).where(User.email == body.email)):
        raise DomainError("An account with this email already exists.", code="email_taken")
    user = User(email=body.email, name=body.name.strip(), password_hash=hash_password(body.password),
                role="admin" if first else "member", last_login_at=utcnow())
    db.add(user)
    db.commit()
    db.refresh(user)
    token, ttl = create_token(settings, user)
    return {"access_token": token, "expires_in": ttl, "user": user}


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: DbDep, settings: SettingsDep):
    user = db.scalar(select(User).where(User.email == body.email.strip().lower()))
    # Same message for unknown email and wrong password, so accounts can't be enumerated.
    if user is None or not verify_password(body.password, user.password_hash):
        raise Unauthorized("Incorrect email or password.")
    if not user.is_active:
        raise Forbidden("This account is deactivated.")
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
    user.last_login_at = utcnow()
    db.commit()
    token, ttl = create_token(settings, user)
    return {"access_token": token, "expires_in": ttl, "user": user}


@router.get("/me", response_model=UserOut)
def me(request: Request, db: DbDep):
    user = current_user(request, db)
    if user is None:
        raise Unauthorized("Not signed in.")
    return user


@router.get("/users", response_model=list[UserOut])
def list_users(request: Request, db: DbDep):
    require_admin(request, db)
    return list(db.scalars(select(User).order_by(User.created_at)))


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(user_id: str, body: UserUpdate, request: Request, db: DbDep):
    admin = require_admin(request, db)
    user = db.get(User, user_id)
    if user is None:
        raise NotFoundError("User not found.")
    if user.id == admin.id and (body.role not in (None, "admin") or body.is_active is False):
        raise DomainError("You can't demote or deactivate yourself.", code="invalid_request")
    if body.role is not None:
        user.role = body.role
    if body.is_active is not None:
        user.is_active = body.is_active
    db.commit()
    db.refresh(user)
    return user
