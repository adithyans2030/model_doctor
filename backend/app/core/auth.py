"""Password hashing (Argon2id) and signed access tokens (JWT, HS256)."""

import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.core.config import Settings

ROLES = ("viewer", "member", "admin")
_hasher = PasswordHasher()  # Argon2id with the library's recommended parameters


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


def signing_key(settings: Settings) -> str:
    """MD_SECRET_KEY if set; otherwise a random key generated once and kept in the data dir."""
    if settings.secret_key:
        return settings.secret_key
    path: Path = settings.data_dir / ".secret_key"
    if path.exists():
        return path.read_text().strip()
    key = secrets.token_urlsafe(48)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key)
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - Windows
        pass
    return key


def create_token(settings: Settings, user) -> tuple[str, int]:
    ttl = settings.token_ttl_minutes * 60
    now = datetime.now(timezone.utc)
    payload = {"sub": user.id, "email": user.email, "name": user.name, "role": user.role,
               "iat": now, "exp": now + timedelta(seconds=ttl), "iss": "modeldoctor"}
    return jwt.encode(payload, signing_key(settings), algorithm="HS256"), ttl


def decode_token(settings: Settings, token: str) -> dict | None:
    try:
        return jwt.decode(token, signing_key(settings), algorithms=["HS256"], issuer="modeldoctor",
                          options={"require": ["exp", "sub"]})
    except jwt.PyJWTError:
        return None
