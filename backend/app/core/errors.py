"""Domain errors and their JSON representation.

Every error the API returns has the shape ``{"error": {"code": ..., "message": ...}}``.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class DomainError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class NotFoundError(DomainError):
    status_code = 404
    code = "not_found"


class InvalidFileError(DomainError):
    status_code = 422
    code = "invalid_file"


class FileTooLargeError(DomainError):
    status_code = 413
    code = "file_too_large"


class InvalidTargetError(DomainError):
    status_code = 422
    code = "invalid_target"


class UnsupportedTaskError(DomainError):
    status_code = 422
    code = "unsupported_task"


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def _domain(_: Request, exc: DomainError):
        return _error(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()
        ]
        return _error(422, "validation_error", "Request validation failed.", details)
