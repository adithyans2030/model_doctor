"""Local file storage for uploaded datasets.

Files are written under a random UUID name, so a user-supplied filename never
touches the filesystem path.
"""

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.core.errors import FileTooLargeError, InvalidFileError

CHUNK = 1024 * 1024


@dataclass
class StoredFile:
    stored_filename: str
    path: Path
    size_bytes: int
    sha256: str
    head: bytes  # first bytes of the file, for content sniffing


def save_stream(src: BinaryIO, directory: Path, max_bytes: int, head_size: int = 65536) -> StoredFile:
    directory.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{uuid.uuid4().hex}.csv"
    final_path = directory / stored_filename
    tmp_path = directory / f"{stored_filename}.part"

    digest = hashlib.sha256()
    size = 0
    head = b""
    try:
        with open(tmp_path, "wb") as out:
            while chunk := src.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise FileTooLargeError(
                        f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
                    )
                if len(head) < head_size:
                    head += chunk[: head_size - len(head)]
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise InvalidFileError("The uploaded file is empty.")
        os.replace(tmp_path, final_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return StoredFile(stored_filename, final_path, size, digest.hexdigest(), head)


def delete_file(directory: Path, stored_filename: str) -> None:
    (directory / stored_filename).unlink(missing_ok=True)
