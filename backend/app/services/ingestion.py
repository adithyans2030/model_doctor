"""CSV validation and safe parsing.

Only plain-text CSV is accepted. Nothing in an uploaded file is ever executed:
the file is parsed as data by pandas, never unpickled or evaluated.
"""

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from charset_normalizer import from_bytes

from app.core.config import Settings
from app.core.errors import InvalidFileError

ALLOWED_EXTENSIONS = {".csv"}
CANDIDATE_DELIMITERS = ",;\t|"

# Magic numbers of common non-CSV formats people upload by mistake.
_MAGIC = {
    b"PK\x03\x04": "a ZIP archive (possibly an .xlsx workbook)",
    b"\xd0\xcf\x11\xe0": "a legacy Office file (possibly an .xls workbook)",
    b"%PDF": "a PDF",
    b"\x80\x04": "a Python pickle",
    b"\x80\x05": "a Python pickle",
    b"PAR1": "a Parquet file",
}

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass
class CsvFormat:
    encoding: str
    delimiter: str
    header: list[str]  # raw header names as they appear in the file


def sanitize_filename(name: str | None) -> str:
    """Return a display-safe base name. Used for display only, never as a path."""
    if not name:
        raise InvalidFileError("The upload has no filename.")
    base = name.replace("\\", "/").split("/")[-1]
    base = _CONTROL_CHARS.sub("", base).strip()
    if not base:
        raise InvalidFileError("The upload has no filename.")
    suffix = Path(base).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise InvalidFileError(
            f"Unsupported file type '{suffix or '(none)'}'. Upload a .csv file."
        )
    return base[:255]


def check_content(head: bytes) -> None:
    for magic, label in _MAGIC.items():
        if head.startswith(magic):
            raise InvalidFileError(f"The file looks like {label}, not a CSV. Export it as CSV first.")
    if b"\x00" in head:
        raise InvalidFileError(
            "The file contains binary data. Upload a plain-text CSV saved as UTF-8."
        )


def detect_encoding(head: bytes) -> str:
    try:
        head.decode("utf-8-sig")
        return "utf-8-sig" if head.startswith(b"\xef\xbb\xbf") else "utf-8"
    except UnicodeDecodeError as exc:
        # The sample may cut a multi-byte character in half at its very end.
        if exc.start >= len(head) - 3:
            return "utf-8"
    best = from_bytes(head).best()
    if best is None or not best.encoding:
        raise InvalidFileError("Could not determine the file's text encoding. Save it as UTF-8.")
    return best.encoding


def detect_format(head: bytes) -> CsvFormat:
    check_content(head)
    encoding = detect_encoding(head)
    text = head.decode(encoding, errors="ignore")
    lines = text.splitlines()
    if not lines or not lines[0].strip():
        raise InvalidFileError("The file has no header row.")

    # Drop the last line of the sample, which may be truncated.
    sample = "\n".join(lines[:-1] if len(lines) > 1 else lines)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=CANDIDATE_DELIMITERS).delimiter
    except csv.Error:
        delimiter = ","

    header = next(csv.reader(io.StringIO(lines[0]), delimiter=delimiter))
    return CsvFormat(encoding=encoding, delimiter=delimiter, header=header)


def read_header(path: Path, encoding: str, delimiter: str) -> list[str]:
    """Raw header names of a stored CSV (needed to map stripped names back for usecols)."""
    with open(path, encoding=encoding, newline="") as f:
        return next(csv.reader(f, delimiter=delimiter), [])


def validate_header(header: list[str], settings: Settings) -> None:
    names = [h.strip() for h in header]
    if len(names) < 2:
        raise InvalidFileError(
            "The CSV needs at least two columns (features plus a target). "
            "If it does, check that the delimiter is a comma, semicolon, tab or pipe."
        )
    if len(names) > settings.max_cols:
        raise InvalidFileError(
            f"The CSV has {len(names)} columns; the limit is {settings.max_cols}."
        )
    seen, dupes = set(), []
    for n in names:
        if n and n in seen:
            dupes.append(n)
        seen.add(n)
    if dupes:
        raise InvalidFileError(f"Duplicate column names: {', '.join(sorted(set(dupes)))}.")


def read_csv(
    path: Path,
    fmt: CsvFormat,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read a stored CSV. Column names are stripped of surrounding whitespace."""
    raw_by_clean = {h.strip(): h for h in fmt.header}
    raw_usecols = [raw_by_clean.get(c, c) for c in usecols] if usecols else None
    try:
        df = pd.read_csv(
            path,
            sep=fmt.delimiter,
            encoding=fmt.encoding,
            usecols=raw_usecols,
            nrows=nrows,
            low_memory=False,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise InvalidFileError(f"Could not parse the CSV: {first_line}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_upload(path: Path, head: bytes, settings: Settings) -> tuple[pd.DataFrame, CsvFormat]:
    fmt = detect_format(head)
    validate_header(fmt.header, settings)
    df = read_csv(path, fmt, nrows=settings.max_rows + 1)
    if len(df) > settings.max_rows:
        raise InvalidFileError(f"The CSV has more than {settings.max_rows:,} rows.")
    if len(df) < settings.min_rows:
        raise InvalidFileError(
            f"The CSV has {len(df)} data rows; at least {settings.min_rows} are needed "
            "for meaningful diagnostics."
        )
    return df, fmt
