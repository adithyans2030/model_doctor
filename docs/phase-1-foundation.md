# Phase 1: Foundation and dataset ingestion

**Goal:** build a safe, tested backend that accepts a dataset, understands its
schema and validates the prediction target. Every later phase starts from this.

## What was built

| Piece | File | Responsibility |
|---|---|---|
| App factory | `backend/app/main.py` | `create_app(settings)` builds the FastAPI app, DB engine, CORS, routers |
| Settings | `backend/app/core/config.py` | All limits and paths, overridable with `MD_*` env vars or `.env` |
| Errors | `backend/app/core/errors.py` | Domain exceptions mapped to `{"error": {"code", "message"}}` |
| Database | `backend/app/db/` | SQLAlchemy 2 models `Project`, `Dataset`; SQLite by default |
| Migrations | `backend/alembic/` | `0001_initial` creates projects and datasets |
| Storage | `backend/app/storage.py` | Streams uploads to disk under a UUID name with a size cap |
| Ingestion | `backend/app/services/ingestion.py` | File-type checks, encoding/delimiter detection, safe parsing |
| Profiling | `backend/app/services/profiling.py` | Per-column type inference |
| Target | `backend/app/services/target.py` | Target validation and task-type inference |
| Use cases | `backend/app/services/datasets.py` | Ingest, preview, set target, delete, `load_dataframe()` |

## Design decisions

- **App factory instead of a global app.** Tests build an isolated app with a
  temporary database and data folder (`tests/conftest.py`), so tests never touch
  real data. Run with `uvicorn app.main:create_app --factory`.
- **Thin routes, logic in services.** Routes only translate HTTP to service
  calls. Later phases reuse the services (for example
  `load_dataframe(dataset, settings)`) without going through HTTP.
- **SQLite now, Postgres later.** Only portable column types are used (String
  UUIDs, JSON), so moving to Postgres only means changing `MD_DATABASE_URL`.
  Docker Compose does that in Phase 7.
- **Classification only.** A continuous numeric target is rejected with
  `unsupported_task` instead of being half-supported.

## Upload pipeline

1. **Filename.** Only the base name is kept and control characters are removed.
   It is used for display only. The extension must be `.csv`.
2. **Streaming save.** The upload is read in 1 MB chunks and aborted with `413`
   once it passes `MD_MAX_UPLOAD_MB`. It is written to `<uuid>.csv.part`, then
   renamed. A SHA-256 hash is computed along the way.
3. **Content sniffing.** The first 64 KB are checked for ZIP/XLSX, XLS, PDF,
   Parquet and pickle signatures, and for NUL bytes (binary files).
4. **Encoding.** UTF-8 or UTF-8 with a BOM is tried first, with
   `charset-normalizer` as a fallback (for example Latin-1).
5. **Delimiter.** `csv.Sniffer` chooses between `,` `;` tab and `|`.
6. **Structure.** At least 2 columns, at most `MD_MAX_COLS`, no duplicate
   header names, and between `MD_MIN_ROWS` and `MD_MAX_ROWS` rows.
7. **Profile.** Each column gets a type, missing count/percentage, number of
   unique values and sample values.

If any step fails, the stored file is deleted and the transaction is rolled back.

## Type inference rules (`profiling.py`)

| Inferred type | Rule (applied in this order) |
|---|---|
| `empty` | all values missing |
| `boolean` | bool dtype; numeric with only {0,1}; or two strings from {true,false,yes,no,y,n,t,f} ignoring case |
| `id` | numeric integers, all unique, and monotonic or ID-like name; strings ≥99% unique (≥90% with an ID-like name) |
| `numeric` | numeric dtype, or ≥95% of text values parse as numbers (flagged `numeric_stored_as_text`) |
| `datetime` | ≥95% of a sample parses as dates and values contain date separators |
| `text` | average length ≥30, mostly containing spaces, ≥50% unique |
| `categorical` | everything else |

## Target rules (`target.py`)

Rejected as targets: `id`, `text`, `datetime` and `empty` columns, constant
columns, columns with fewer than `MD_MIN_ROWS` labelled rows, and columns with
more than `MD_MAX_CLASSES` classes. A numeric target with non-integer values,
or more than `MD_MAX_INTEGER_TARGET_CLASSES` distinct values, counts as
regression and is rejected. Boolean labels are lower-cased, so `Yes` and `yes`
become one class.

Warnings (the target is still accepted): rows missing a target value, labels
that differ only by case or spacing, and classes smaller than
`MD_MIN_CLASS_SAMPLES`.

## API

| Method | Path |
|---|---|
| GET | `/api/v1/health` |
| POST / GET | `/api/v1/projects` |
| GET / DELETE | `/api/v1/projects/{id}` |
| POST / GET | `/api/v1/projects/{id}/datasets` |
| GET / DELETE | `/api/v1/datasets/{id}` |
| GET | `/api/v1/datasets/{id}/preview?rows=20` |
| POST | `/api/v1/datasets/{id}/target` with body `{"column": "churned"}` |

## Tests

`tests/test_projects.py`, `test_datasets.py`, `test_target.py` and
`test_profiling.py` cover:
- Projects: CRUD, and that deleting a project also deletes its files.
- Uploads: every rejection path (wrong extension, empty file, ZIP, pickle,
  binary, too few rows, a single column, duplicate headers, malformed rows,
  too many columns or rows, oversized files), path-traversal filenames, and
  Latin-1, semicolon-delimited and BOM files.
- Preview and type inference.
- Every target rule and warning.

## Demo data

`backend/scripts/make_demo_data.py` produces `sample_data/churn_demo.csv`:
5,050 rows with 9 planted problems. Later phases are tested against these
known answers.
