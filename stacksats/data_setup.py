"""Packaged assets and explicit data-setup workflows for StackSats."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
import datetime as dt
import hashlib
import http.cookiejar
import json
import os
import re
import shutil
import urllib.parse
import urllib.request

import polars as pl

STACKSATS_HOME = Path.home() / ".stacksats"
MANAGED_DATA_DIR = STACKSATS_HOME / "data"
MANAGED_BRK_DIR = MANAGED_DATA_DIR / "brk"
MANAGED_RUNTIME_PARQUET = MANAGED_DATA_DIR / "bitcoin_analytics.parquet"
ASSET_PACKAGE = "stacksats.assets"
PACKAGED_MANIFEST_NAME = "brk_data_manifest.json"
PACKAGED_SCHEMA_NAME = "merged-metrics-parquet-schema.md"
PACKAGED_DEMO_PARQUET_NAME = "bitcoin_analytics_demo.parquet"
GOOGLE_DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"
REQUEST_TIMEOUT_SECONDS = 60
RUNTIME_PROJECTION_METRICS = (
    "market_cap",
    "supply_btc",
    "mvrv",
    "adjusted_sopr",
    "adjusted_sopr_7d_ema",
    "realized_cap_growth_rate",
    "market_cap_growth_rate",
)
PLACEHOLDER_EXACT = {"TODO", "TBD", "REPLACE"}


class ManifestError(ValueError):
    """Raised when the packaged BRK manifest is invalid."""


class DownloadError(RuntimeError):
    """Raised when asset download or verification fails."""


@dataclass(frozen=True)
class AssetSpec:
    """Resolved definition for a fetchable or packaged asset."""

    name: str
    source: str
    sha256: str
    size_bytes: int
    version: str
    file_id: str | None = None
    resource: str | None = None


@dataclass(frozen=True)
class DataManifest:
    """Root manifest for explicit BRK setup."""

    gdrive_folder_url: str
    parquet: AssetSpec
    schema: AssetSpec
    updated_at_utc: str


@dataclass(frozen=True)
class RuntimeParquetResolution:
    """Resolved runtime parquet path and candidate search order."""

    path: Path
    source: str
    checked_paths: tuple[tuple[str, Path], ...]


def _asset_resource(name: str):
    return files(ASSET_PACKAGE).joinpath(name)


def packaged_text(name: str) -> str:
    """Read a packaged text asset."""

    return _asset_resource(name).read_text(encoding="utf-8")


def packaged_bytes(name: str) -> bytes:
    """Read a packaged binary asset."""

    return _asset_resource(name).read_bytes()


@contextmanager
def packaged_demo_parquet_path():
    """Yield a filesystem path to the bundled demo parquet."""

    with as_file(_asset_resource(PACKAGED_DEMO_PARQUET_NAME)) as path:
        yield path


def _is_placeholder(text: str) -> bool:
    value = text.strip()
    upper = value.upper()
    if "<" in value or ">" in value:
        return True
    if upper in PLACEHOLDER_EXACT:
        return True
    return upper.startswith("REPLACE_WITH_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_asset(name: str, payload: object) -> AssetSpec:
    if not isinstance(payload, dict):
        raise ManifestError(f"Manifest field '{name}' must be an object.")

    required = ("name", "sha256", "size_bytes", "version")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ManifestError(f"Manifest field '{name}' missing keys: {', '.join(missing)}.")

    asset_name = str(payload["name"]).strip()
    source_raw = payload.get("source")
    if source_raw is None:
        if "file_id" in payload:
            source = "gdrive"
        elif "resource" in payload:
            source = "packaged"
        else:
            raise ManifestError(
                f"Manifest field '{name}.source' missing and could not be inferred."
            )
    else:
        source = str(source_raw).strip().lower()
    sha256 = str(payload["sha256"]).strip().lower()
    version = str(payload["version"]).strip()

    if not asset_name:
        raise ManifestError(f"Manifest field '{name}.name' must be non-empty.")
    if source not in {"gdrive", "packaged"}:
        raise ManifestError(f"Manifest field '{name}.source' must be 'gdrive' or 'packaged'.")
    if not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ManifestError(f"Manifest field '{name}.sha256' must be a 64-char hex digest.")

    size_bytes = payload["size_bytes"]
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ManifestError(f"Manifest field '{name}.size_bytes' must be a positive integer.")
    if not version:
        raise ManifestError(f"Manifest field '{name}.version' must be non-empty.")

    file_id = None
    resource = None
    if source == "gdrive":
        if "file_id" not in payload:
            raise ManifestError(f"Manifest field '{name}.file_id' is required for gdrive assets.")
        file_id = str(payload["file_id"]).strip()
        if not file_id or _is_placeholder(file_id):
            raise ManifestError(
                f"Manifest field '{name}.file_id' is missing or placeholder. "
                "Set it to a real Google Drive file id."
            )
    else:
        if "resource" not in payload:
            raise ManifestError(
                f"Manifest field '{name}.resource' is required for packaged assets."
            )
        resource = str(payload["resource"]).strip()
        if not resource:
            raise ManifestError(f"Manifest field '{name}.resource' must be non-empty.")

    return AssetSpec(
        name=asset_name,
        source=source,
        sha256=sha256,
        size_bytes=size_bytes,
        version=version,
        file_id=file_id,
        resource=resource,
    )


def load_manifest(path: Path | None = None) -> DataManifest:
    """Load the packaged or explicit BRK manifest."""

    raw_text = packaged_text(PACKAGED_MANIFEST_NAME) if path is None else path.read_text(encoding="utf-8")
    raw = json.loads(raw_text)
    if not isinstance(raw, dict):
        raise ManifestError("Manifest root must be an object.")

    required = ("gdrive_folder_url", "parquet", "schema", "updated_at_utc")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ManifestError(f"Manifest missing top-level keys: {', '.join(missing)}.")

    folder_url = str(raw["gdrive_folder_url"]).strip()
    updated = str(raw["updated_at_utc"]).strip()
    if not folder_url.startswith("https://drive.google.com/"):
        raise ManifestError("Manifest field 'gdrive_folder_url' must be a Google Drive URL.")
    if not updated:
        raise ManifestError("Manifest field 'updated_at_utc' must be non-empty.")

    return DataManifest(
        gdrive_folder_url=folder_url,
        parquet=_parse_asset("parquet", raw["parquet"]),
        schema=_parse_asset("schema", raw["schema"]),
        updated_at_utc=updated,
    )


def _stream_response_to_file(response, output: Path) -> int:
    total = 0
    with output.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            handle.write(chunk)
    return total


def _extract_confirm_token(html: str) -> str | None:
    match = re.search(r"[?&]confirm=([0-9A-Za-z_-]+)", html)
    return match.group(1) if match else None


def _download_from_gdrive(file_id: str, output_path: Path) -> int:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    first_url = GOOGLE_DRIVE_DOWNLOAD_URL.format(file_id=urllib.parse.quote(file_id, safe=""))
    with opener.open(first_url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            html = response.read().decode("utf-8", errors="ignore")
            token = _extract_confirm_token(html)
            if not token:
                raise DownloadError(
                    "Google Drive returned HTML without a download token. "
                    "Check sharing permissions or quota for the public asset."
                )
            confirm_url = (
                "https://drive.google.com/uc?export=download"
                f"&confirm={urllib.parse.quote(token, safe='')}"
                f"&id={urllib.parse.quote(file_id, safe='')}"
            )
            with opener.open(confirm_url, timeout=REQUEST_TIMEOUT_SECONDS) as second:
                return _stream_response_to_file(second, output_path)

        return _stream_response_to_file(response, output_path)


def _copy_packaged_asset(resource: str, output_path: Path) -> int:
    payload = packaged_bytes(resource)
    output_path.write_bytes(payload)
    return len(payload)


def _download_and_verify(
    asset: AssetSpec,
    destination: Path,
    *,
    overwrite: bool,
    downloader,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        existing_sha = _sha256(destination)
        existing_size = destination.stat().st_size
        if existing_sha != asset.sha256 or existing_size != asset.size_bytes:
            raise DownloadError(
                f"{destination} exists but does not match manifest. Pass --overwrite to replace it."
            )
        print(f"[fetch_brk_data] Existing file verified, skipping: {destination}")
        return

    tmp = destination.with_suffix(destination.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    try:
        if asset.source == "gdrive":
            if asset.file_id is None:
                raise DownloadError(f"Manifest is missing file_id for {asset.name}.")
            bytes_written = downloader(asset.file_id, tmp)
        else:
            if asset.resource is None:
                raise DownloadError(f"Manifest is missing resource for {asset.name}.")
            bytes_written = _copy_packaged_asset(asset.resource, tmp)

        if bytes_written <= 0:
            raise DownloadError(f"Download produced no bytes for {asset.name}.")

        actual_size = tmp.stat().st_size
        if actual_size != asset.size_bytes:
            raise DownloadError(
                f"Size mismatch for {asset.name}: expected {asset.size_bytes}, got {actual_size}."
            )

        actual_sha = _sha256(tmp)
        if actual_sha != asset.sha256:
            raise DownloadError(
                f"SHA-256 mismatch for {asset.name}: expected {asset.sha256}, got {actual_sha}."
            )

        shutil.move(str(tmp), str(destination))
        print(f"[fetch_brk_data] Downloaded and verified: {destination}")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def fetch_assets(
    *,
    manifest_path: Path | None = None,
    target_dir: Path,
    schema_dir: Path | None = None,
    overwrite: bool = False,
    downloader=None,
) -> tuple[Path, Path]:
    """Fetch the canonical merged-metrics parquet plus schema sidecar."""

    if downloader is None:
        downloader = _download_from_gdrive

    manifest = load_manifest(manifest_path)
    resolved_target_dir = target_dir.expanduser().resolve()
    resolved_schema_dir = (
        schema_dir.expanduser().resolve() if schema_dir is not None else resolved_target_dir
    )
    parquet_path = resolved_target_dir / manifest.parquet.name
    schema_path = resolved_schema_dir / manifest.schema.name

    _download_and_verify(
        manifest.parquet,
        parquet_path,
        overwrite=overwrite,
        downloader=downloader,
    )
    _download_and_verify(
        manifest.schema,
        schema_path,
        overwrite=overwrite,
        downloader=downloader,
    )

    print(f"[fetch_brk_data] Source folder: {manifest.gdrive_folder_url}")
    print(f"[fetch_brk_data] Manifest updated_at_utc: {manifest.updated_at_utc}")
    print(f"[fetch_brk_data] Canonical parquet: {parquet_path}")
    return parquet_path, schema_path


def _candidate_runtime_paths(path_override: str | None) -> list[tuple[str, Path]]:
    seen: set[str] = set()
    candidates: list[tuple[str, Path]] = []
    raw_candidates = [
        ("env STACKSATS_ANALYTICS_PARQUET", os.getenv("STACKSATS_ANALYTICS_PARQUET")),
        ("explicit parquet_path", path_override),
        ("managed default", str(MANAGED_RUNTIME_PARQUET)),
        ("legacy local fallback", "./bitcoin_analytics.parquet"),
    ]
    for label, raw in raw_candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((label, path))
    return candidates


def resolve_runtime_parquet(path_override: str | None = None) -> RuntimeParquetResolution:
    """Resolve the runtime parquet using the supported precedence order."""

    candidates = _candidate_runtime_paths(path_override)
    for label, path in candidates:
        if path.exists():
            return RuntimeParquetResolution(
                path=path,
                source=label,
                checked_paths=tuple(candidates),
            )

    checked_lines = "\n".join(
        f"- {label}: {path.expanduser()}" for label, path in candidates
    )
    raise FileNotFoundError(
        "No runtime parquet could be resolved.\n"
        f"Checked:\n{checked_lines}\n"
        "Next steps:\n"
        "- Run `stacksats demo backtest`\n"
        "- Run `stacksats data fetch`\n"
        "- Run `stacksats data prepare`\n"
        "- Or set STACKSATS_ANALYTICS_PARQUET to a valid runtime parquet."
    )


def _normalize_runtime_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if "date" not in frame.columns:
        raise ValueError("Runtime parquet must contain a 'date' column.")
    if "price_usd" not in frame.columns:
        raise ValueError("Runtime parquet must contain a 'price_usd' column.")
    if frame["date"].dtype == pl.Utf8:
        frame = frame.with_columns(pl.col("date").str.to_datetime())
    if "Datetime" in str(frame["date"].dtype):
        frame = frame.with_columns(pl.col("date").dt.replace_time_zone(None).dt.truncate("1d"))
    frame = frame.unique(subset=["date"], keep="last").sort("date")
    frame = frame.with_columns(pl.col("price_usd").cast(pl.Float64, strict=False))
    if "mvrv" in frame.columns:
        frame = frame.with_columns(pl.col("mvrv").cast(pl.Float64, strict=False))
    return frame


def project_runtime_parquet(source: Path) -> pl.DataFrame:
    """Project canonical merged-metrics parquet into runtime columns."""

    schema = pl.read_parquet_schema(source)
    columns = set(schema)
    if {"date", "price_usd"}.issubset(columns):
        return _normalize_runtime_frame(pl.read_parquet(source))
    if {"day_utc", "metric", "value"}.issubset(columns):
        return (
            pl.scan_parquet(source)
            .filter(pl.col("metric").is_in(RUNTIME_PROJECTION_METRICS))
            .select("day_utc", "metric", "value")
            .collect()
            .pivot(values="value", index="day_utc", on="metric")
            .with_columns((pl.col("market_cap") / pl.col("supply_btc")).alias("price_usd"))
            .rename({"day_utc": "date"})
            .select(
                "date",
                "price_usd",
                "mvrv",
                "adjusted_sopr",
                "adjusted_sopr_7d_ema",
                "realized_cap_growth_rate",
                "market_cap_growth_rate",
            )
            .filter(pl.col("price_usd").is_finite() & (pl.col("price_usd") > 0))
        )
    raise ValueError(
        "Unsupported source parquet. Expected canonical merged-metrics columns "
        "('day_utc', 'metric', 'value') or runtime columns ('date', 'price_usd')."
    )


def prepare_runtime_parquet(
    source: Path,
    *,
    output: Path = MANAGED_RUNTIME_PARQUET,
    overwrite: bool = False,
) -> Path:
    """Create a runtime parquet from canonical merged metrics or wide runtime data."""

    source_path = source.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source parquet not found: {source_path}")

    output_path = output.expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = project_runtime_parquet(source_path)
    frame.write_parquet(output_path)
    return output_path


def latest_fetched_parquet(brk_dir: Path = MANAGED_BRK_DIR) -> Path:
    """Return the newest canonical parquet downloaded via `data fetch`."""

    directory = brk_dir.expanduser().resolve()
    matches = sorted(directory.glob("*.parquet"), key=lambda item: item.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"No canonical parquet found under {directory}. Run `stacksats data fetch` first."
        )
    return matches[-1]


def data_doctor(path_override: str | None = None) -> dict[str, object]:
    """Inspect the current runtime parquet resolution state."""

    candidates = _candidate_runtime_paths(path_override)
    checked_paths = [
        {
            "source": label,
            "path": str(path.expanduser()),
            "exists": path.exists(),
        }
        for label, path in candidates
    ]
    diagnosis: dict[str, object] = {
        "checked_paths": checked_paths,
        "managed_default_path": str(MANAGED_RUNTIME_PARQUET),
        "next_steps": [
            "stacksats demo backtest",
            "stacksats data fetch",
            "stacksats data prepare",
        ],
    }

    try:
        resolution = resolve_runtime_parquet(path_override)
    except FileNotFoundError as exc:
        diagnosis["status"] = "missing"
        diagnosis["error"] = str(exc)
        return diagnosis

    diagnosis["resolution_source"] = resolution.source
    diagnosis["resolved_path"] = str(resolution.path.expanduser())

    try:
        frame = _normalize_runtime_frame(pl.read_parquet(resolution.path))
    except Exception as exc:
        diagnosis["status"] = "invalid"
        diagnosis["error"] = str(exc)
        return diagnosis

    diagnosis["status"] = "ok"
    diagnosis["columns"] = frame.columns
    diagnosis["row_count"] = int(frame.height)
    diagnosis["has_price_usd"] = "price_usd" in frame.columns
    diagnosis["has_mvrv"] = "mvrv" in frame.columns
    diagnosis["has_daily_gaps"] = False
    diagnosis["gap_count"] = 0
    if frame.is_empty():
        diagnosis["coverage_start"] = None
        diagnosis["coverage_end"] = None
    else:
        diagnosis["coverage_start"] = str(frame["date"].min())[:10]
        diagnosis["coverage_end"] = str(frame["date"].max())[:10]

        dates = frame["date"].to_list()
        gap_pairs: list[tuple[str, str]] = []
        for prev_raw, curr_raw in zip(dates, dates[1:]):
            prev = prev_raw.date() if isinstance(prev_raw, dt.datetime) else prev_raw
            curr = curr_raw.date() if isinstance(curr_raw, dt.datetime) else curr_raw
            if not isinstance(prev, dt.date) or not isinstance(curr, dt.date):
                continue
            if (curr - prev).days > 1:
                gap_pairs.append((prev.isoformat(), curr.isoformat()))

        if gap_pairs:
            diagnosis["status"] = "warning"
            diagnosis["has_daily_gaps"] = True
            diagnosis["gap_count"] = len(gap_pairs)
            diagnosis["first_gap_after"] = gap_pairs[0][0]
            diagnosis["first_gap_before"] = gap_pairs[0][1]
            diagnosis["warning"] = "Runtime parquet contains gaps in daily coverage."
    return diagnosis
