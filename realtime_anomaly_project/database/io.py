from __future__ import annotations
import json
import csv
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Union

"""
database/io.py

Lightweight, safe I/O helpers for the realtime_anomaly_project.
Provides atomic writes and flexible readers/writers for JSON, CSV and (optionally) parquet.
"""


logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


def _p(path: PathLike) -> Path:
    return Path(path)


def ensure_dir(path: PathLike, exist_ok: bool = True) -> Path:
    """
    Ensure parent directory exists for the given path (or create the path if it is a directory).
    Returns the Path object for convenience.
    """
    p = _p(path)
    target = p if p.exists() and p.is_dir() else p.parent
    target.mkdir(parents=True, exist_ok=exist_ok)
    return p


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    """
    Write bytes to dst atomically by writing to a temp file in the same directory then replacing.
    """
    ensure_dir(dst)
    dirpath = dst.parent
    with tempfile.NamedTemporaryFile(dir=dirpath, delete=False) as tf:
        tmpname = Path(tf.name)
        tf.write(data)
        tf.flush()
    # replace is atomic on most OSes
    tmpname.replace(dst)


def _atomic_write_text(dst: Path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(dst, text.encode(encoding))


def save_json(path: PathLike, obj: Any, indent: int = 2, encoding: str = "utf-8") -> None:
    """
    Save Python object as JSON atomically.
    """
    p = _p(path)
    text = json.dumps(obj, indent=indent, ensure_ascii=False)
    _atomic_write_text(p, text, encoding=encoding)
    logger.debug("Wrote JSON to %s", p)


def load_json(path: PathLike, encoding: str = "utf-8") -> Any:
    """
    Load JSON from file and return the decoded Python object.
    """
    p = _p(path)
    with p.open("r", encoding=encoding) as fh:
        return json.load(fh)


def save_csv(path: PathLike, rows: Iterable[Mapping[str, Any]], fieldnames: Optional[List[str]] = None,
             encoding: str = "utf-8", newline: str = "") -> None:
    """
    Save an iterable of mapping rows (dict-like) to CSV. Fieldnames can be provided; otherwise inferred
    from the first row.
    """
    p = _p(path)
    ensure_dir(p)
    it = iter(rows)
    try:
        first = next(it)
    except StopIteration:
        # write an empty file
        _atomic_write_text(p, "", encoding=encoding)
        return

    if fieldnames is None:
        if isinstance(first, Mapping):
            fieldnames = list(first.keys())
        else:
            raise TypeError("save_csv expects iterable of Mapping when fieldnames not provided")

    # write using a temp file then replace
    dirpath = p.parent
    with tempfile.NamedTemporaryFile("w", dir=dirpath, delete=False, encoding=encoding, newline=newline) as tf:
        tmpname = Path(tf.name)
        writer = csv.DictWriter(tf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(first)
        for r in it:
            writer.writerow(r)
    tmpname.replace(p)
    logger.debug("Wrote CSV to %s", p)


def load_csv(path: PathLike, encoding: str = "utf-8") -> Iterator[Mapping[str, str]]:
    """
    Lazily yield rows from a CSV file as dictionaries (string values). Use list(load_csv(...)) if you need all rows.
    """
    p = _p(path)
    with p.open("r", encoding=encoding, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def save_parquet(path: PathLike, df) -> None:
    """
    Save a pandas DataFrame to parquet. pandas and pyarrow or fastparquet must be installed.
    """
    p = _p(path)
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas is required for save_parquet") from e

    ensure_dir(p)
    # pandas' to_parquet writes to path atomically if engine supports it; still use temp file to be safe
    dirpath = p.parent
    with tempfile.NamedTemporaryFile(delete=False, dir=dirpath) as tf:
        tmp = Path(tf.name)
    try:
        df.to_parquet(tmp)
        tmp.replace(p)
        logger.debug("Wrote parquet to %s", p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def load_parquet(path: PathLike):
    """
    Load parquet into a pandas DataFrame. pandas must be installed.
    """
    p = _p(path)
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas is required for load_parquet") from e
    return pd.read_parquet(p)


# Convenience generic loader/saver based on extension
def save(path: PathLike, obj: Any) -> None:
    p = _p(path)
    ext = p.suffix.lower()
    if ext in {".json", ".ndjson"}:
        save_json(p, obj)
    elif ext in {".csv"}:
        if not isinstance(obj, Iterable):
            raise TypeError("For CSV, obj should be an iterable of mappings")
        save_csv(p, obj)  # type: ignore
    elif ext in {".parquet"}:
        save_parquet(p, obj)
    else:
        # fallback: write text
        _atomic_write_text(p, str(obj))


def load(path: PathLike) -> Any:
    p = _p(path)
    ext = p.suffix.lower()
    if ext == ".json":
        return load_json(p)
    if ext == ".csv":
        return list(load_csv(p))
    if ext == ".parquet":
        return load_parquet(p)
    # fallback: return raw bytes
    with p.open("rb") as fh:
        return fh.read()