#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import csv
import inspect
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from vllm import envs

CSV_FIELDNAMES = [
    "timestamp",
    "file_path",
    "section",
    "detail",
    "duration_sec",
]

_WRITE_LOCK = threading.Lock()
_TRUNCATED_PATHS: set[str] = set()


def get_rank_profile_path(base_path: str, rank: int | None) -> str:
    if not base_path:
        return ""
    if rank is None:
        return base_path
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".csv"
    return f"{root}_rank{rank}{ext}"


def _format_timestamp(value: datetime | None) -> str:
    current = value or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def _caller_file(depth: int = 2) -> str:
    frame = inspect.stack()[depth]
    return os.path.relpath(os.path.abspath(frame.filename), start=os.getcwd())


def _append_profile_row(
    path: str,
    section: str,
    detail: str,
    duration_sec: float,
    *,
    start_time: datetime | None,
    file_path: str,
) -> None:
    if not path:
        return

    with _WRITE_LOCK:
        first_write = path not in _TRUNCATED_PATHS
        mode = "w" if first_write else "a"
        if first_write:
            _TRUNCATED_PATHS.add(path)
        with open(path, mode, newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDNAMES)
            if first_write:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": _format_timestamp(start_time),
                    "file_path": file_path,
                    "section": section,
                    "detail": detail,
                    "duration_sec": round(duration_sec, 6),
                }
            )


class ElasticEPProfile:
    def __init__(
        self,
        *,
        base_path: str = "",
        rank: int | None = None,
        file_path: str | None = None,
    ) -> None:
        self._base_path = base_path
        self._rank = rank
        self._file_path = file_path or _caller_file(depth=3)

    @property
    def enabled(self) -> bool:
        return bool(self._base_path)

    @property
    def path(self) -> str:
        return get_rank_profile_path(self._base_path, self._rank)

    def record(
        self,
        section: str,
        detail: str,
        duration_sec: float,
        *,
        start_time: datetime | None = None,
    ) -> None:
        _append_profile_row(
            self.path,
            section,
            detail,
            duration_sec,
            start_time=start_time,
            file_path=self._file_path,
        )

    @contextmanager
    def track(self, section: str, detail: str = "") -> Any:
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        start_wall = datetime.now(timezone.utc)
        try:
            yield
        finally:
            self.record(
                section,
                detail,
                time.perf_counter() - start,
                start_time=start_wall,
            )


def get_eep_profile(
    *,
    rank: int | None = None,
    file_path: str | None = None,
) -> ElasticEPProfile:
    return ElasticEPProfile(
        base_path=envs.VLLM_EEP_PROFILE_CSV,
        rank=rank,
        file_path=file_path,
    )
