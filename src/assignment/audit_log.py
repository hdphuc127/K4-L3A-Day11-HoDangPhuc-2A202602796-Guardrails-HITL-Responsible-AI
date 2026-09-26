"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._inputs: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp keyed by request_id; returns the id."""
        request_id = request_id or uuid.uuid4().hex
        self._open[request_id] = time.perf_counter()
        self._inputs[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            # Store a hash + redacted preview, never the raw text (it may carry secrets)
            "input_sha256": hashlib.sha256((text or "").encode("utf-8")).hexdigest(),
            "input_preview": _safe_preview(text),
            "input_chars": len(text or ""),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        start = self._open.pop(request_id, None) if request_id else None
        entry = self._inputs.pop(request_id, None) if request_id else None
        entry = entry or {"request_id": request_id, "user_id": user_id, "timestamp": utc_now_iso()}
        entry.update({
            "output_preview": _safe_preview(text),
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1) if start else None,
        })
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _safe_preview(text: str | None, limit: int = 200) -> str:
    from guardrails.output_guardrails import content_filter

    return content_filter(text or "")["redacted"][:limit]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
