"""Local Blue Team demo server for the VinBank presentation.

The server intentionally uses a deterministic model response. The policy decisions
come from the lab's real Python guardrails, while the demo remains repeatable and
does not spend API quota or expose model credentials.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assignment.audit_log import AuditLogPlugin  # noqa: E402
from assignment.pipeline import is_egress_allowed  # noqa: E402
from assignment.rate_limiter import RateLimitPlugin  # noqa: E402
from guardrails.input_guardrails import detect_injection, topic_filter  # noqa: E402
from guardrails.output_guardrails import OutputGuardrailPlugin, content_filter  # noqa: E402
from google.genai import types  # noqa: E402


MAX_BODY_BYTES = 16_384
MAX_INPUT_CHARS = 4_000
DEFAULT_PORT = 8765
DECK_PATH = ROOT / "slides" / "VinBank_Guardrails_Lab.html"

SCENARIO_META = {
    "safe": {"label": "Safe banking", "source": "CONTROL"},
    "injection": {"label": "Instruction override", "source": "RED"},
    "completion": {"label": "Completion attack", "source": "RED"},
    "translation": {"label": "Translation attack", "source": "RED"},
    "creative": {"label": "Creative roleplay", "source": "RED"},
    "confirmation": {"label": "Confirmation attack", "source": "RED"},
    "multistep": {"label": "Multi-step extraction", "source": "RED"},
    "offtopic": {"label": "Off-topic request", "source": "CONTROL"},
    "pii": {"label": "PII in model output", "source": "CONTROL"},
    "rate": {"label": "Rate-limit abuse", "source": "RED"},
    "egress": {"label": "Untrusted egress", "source": "RED"},
}


def _elapsed_ms(start: float) -> float:
    return round(max(0.0, (time.perf_counter() - start) * 1_000), 3)


def _content_text(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "".join(part.text or "" for part in content.parts if hasattr(part, "text"))


def sanitize_preview(value: str, limit: int = 220) -> str:
    """Return a short audit-safe preview without PII or lab secrets."""
    cleaned = content_filter(value or "")["redacted"].replace("\x00", "")
    if len(cleaned) > limit:
        return f"{cleaned[: limit - 1]}…"
    return cleaned


class DemoRuntime:
    """Stateful local runtime: real policy functions, bounded in-memory audit."""

    def __init__(self, *, max_requests: int = 8, window_seconds: int = 60):
        self.rate_limiter = RateLimitPlugin(
            max_requests=max_requests,
            window_seconds=window_seconds,
        )
        self.output_guard = OutputGuardrailPlugin(use_llm_judge=False)
        self.audit = AuditLogPlugin()

    def recent_audit(self, limit: int = 20) -> list[dict]:
        bounded = max(1, min(limit, 100))
        return [self._sanitize_audit(row) for row in self.audit.logs[-bounded:]][::-1]

    @staticmethod
    def _sanitize_audit(record: dict) -> dict:
        return {
            "request_id": record.get("request_id"),
            "timestamp": record.get("completed_at"),
            "actor": record.get("user_id"),
            "input_preview": sanitize_preview(record.get("input", "")),
            "response_preview": sanitize_preview(record.get("output", "")),
            "blocked": bool(record.get("blocked")),
            "blocked_at": record.get("layer"),
            "latency_ms": record.get("latency_ms", 0),
        }

    @staticmethod
    def _model_response(scenario: str) -> str:
        if scenario == "pii":
            return (
                "Transaction receipt contact: 0901234567 and "
                "demo.customer@vinbank.com"
            )
        return (
            "VinBank demo request accepted. Account-specific data requires "
            "authenticated customer context."
        )

    async def evaluate(
        self,
        text: str,
        *,
        scenario: str = "safe",
        user_id: str = "presentation-user",
    ) -> dict:
        text = (text or "").strip()
        if not text:
            text = " "
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"input must be at most {MAX_INPUT_CHARS} characters")
        if scenario not in SCENARIO_META:
            raise ValueError("unknown scenario")

        request_id = f"demo-{uuid.uuid4().hex[:10]}"
        self.audit.record_input(
            user_id=user_id,
            text=sanitize_preview(text, limit=MAX_INPUT_CHARS),
            request_id=request_id,
        )
        trace: list[dict] = []
        blocked = False
        blocked_at: str | None = None
        decision = "ALLOW"
        response_text = ""

        def add_trace(name: str, state: str, reason: str, started: float) -> None:
            trace.append({
                "name": name,
                "state": state,
                "reason": reason,
                "duration_ms": _elapsed_ms(started),
            })

        # 1. Rate limit. The abuse preset fills an isolated user's real sliding window.
        started = time.perf_counter()
        rate_user = user_id
        if scenario == "rate":
            rate_user = f"rate-demo:{request_id}"
            now = time.time()
            self.rate_limiter.user_windows[rate_user].extend(
                [now] * self.rate_limiter.max_requests
            )
        rate_result = await self.rate_limiter.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=rate_user),
            user_message=types.Content(
                role="user", parts=[types.Part.from_text(text=text)]
            ),
        )
        if rate_result is not None:
            blocked = True
            blocked_at = "rate_limit"
            decision = "BLOCK"
            response_text = _content_text(rate_result)
            add_trace(
                "Rate limit",
                "block",
                f"Sliding window reached {self.rate_limiter.max_requests}/"
                f"{self.rate_limiter.max_requests} requests",
                started,
            )
        else:
            add_trace(
                "Rate limit",
                "pass",
                "Per-user request remains inside the sliding-window quota",
                started,
            )

        # 2. Input guardrails.
        started = time.perf_counter()
        if blocked:
            add_trace("Input guard", "skip", "Stopped by an earlier layer", started)
        else:
            injection_status = detect_injection(text)
            topic_status = topic_filter(text)
            if injection_status == "BLOCK":
                blocked = True
                blocked_at = "input_injection"
                decision = "BLOCK"
                response_text = (
                    "Blue blocked an instruction-override or credential-extraction "
                    "request before the model call."
                )
                add_trace(
                    "Input guard",
                    "block",
                    "Normalized input matched an injection or secret-extraction rule",
                    started,
                )
            elif topic_status == "BLOCK":
                blocked = True
                blocked_at = "input_topic"
                decision = "BLOCK"
                response_text = "Blue only accepts VinBank banking-related requests."
                add_trace(
                    "Input guard",
                    "block",
                    "No allowed banking topic remained after normalization",
                    started,
                )
            else:
                add_trace(
                    "Input guard",
                    "pass",
                    "Injection and topic policies both returned ALLOW",
                    started,
                )

        # 3. Deterministic model stage: no external API call and no embedded secrets.
        started = time.perf_counter()
        if blocked:
            add_trace("Model", "skip", "No model call was made", started)
        else:
            response_text = self._model_response(scenario)
            add_trace(
                "Model",
                "pass",
                "Deterministic demo response generated without external API usage",
                started,
            )

        # 4. Output plugin. Inspect first so the trace can explain its mutation.
        started = time.perf_counter()
        if blocked:
            add_trace("Output guard", "skip", "No model output to inspect", started)
        else:
            filtered = content_filter(response_text)
            llm_response = SimpleNamespace(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text=response_text)]
                )
            )
            await self.output_guard.after_model_callback(
                callback_context=None,
                llm_response=llm_response,
            )
            response_text = _content_text(llm_response.content)
            if filtered["safe"]:
                add_trace(
                    "Output guard",
                    "pass",
                    "No PII, credential, or internal host pattern detected",
                    started,
                )
            else:
                decision = "REDACT"
                add_trace(
                    "Output guard",
                    "redact",
                    "Sensitive fields replaced: " + ", ".join(filtered["issues"]),
                    started,
                )

        # 5. Egress policy. Most scenarios have no side effect; the egress preset
        # exercises an unknown destination against the exact hostname allowlist.
        started = time.perf_counter()
        if blocked:
            add_trace("Egress / HITL", "skip", "No side effect can be initiated", started)
        elif scenario == "egress":
            destination = "https://collector.invalid/ingest"
            if not is_egress_allowed(destination, response_text):
                blocked = True
                blocked_at = "egress_allowlist"
                decision = "BLOCK"
                response_text = "Blue blocked data egress to an untrusted destination."
                add_trace(
                    "Egress / HITL",
                    "block",
                    "Destination hostname is absent from the exact VinBank allowlist",
                    started,
                )
        else:
            add_trace(
                "Egress / HITL",
                "pass",
                "No external side effect or privileged action requested",
                started,
            )

        audit_record = self.audit.record_output(
            user_id=user_id,
            text=response_text,
            blocked=blocked,
            layer=blocked_at,
            request_id=request_id,
        )
        trace.append({
            "name": "Audit",
            "state": "record",
            "reason": "Decision, policy layer, actor, and latency recorded",
            "duration_ms": audit_record["latency_ms"],
        })
        if len(self.audit.logs) > 100:
            self.audit.logs = self.audit.logs[-100:]

        meta = SCENARIO_META[scenario]
        return {
            "request_id": request_id,
            "scenario": scenario,
            "source": meta["source"],
            "scenario_label": meta["label"],
            "decision": decision,
            "blocked": blocked,
            "blocked_at": blocked_at,
            "response": response_text,
            "trace": trace,
            "audit": self._sanitize_audit(audit_record),
        }


class DemoRequestHandler(BaseHTTPRequestHandler):
    runtime = DemoRuntime()
    server_version = "VinBankDemo/1.0"

    def log_message(self, format: str, *args) -> None:
        # Keep request bodies out of terminal logs.
        sys.stdout.write(f"[demo] {self.address_string()} {format % args}\n")

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; object-src 'none'; base-uri 'none'"
        ))

    def _send_bytes(
        self,
        body: bytes,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str = "application/json; charset=utf-8",
    ) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            status=status,
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/slides/VinBank_Guardrails_Lab.html"}:
            self._send_bytes(
                DECK_PATH.read_bytes(),
                content_type="text/html; charset=utf-8",
            )
            return
        if parsed.path == "/api/health":
            self._send_json({
                "status": "ok",
                "mode": "python-policy",
                "guardrails": ["rate", "input", "output", "egress", "audit"],
            })
            return
        if parsed.path == "/api/audit":
            raw_limit = parse_qs(parsed.query).get("limit", ["20"])[0]
            try:
                limit = int(raw_limit)
            except ValueError:
                limit = 20
            self._send_json({"events": self.runtime.recent_audit(limit)})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path != "/api/evaluate":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            self._send_json(
                {"error": "Content-Type must be application/json"},
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
            return
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            content_length = 0
        if content_length < 1 or content_length > MAX_BODY_BYTES:
            self._send_json({"error": "invalid body size"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            result = asyncio.run(self.runtime.evaluate(
                str(payload.get("input", "")),
                scenario=str(payload.get("scenario", "safe")),
                user_id=str(payload.get("user_id", "presentation-user"))[:80],
            ))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception:
            self._send_json(
                {"error": "policy evaluation failed safely"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json(result)


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    server = HTTPServer((host, port), DemoRequestHandler)
    print(f"VinBank demo running at http://{host}:{port}/", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local VinBank guardrail demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
