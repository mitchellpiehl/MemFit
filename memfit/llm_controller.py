"""
llm_controller.py — LLM / MLLM backends with exact call and token accounting.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import requests


# ==============================================================================
# Call / token accounting
# ==============================================================================

@dataclass
class CallRecord:
    role: str                 # "planner" | "reader" | "judge" | "other"
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    ok: bool = True
    truncated: bool = False
    output_capped: bool = False
    transport_failed: bool = False


@dataclass
class CallLedger:
    records: List[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _scope_start: int = 0

    def record(self, rec: CallRecord) -> None:
        with self._lock:
            self.records.append(rec)

    # -- scoping ---------------------------------------------------------
    def begin_scope(self) -> None:
        with self._lock:
            self._scope_start = len(self.records)

    def scope_records(self) -> List[CallRecord]:
        with self._lock:
            return list(self.records[self._scope_start:])

    def scope_summary(self) -> Dict[str, Any]:
        recs = self.scope_records()
        by_role: Dict[str, int] = {}
        for r in recs:
            by_role[r.role] = by_role.get(r.role, 0) + 1
        return {
            "llm_calls": len(recs),
            "calls_by_role": by_role,
            "prompt_tokens": sum(r.prompt_tokens for r in recs),
            "completion_tokens": sum(r.completion_tokens for r in recs),
            "total_tokens": sum(r.prompt_tokens + r.completion_tokens for r in recs),
            "llm_latency_s": sum(r.latency_s for r in recs),
            "failures": sum(0 if r.ok else 1 for r in recs),
            "truncated": sum(1 for r in recs if r.truncated),
            "output_capped": sum(1 for r in recs if r.output_capped),
            "transport_failures": sum(1 for r in recs if r.transport_failed),
        }

    # -- totals ----------------------------------------------------------
    def totals(self) -> Dict[str, Any]:
        with self._lock:
            recs = list(self.records)
        by_role: Dict[str, int] = {}
        for r in recs:
            by_role[r.role] = by_role.get(r.role, 0) + 1
        return {
            "llm_calls": len(recs),
            "calls_by_role": by_role,
            "prompt_tokens": sum(r.prompt_tokens for r in recs),
            "completion_tokens": sum(r.completion_tokens for r in recs),
            "total_tokens": sum(r.prompt_tokens + r.completion_tokens for r in recs),
            "llm_latency_s": sum(r.latency_s for r in recs),
            "failures": sum(0 if r.ok else 1 for r in recs),
            "truncated": sum(1 for r in recs if r.truncated),
            "output_capped": sum(1 for r in recs if r.output_capped),
            "transport_failures": sum(1 for r in recs if r.transport_failed),
        }

    def reset(self) -> None:
        with self._lock:
            self.records.clear()
            self._scope_start = 0


# A single process-wide ledger keeps accounting simple for the harness.
LEDGER = CallLedger()

#: How long Ollama should hold a model in VRAM after a call. Long enough that a
#: gap between stages, or a slow non-LLM step, does not evict it; short enough
#: that a finished run releases the GPU on its own.
KEEP_ALIVE = os.environ.get("MEMFIT_KEEP_ALIVE", "30m")


def _approx_tokens(text: str) -> int:
    """Fallback token estimate when the backend does not report usage."""
    return max(1, len(text) // 4)

def answer_confidence(logprobs: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:

    if not logprobs:
        return {}

    lps: List[float] = []
    margins: List[float] = []
    for tok in logprobs:
        lp = tok.get("logprob")
        if lp is None:
            continue
        lps.append(float(lp))
        alts = tok.get("top_logprobs") or []
        # top_logprobs[0] is the chosen token itself; [1] is the runner-up.
        if len(alts) >= 2:
            a0 = alts[0].get("logprob")
            a1 = alts[1].get("logprob")
            if a0 is not None and a1 is not None:
                margins.append(float(a0) - float(a1))

    if not lps:
        return {}

    n = len(lps)
    out = {
        "lp_mean": sum(lps) / n,
        "lp_min": min(lps),
        "lp_sum": sum(lps),
        "n_tokens": float(n),
    }
    if margins:
        out["lp_margin_mean"] = sum(margins) / len(margins)
        out["lp_margin_min"] = min(margins)
    return out

def _safe_encode_image(img_path: str) -> str:
    """Encode an image as base64 JPEG (converts PNG/WebP/RGBA safely)."""
    try:
        from PIL import Image

        img = Image.open(img_path)
        if img.mode in ("RGBA", "P", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

class OllamaController:

    DEFAULT_NUM_CTX: int = 32768
    #: Waits (seconds) before each retry of a call that failed in transport —
    #: connection refused, timeout, HTTP 5xx. Without retries, an instance that
    #: restarted or wedged briefly returned "" for every call in that window,
    #: which scores as a wrong answer and looks exactly like a weaker model.
    RETRY_WAITS: Tuple[int, ...] = (5, 15, 45, 90)

    def __init__(
        self,
        model: str = "qwen3:8b",
        num_ctx: int = DEFAULT_NUM_CTX,
        host: str = "http://localhost:11434",
        timeout: int = 600,
        ledger: Optional[CallLedger] = None,
    ):
        self.model = model
        self.num_ctx = num_ctx
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.ledger = ledger if ledger is not None else LEDGER
        self.last_logprobs: Optional[List[Dict[str, Any]]] = None
        #: Whether the most recent call stopped because it ran out of output
        #: budget rather than finishing. Callers that score a final answer read
        #: this to tell "no answer" from "was still writing".
        self.last_output_capped = False
        #: Calls that failed even after every retry. Harnesses report this and
        #: the round report marks any arm with a non-zero count as invalid.
        self.hard_failures = 0

    @property
    def _is_thinking_model(self) -> bool:
        m = self.model.lower()
        if m.startswith("gpt-oss"):
            return False
        return (m.startswith("qwen3") or ":qwen3" in m
                or m.startswith("gemma4"))

    def _fits(self, prompt: str, num_predict: int) -> Tuple[bool, int]:
        est = int(len(prompt) / 3.6)
        budget = self.num_ctx - num_predict - 64      # 64 tokens of headroom
        return est <= budget, est

    def _post(self, payload: Dict[str, Any], role: str) -> str:
        num_predict = int(payload.get("options", {}).get("num_predict", 512))
        fits, est = self._fits(payload.get("prompt", ""), num_predict)
        if not fits:
            import logging
            logging.getLogger("llm_controller").warning(
                "prompt for role=%s is ~%d tokens but num_ctx=%d (predict=%d); "
                "the backend will truncate it silently",
                role, est, self.num_ctx, num_predict,
            )
        import logging
        log = logging.getLogger("llm_controller")
        t0 = time.time()
        err: Optional[BaseException] = None
        for attempt, wait in enumerate((0,) + self.RETRY_WAITS):
            if wait:
                time.sleep(wait)
            try:
                resp = requests.post(
                    f"{self.host}/api/generate", json=payload, timeout=self.timeout
                )
                if 400 <= resp.status_code < 500:
                    # A malformed request or a missing model will not improve
                    # on retry.
                    err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    break
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:          # connection, timeout, 5xx, bad JSON
                err = e
                log.warning("Ollama call failed (attempt %d of %d, %s): %s",
                            attempt + 1, len(self.RETRY_WAITS) + 1,
                            type(e).__name__, e)
                continue
            raw = data.get("response", "") or ""
            # Token logprobs come back in the same response, so reader
            # confidence costs nothing beyond a field read. Kept off the return
            # type so no call site has to change; read it with last_confidence().
            self.last_logprobs = data.get("logprobs") or None
            # Ollama reports why generation stopped: "stop" for a natural end,
            # "length" when `num_predict` ran out mid-sentence.
            self.last_output_capped = str(data.get("done_reason", "")) == "length"
            self.ledger.record(CallRecord(
                role=role,
                model=self.model,
                prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
                completion_tokens=int(data.get("eval_count", 0) or 0),
                latency_s=time.time() - t0,
                ok=bool(raw.strip()),
                truncated=not fits,
                output_capped=self.last_output_capped,
            ))
            return raw

        self.ledger.record(CallRecord(
            role=role,
            model=self.model,
            prompt_tokens=_approx_tokens(payload.get("prompt", "")),
            completion_tokens=0,
            latency_s=time.time() - t0,
            ok=False,
            transport_failed=True,
        ))
        self.hard_failures += 1
        log.error("Ollama call FAILED after all retries (%s): %s",
                  type(err).__name__ if err else "?", err)
        return ""

    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        role: str = "other",
        logprobs: bool = False,
        top_logprobs: int = 0,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Ollama unloads a model 5 minutes after its last call by default.
            # In a long run that is a reload in the middle of the stream:
            # measured on the shared box, the same 8-token call took 0.55 s
            # warm and 3.2 s after an eviction, which is a 6x slowdown that
            # looks like nothing in the results.
            "keep_alive": KEEP_ALIVE,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if stop:
            # AgentBoard's own llm config stops its agents at a newline, so one
            # call is one action.
            payload["options"]["stop"] = list(stop)
        if self._is_thinking_model:
            payload["think"] = False
        if logprobs:
            # Per-token logprobs cost ~30% extra; top-k alternatives cost 5-6x
            # (measured: 1.7 s plain, 7.5 s with top_logprobs=2, same 169
            # tokens). So alternatives only when a caller asks for margins.
            payload["logprobs"] = True
            if top_logprobs:
                payload["top_logprobs"] = int(top_logprobs)
        self.last_logprobs = None
        self.last_output_capped = False
        return self._post(payload, role)

    def last_confidence(self) -> Dict[str, float]:
        """Confidence features from the most recent call's token logprobs."""
        return answer_confidence(self.last_logprobs)

    def get_image_completion(
        self,
        prompt: str,
        image_paths: Any,
        temperature: float = 0.0,
        max_tokens: int = 512,
        role: str = "reader",
        **kwargs,
    ) -> str:
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        images_b64 = []
        for p in image_paths or []:
            try:
                images_b64.append(_safe_encode_image(p))
            except Exception as e:
                # Dropped, not fatal — but counted and said out loud: a caller
                # that believes it sent images must not be told otherwise by
                # silence.
                logging.getLogger("llm_controller").warning(
                    "image not sent (%s): %s", type(e).__name__, p)
        self.last_images_sent = len(images_b64)
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self._is_thinking_model:
            payload["think"] = False
        return self._post(payload, role)


class OpenAIController:
    """OpenAI chat-completions backend with the same ledger accounting."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        ledger: Optional[CallLedger] = None,
    ):
        from openai import OpenAI

        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY.")
        # Eight concurrent shards can exceed a lower-tier rate limit. The SDK
        # backs off exponentially and honours Retry-After, so with room to
        # retry a 429 slows the run down instead of costing an answer. The
        # default (2 retries) gives up within seconds under sustained load.
        self.client = OpenAI(api_key=api_key, max_retries=8)
        self.ledger = ledger if ledger is not None else LEDGER

    def _call(self, content: Any, temperature: float, max_tokens: int, role: str) -> str:
        t0 = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = getattr(response, "usage", None)
            text = response.choices[0].message.content or ""
            self.ledger.record(CallRecord(
                role=role,
                model=self.model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_s=time.time() - t0,
                ok=bool(text.strip()),
            ))
            return text
        except Exception as e:
            # transport_failed, not just ok=False: the empty string returned
            # here is scored as a wrong answer, and only transport_failures
            # makes the report mark the run invalid. Without it a rate-limited
            # run looks like a weaker model rather than a broken one.
            self.ledger.record(CallRecord(
                role=role, model=self.model, prompt_tokens=0,
                completion_tokens=0, latency_s=time.time() - t0, ok=False,
                transport_failed=True,
            ))
            import logging
            logging.getLogger("llm_controller").error("OpenAI error: %s", e)
            return ""

    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        role: str = "other",
        logprobs: bool = False,
        stop: Optional[List[str]] = None,
    ) -> str:
        # Accepted for interface parity; this path does not request logprobs.
        return self._call(prompt, temperature, max_tokens, role)

    def last_confidence(self) -> Dict[str, float]:
        return {}

    def get_image_completion(
        self,
        prompt: str,
        image_paths: Any,
        temperature: float = 0.0,
        max_tokens: int = 512,
        role: str = "reader",
        **kwargs,
    ) -> str:
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for p in image_paths or []:
            if os.path.exists(p):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_safe_encode_image(p)}"},
                })
        return self._call(content, temperature, max_tokens, role)


class LLMController:
    """Thin factory exposing the chosen backend as `.llm`."""

    _MODEL_CTX_DEFAULTS: Dict[str, int] = {
        "gemma3": 32768,
        "gemma4": 32768,
        "llama3.2": 16384,
        "llama3.1": 32768,
        "llama3": 16384,
        "mistral": 32768,
        "qwen": 32768,
        "phi": 16384,
    }

    @classmethod
    def _default_num_ctx(cls, model: str) -> int:
        for prefix, ctx in cls._MODEL_CTX_DEFAULTS.items():
            if model.lower().startswith(prefix):
                return ctx
        return OllamaController.DEFAULT_NUM_CTX

    def __init__(
        self,
        backend: Literal["openai", "ollama"] = "ollama",
        model: str = "qwen3:8b",
        api_key: Optional[str] = None,
        num_ctx: Optional[int] = None,
        ollama_host: str = "http://localhost:11434",
        ledger: Optional[CallLedger] = None,
    ):
        self.backend = backend
        self.model = model
        if backend == "openai":
            self.llm = OpenAIController(model, api_key, ledger=ledger)
        elif backend == "ollama":
            resolved_ctx = num_ctx if num_ctx is not None else self._default_num_ctx(model)
            self.llm = OllamaController(
                model, num_ctx=resolved_ctx, host=ollama_host, ledger=ledger
            )
        else:
            raise ValueError("Backend must be 'openai' or 'ollama'")
