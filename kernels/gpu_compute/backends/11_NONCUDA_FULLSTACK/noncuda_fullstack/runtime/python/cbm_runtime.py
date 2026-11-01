# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

import json
import time
from pathlib import Path
from typing import Dict, Any
from urllib import request


class CbmRuntime:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.runtime_dir = project_root / "runtime" / "noncuda"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.dev_log = self.runtime_dir / "cbm_dev_thinking.log"
        self.base_url = "http://127.0.0.1:1234"
        self.model = "essentialai/rnj-1"

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": int(time.time() * 1000), "event": event, **payload}
        with self.dev_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    def _post_json(self, endpoint: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        req = request.Request(
            url=f"{self.base_url}{endpoint}",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def think(self, probe: str) -> Dict[str, Any]:
        prompt = (
            "You are CBM runtime monitor. Return strict JSON with keys: "
            "thought,response,confusion_score,confidence. "
            "Use concise technical reasoning.\n"
            f"probe={probe}"
        )
        payload = {
            "model": self.model,
            "input": prompt,
            "reasoning": {"effort": "high"},
            "parallel_tool_calls": True,
        }
        try:
            result = self._post_json("/v1/responses", payload, timeout=90)
            text = result.get("output_text", "")
            parsed = self._parse_json(text)
            self._log("think", {"endpoint": "/v1/responses", "result": parsed})
            return parsed
        except Exception as e:
            fallback = {
                "thought": f"LM Studio call failed: {e}",
                "response": "fallback",
                "confusion_score": 50.0,
                "confidence": 0.2,
            }
            self._log("think_error", {"error": str(e), "fallback": fallback})
            return fallback

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            obj = json.loads(text)
        except Exception:
            obj = {
                "thought": text[:600],
                "response": "unstructured",
                "confusion_score": 50.0,
                "confidence": 0.5,
            }
        obj["confusion_score"] = float(max(0.0, min(100.0, obj.get("confusion_score", 50.0))))
        obj["confidence"] = float(max(0.0, min(1.0, obj.get("confidence", 0.5))))
        return obj

    def write_status(self, status: Dict[str, Any]) -> None:
        out = self.runtime_dir / "runtime_status.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
