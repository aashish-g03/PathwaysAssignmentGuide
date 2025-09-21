import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Reply:
    intent: str
    text: str
    citations: List[str]
    set_filters: Optional[Dict[str, Any]] = None
    attachments: Optional[Dict[str, Any]] = None

class IntentRouter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def detect(self, q: str) -> str:
        qs = q.lower().strip()
        for intent, meta in self.config["intents"].items():
            for trig in meta.get("triggers", []):
                if trig in qs:
                    return intent
        if re.search(r"\bwhat is|define|cbd|scenario|unit|methodology\b", qs):
            return "definition"
        if re.search(r"\bget|show|download|export|filter|slice|compare|companies|company\b", qs):
            return "slice_request"
        if re.search(r"\bexplain|summarize|delta\b", qs):
            return "explain_view"
        if re.search(r"\bmissing|coverage|available|not showing|no data\b", qs):
            return "diagnostics"
        return "definition"
