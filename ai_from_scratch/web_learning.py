"""Lightweight web-learning helpers using public encyclopedia summaries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple
from urllib import error, parse, request

WIKI_SUMMARY_API = "https://zh.wikipedia.org/api/rest_v1/page/summary/{}"


def _normalize_topic(topic: str) -> str:
    return re.sub(r"\s+", " ", topic).strip()


def fetch_wikipedia_summary(topic: str) -> str | None:
    """Fetch a short Chinese summary from Wikipedia REST API."""
    topic = _normalize_topic(topic)
    if not topic:
        return None

    encoded = parse.quote(topic)
    url = WIKI_SUMMARY_API.format(encoded)
    req = request.Request(url, headers={"User-Agent": "ai-from-scratch/1.0"})

    try:
        with request.urlopen(req, timeout=8) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    extract = payload.get("extract")
    if not isinstance(extract, str) or not extract.strip():
        return None

    clean = extract.strip().replace("\n", " ")
    return clean[:500]


class WebKnowledgeBase:
    """Persistent local cache for web-learned knowledge snippets."""

    def __init__(self, path: str | Path = ".ai_web_knowledge.json") -> None:
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> Dict[str, Dict[str, str]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        cleaned: Dict[str, Dict[str, str]] = {}
        for topic, item in payload.items():
            if not isinstance(topic, str) or not isinstance(item, dict):
                continue
            summary = item.get("summary")
            source = item.get("source", "")
            if isinstance(summary, str) and summary.strip():
                cleaned[topic] = {"summary": summary.strip(), "source": str(source)}
        return cleaned

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def learn_topic(self, topic: str) -> Tuple[bool, str]:
        topic = _normalize_topic(topic)
        if not topic:
            return False, "请告诉我要学习的主题，例如：学习 人工智能。"

        summary = fetch_wikipedia_summary(topic)
        if not summary:
            return False, f"我暂时没能从网上学到“{topic}”，你可以换个更具体的关键词试试。"

        self.data[topic] = {"summary": summary, "source": "zh.wikipedia.org"}
        self.save()
        return True, f"我刚刚上网学习了“{topic}”。"

    def find_relevant(self, query: str) -> Tuple[str, str] | None:
        q = query.lower()
        for topic, item in self.data.items():
            if topic.lower() in q:
                return topic, item["summary"]
        return None
