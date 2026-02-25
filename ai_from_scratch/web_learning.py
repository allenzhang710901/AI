"""Lightweight web-learning helpers using public encyclopedia/search summaries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple
from urllib import error, parse, request

WIKI_SUMMARY_API = "https://zh.wikipedia.org/api/rest_v1/page/summary/{}"
DUCKDUCKGO_API = "https://api.duckduckgo.com/?{}"
TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+")
STOPWORDS = {"是", "啥", "什", "么", "有", "哪", "些", "一", "下", "介", "绍", "资", "料", "吗", "呢", "的"}


def _normalize_topic(topic: str) -> str:
    return re.sub(r"\s+", " ", topic).strip()


def _http_get_json(url: str) -> dict | None:
    req = request.Request(url, headers={"User-Agent": "ai-from-scratch/1.0"})
    try:
        with request.urlopen(req, timeout=8) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _fetch_wikipedia_summary(topic: str) -> str | None:
    encoded = parse.quote(topic)
    url = WIKI_SUMMARY_API.format(encoded)
    payload = _http_get_json(url)
    if not payload:
        return None

    extract = payload.get("extract")
    if not isinstance(extract, str) or not extract.strip():
        return None

    clean = extract.strip().replace("\n", " ")
    return clean[:500]


def _fetch_duckduckgo_summary(topic: str) -> str | None:
    query = parse.urlencode({"q": topic, "format": "json", "no_html": 1, "skip_disambig": 0})
    url = DUCKDUCKGO_API.format(query)
    payload = _http_get_json(url)
    if not payload:
        return None

    candidates: list[str] = []
    abstract = payload.get("AbstractText")
    if isinstance(abstract, str) and abstract.strip():
        candidates.append(abstract.strip())

    heading = payload.get("Heading")
    if isinstance(heading, str) and heading.strip() and candidates:
        candidates[0] = f"{heading.strip()}：{candidates[0]}"

    related = payload.get("RelatedTopics")
    if isinstance(related, list):
        for item in related:
            if isinstance(item, dict):
                text = item.get("Text")
                if isinstance(text, str) and text.strip():
                    candidates.append(text.strip())
                    break

    if not candidates:
        return None

    return candidates[0][:500]


def fetch_web_summary(topic: str) -> str | None:
    """Fetch a short summary from multiple public sources."""
    topic = _normalize_topic(topic)
    if not topic:
        return None

    summary = _fetch_wikipedia_summary(topic)
    if summary:
        return summary

    return _fetch_duckduckgo_summary(topic)


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

    def save(self) -> bool:
        try:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            return False
        return True

    def learn_topic(self, topic: str) -> Tuple[bool, str]:
        topic = _normalize_topic(topic)
        if not topic:
            return False, "请告诉我要学习的主题，例如：学习 人工智能。"

        summary = fetch_web_summary(topic)
        if not summary:
            return False, f"我暂时没能从网上学到“{topic}”，你可以换个更具体的关键词试试。"

        self.data[topic] = {"summary": summary, "source": "web"}
        if not self.save():
            return False, "我学到了内容，但本地保存失败（可能是目录写权限问题）。"
        return True, f"我刚刚上网学习了“{topic}”。"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        lowered = text.lower()
        tokens = {t for t in TOKEN_RE.findall(lowered) if t.strip()}
        chinese_chars = {ch for ch in lowered if "一" <= ch <= "鿿"}
        all_tokens = tokens.union(chinese_chars)
        return {t for t in all_tokens if t not in STOPWORDS and len(t) >= 1}

    def find_best_relevant(self, query: str) -> Tuple[str, str, float, list[str]] | None:
        query_tokens = self._tokens(query)
        if not query_tokens:
            return None

        best: Tuple[str, str, float, list[str]] | None = None
        for topic, item in self.data.items():
            summary = item["summary"]
            target_tokens = self._tokens(topic + " " + summary)
            if not target_tokens:
                continue
            matched = sorted([t for t in query_tokens if t in target_tokens])
            score = len(matched) / max(len(query_tokens), 1)
            missing = sorted([t for t in query_tokens if t not in target_tokens])
            candidate = (topic, summary, score, missing)
            if best is None or score > best[2]:
                best = candidate

        return best

    def find_relevant(self, query: str) -> Tuple[str, str] | None:
        best = self.find_best_relevant(query)
        if not best:
            return None
        topic, summary, score, _ = best
        if score < 0.5:
            return None
        return topic, summary
