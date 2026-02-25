"""Lightweight web-learning helpers using public encyclopedia/search summaries."""

from __future__ import annotations

import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, Tuple
from urllib import error, parse, request

WIKI_SUMMARY_API = "https://zh.wikipedia.org/api/rest_v1/page/summary/{}"
DUCKDUCKGO_API = "https://api.duckduckgo.com/?{}"
TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+|[\u4e00-\u9fff]{2,}")
STOPWORDS = {"是", "啥", "什么", "有哪些", "有啥", "一下", "介绍", "资料", "吗", "呢", "的"}
DEFAULT_DEEP_SYNC_SEEDS = [
    "人工智能",
    "机器学习",
    "深度学习",
    "Python",
    "数学",
    "物理学",
    "化学",
    "生物学",
    "历史",
    "地理",
    "经济学",
    "哲学",
    "编程",
    "网络安全",
    "数据库",
    "操作系统",
    "Minecraft",
    "王者荣耀",
    "英雄联盟",
    "红楼梦",
]


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


def discover_related_topics(topic: str) -> list[str]:
    """Discover related topics from DuckDuckGo for deep-sync expansion."""
    query = parse.urlencode({"q": topic, "format": "json", "no_html": 1, "skip_disambig": 0})
    url = DUCKDUCKGO_API.format(query)
    payload = _http_get_json(url)
    if not payload:
        return []

    results: list[str] = []
    related = payload.get("RelatedTopics")
    if not isinstance(related, list):
        return results

    for item in related:
        if isinstance(item, dict):
            text = item.get("Text")
            if isinstance(text, str) and text.strip():
                candidate = text.split(" - ")[0].strip()
                if 2 <= len(candidate) <= 40:
                    results.append(candidate)
            topics = item.get("Topics")
            if isinstance(topics, list):
                for sub in topics:
                    if isinstance(sub, dict):
                        sub_text = sub.get("Text")
                        if isinstance(sub_text, str) and sub_text.strip():
                            candidate = sub_text.split(" - ")[0].strip()
                            if 2 <= len(candidate) <= 40:
                                results.append(candidate)
        if len(results) >= 12:
            break

    dedup: list[str] = []
    seen: set[str] = set()
    for r in results:
        key = r.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    return dedup[:12]


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
        return {t for t in tokens if t not in STOPWORDS and len(t) >= 2}

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


    def deep_sync(self, seed_topics: list[str], time_budget_s: int = 120, max_topics: int = 200) -> dict:
        """Continuously crawl and cache web summaries for a period of time."""
        start = time.time()
        queue = deque([_normalize_topic(t) for t in seed_topics if _normalize_topic(t)])
        if not queue:
            queue = deque(DEFAULT_DEEP_SYNC_SEEDS)

        visited: set[str] = set()
        existing: set[str] = set(k.lower() for k in self.data.keys())
        learned_count = 0
        tried_count = 0
        expanded_from_cache = 0

        while queue and (time.time() - start) < time_budget_s and tried_count < max_topics:
            topic = queue.popleft()
            key = topic.lower()
            if key in visited:
                continue
            visited.add(key)

            # Even if topic is already cached, still try to expand related topics so startup
            # deep-sync can continue discovering new knowledge instead of ending immediately.
            if key in existing:
                expanded_from_cache += 1
                for related in discover_related_topics(topic):
                    rkey = related.lower()
                    if rkey not in visited:
                        queue.append(related)
                continue

            tried_count += 1

            ok, _ = self.learn_topic(topic)
            if ok:
                learned_count += 1
                for related in discover_related_topics(topic):
                    rkey = related.lower()
                    if rkey not in visited:
                        queue.append(related)

        return {
            "learned": learned_count,
            "tried": tried_count,
            "expanded_from_cache": expanded_from_cache,
            "remaining_queue": len(queue),
            "elapsed_s": round(time.time() - start, 2),
        }
