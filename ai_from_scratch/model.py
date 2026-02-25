"""A smarter but still lightweight Chinese intent assistant."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .data import TRAINING_DATA
from .web_learning import WebKnowledgeBase

TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+|[+\-*/()]")
KEYWORD_HINTS = {
    "greeting": ["你好", "哈喽", "在吗", "在么", "嗨", "hello", "hi"],
    "math": ["算", "计算", "多少", "加", "减", "乘", "除", "+", "-", "*", "/"],
    "weather": ["天气", "温度", "下雨", "晴", "阴", "热", "冷", "风"],
    "recommend": ["推荐", "建议", "学什么", "看什么", "怎么选", "路线"],
    "goodbye": ["再见", "拜拜", "退出", "结束", "下次聊"],
}
MATH_EXPR_RE = re.compile(r"[\d\s+\-*/().]+")
MIN_CONFIDENCE = 0.45
MAX_EXPR_LEN = 60
MAX_NUMBER_DIGITS = 12
AUTO_LEARN_MIN_CONFIDENCE = 0.9
AUTO_SAVE_EVERY = 3
WEB_LOOKUP_CONFIDENCE_THRESHOLD = 0.8
RESPONSE_MEMORY_MIN_CONFIDENCE = 0.72
MAX_RESPONSE_MEMORY = 500
THINKING_CUE_WORDS = ["思路", "步骤", "方法", "先", "然后", "最后", "框架", "拆解", "验证"]
REPLICATE_CUE_WORDS = ["复刻", "照着", "按这个思路", "按这个逻辑", "模仿", "套用"]
THINK_REQUEST_CUES = ["思考", "分析", "为什么", "怎么办", "怎么做", "规划", "方案", "决策"]
CAPABILITY_GAP_CUES = ["跟你一样聪明", "几乎一样聪明", "更聪明", "太傻", "死板"]
EMOTION_CUES = {
    "stressed": ["焦虑", "压力", "崩溃", "烦", "累", "慌", "难受", "迷茫"],
    "sad": ["难过", "伤心", "失落", "痛苦", "低落", "孤独"],
    "angry": ["生气", "愤怒", "火大", "烦死", "讨厌", "受不了"],
    "positive": ["开心", "高兴", "太好了", "有希望", "顺利", "感谢"],
}


@dataclass
class Prediction:
    intent: str
    confidence: float


class NaiveBayesIntentClassifier:
    """Character/word token based multinomial Naive Bayes classifier."""

    def __init__(self) -> None:
        self.intent_priors: Dict[str, float] = {}
        self.token_counts: Dict[str, Counter] = {}
        self.total_tokens: Dict[str, int] = {}
        self.vocab: set[str] = set()
        self.is_fitted = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        lowered = text.lower()
        tokens = TOKEN_RE.findall(lowered)
        chinese_chars = [ch for ch in lowered if "\u4e00" <= ch <= "\u9fff"]
        return tokens + chinese_chars

    def fit(self, samples: Dict[str, Iterable[str]]) -> None:
        cached = {intent: list(texts) for intent, texts in samples.items()}
        total_texts = sum(len(texts) for texts in cached.values())
        if total_texts == 0:
            raise ValueError("No samples provided.")

        for intent, texts in cached.items():
            self.intent_priors[intent] = math.log(len(texts) / total_texts)
            counter = Counter()
            for text in texts:
                tokens = self.tokenize(text)
                counter.update(tokens)
                self.vocab.update(tokens)
            self.token_counts[intent] = counter
            self.total_tokens[intent] = sum(counter.values())

        self.is_fitted = True

    def to_dict(self) -> Dict[str, object]:
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit first.")
        return {
            "intent_priors": self.intent_priors,
            "token_counts": {intent: dict(counter) for intent, counter in self.token_counts.items()},
            "total_tokens": self.total_tokens,
            "vocab": sorted(self.vocab),
        }

    def save(self, path: str | Path) -> None:
        data = self.to_dict()
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "NaiveBayesIntentClassifier":
        model = cls()
        model.intent_priors = {k: float(v) for k, v in dict(data["intent_priors"]).items()}
        model.token_counts = {k: Counter(v) for k, v in dict(data["token_counts"]).items()}
        model.total_tokens = {k: int(v) for k, v in dict(data["total_tokens"]).items()}
        model.vocab = set(data["vocab"])
        model.is_fitted = True
        return model

    @classmethod
    def load(cls, path: str | Path) -> "NaiveBayesIntentClassifier":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def predict(self, text: str) -> Prediction:
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit first.")

        tokens = self.tokenize(text)
        if not tokens:
            return Prediction(intent="unknown", confidence=0.0)

        vocab_size = max(len(self.vocab), 1)
        log_probs: Dict[str, float] = {}

        for intent, prior in self.intent_priors.items():
            log_prob = prior
            total = self.total_tokens[intent]
            counts = self.token_counts[intent]
            for token in tokens:
                token_likelihood = (counts[token] + 1) / (total + vocab_size)
                log_prob += math.log(token_likelihood)
            log_probs[intent] = log_prob

        best_intent = max(log_probs, key=log_probs.get)
        confidence = self._softmax_confidence(log_probs, best_intent)
        return Prediction(intent=best_intent, confidence=confidence)

    @staticmethod
    def _softmax_confidence(log_probs: Dict[str, float], target: str) -> float:
        max_log = max(log_probs.values())
        exps = {k: math.exp(v - max_log) for k, v in log_probs.items()}
        total = sum(exps.values())
        if total == 0:
            return 0.0
        return exps[target] / total


class SimpleChineseAIAssistant:
    """Assistant with hybrid intent routing + Naive Bayes fallback + auto learning."""

    def __init__(
        self,
        classifier: NaiveBayesIntentClassifier | None = None,
        learned_data_path: str | Path = ".ai_learned_data.json",
        auto_learn: bool = True,
        web_learning_enabled: bool = False,
        response_memory_path: str | Path = ".ai_response_memory.json",
    ) -> None:
        self.learned_data_path = Path(learned_data_path)
        self.auto_learn = auto_learn
        self.web_learning_enabled = web_learning_enabled
        self.web_knowledge = WebKnowledgeBase()
        self.response_memory_path = Path(response_memory_path)
        self.learned_data = self._load_learned_data()
        self.response_memory = self._load_response_memory()
        self.classifier = classifier or NaiveBayesIntentClassifier()
        if not self.classifier.is_fitted:
            self._refit_with_learned_data()
        self.last_intent: str | None = None
        self.last_user_text: str = ""
        self.user_name: str = ""
        self.new_samples_since_save = 0

    @classmethod
    def from_model_file(cls, model_path: str | Path) -> "SimpleChineseAIAssistant":
        classifier = NaiveBayesIntentClassifier.load(model_path)
        return cls(classifier=classifier)

    def _load_learned_data(self) -> Dict[str, List[str]]:
        if not self.learned_data_path.exists():
            return {}
        try:
            payload = json.loads(self.learned_data_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        if not isinstance(payload, dict):
            return {}
        cleaned: Dict[str, List[str]] = {}
        for intent, items in payload.items():
            if isinstance(intent, str) and isinstance(items, list):
                cleaned[intent] = [x for x in items if isinstance(x, str) and x.strip()]
        return cleaned

    def _load_response_memory(self) -> Dict[str, str]:
        if not self.response_memory_path.exists():
            return {}
        try:
            payload = json.loads(self.response_memory_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        cleaned: Dict[str, str] = {}
        for q, a in payload.items():
            if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
                cleaned[q.strip()] = a.strip()
        return cleaned

    def persist_learned_data(self) -> bool:
        try:
            self.learned_data_path.write_text(
                json.dumps(self.learned_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            return False
        return True

    def persist_response_memory(self) -> bool:
        try:
            self.response_memory_path.write_text(
                json.dumps(self.response_memory, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            return False
        return True

    def persist_all_memory(self) -> bool:
        ok1 = self.persist_learned_data()
        ok2 = self.persist_response_memory()
        return ok1 and ok2

    def startup_deep_sync(self, seconds: int, seed_topics: list[str] | None = None) -> dict:
        if not self.web_learning_enabled:
            return {"learned": 0, "tried": 0, "remaining_queue": 0, "elapsed_s": 0.0}
        # If caller doesn't provide seeds, use WebKnowledgeBase defaults so we still crawl
        # new topics instead of only replaying existing cache keys.
        topics = seed_topics or []
        stats = self.web_knowledge.deep_sync(seed_topics=topics, time_budget_s=seconds, max_topics=max(200, seconds * 5))
        return stats

    def _merged_training_data(self) -> Dict[str, List[str]]:
        merged = {intent: list(samples) for intent, samples in TRAINING_DATA.items()}
        for intent, samples in self.learned_data.items():
            merged.setdefault(intent, [])
            seen = set(merged[intent])
            for text in samples:
                if text not in seen:
                    merged[intent].append(text)
                    seen.add(text)
        return merged

    def _refit_with_learned_data(self) -> None:
        merged = self._merged_training_data()
        self.classifier.fit(merged)

    @staticmethod
    def _safe_eval_expression(text: str) -> str | None:
        expr_match = MATH_EXPR_RE.findall(text)
        if not expr_match:
            return None
        expr = "".join(expr_match).strip()
        if not expr or not re.fullmatch(r"[\d\s+\-*/().]+", expr):
            return None
        if len(expr) > MAX_EXPR_LEN:
            return None
        number_tokens = re.findall(r"\d+", expr)
        if any(len(n) > MAX_NUMBER_DIGITS for n in number_tokens):
            return None
        try:
            result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        except Exception:
            return None
        return str(result)

    @staticmethod
    def _keyword_intent(text: str) -> str | None:
        lowered = text.lower()
        best_intent = None
        best_score = 0
        for intent, words in KEYWORD_HINTS.items():
            score = sum(1 for word in words if word in lowered)
            if score > best_score:
                best_score = score
                best_intent = intent
        if best_score == 0:
            return None
        return best_intent

    @staticmethod
    def _closest_training_intent(text: str) -> Tuple[str, float]:
        best_intent = "unknown"
        best_ratio = 0.0
        for intent, samples in TRAINING_DATA.items():
            for sample in samples:
                ratio = SequenceMatcher(None, text, sample).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_intent = intent
        return best_intent, best_ratio

    @staticmethod
    def _is_low_signal_input(text: str) -> bool:
        stripped = text.strip()
        return bool(re.fullmatch(r"\d+", stripped) or len(stripped) == 1 and stripped not in {"q", "Q"})

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace("在么", "在吗").replace("嗎", "吗")

    def _route_intent(self, user_text: str, bayes_pred: Prediction) -> Prediction:
        if user_text.strip().lower() in {"q", "quit", "exit"}:
            return Prediction("goodbye", 1.0)

        if self._is_low_signal_input(user_text):
            return Prediction("unknown", 0.0)

        keyword_intent = self._keyword_intent(user_text)
        if keyword_intent:
            boost = 0.85 if keyword_intent != bayes_pred.intent else max(0.9, bayes_pred.confidence)
            return Prediction(keyword_intent, boost)

        nearest_intent, nearest_score = self._closest_training_intent(user_text)
        if nearest_score >= 0.72:
            return Prediction(nearest_intent, nearest_score)

        if bayes_pred.confidence < 0.24 and self.last_intent in {"math", "weather"}:
            return Prediction(self.last_intent, 0.35)

        if bayes_pred.confidence < MIN_CONFIDENCE:
            return Prediction("unknown", bayes_pred.confidence)

        return bayes_pred

    @staticmethod
    def _identity_like_text(text: str) -> bool:
        patterns = ["我是", "我不是", "我觉得", "我很", "我有点", "我想说"]
        lowered = text.lower()
        return any(p in lowered for p in patterns)

    def _should_auto_learn(self, text: str, prediction: Prediction) -> bool:
        if not self.auto_learn:
            return False
        if prediction.intent in {"unknown", "goodbye"}:
            return False
        if prediction.confidence < AUTO_LEARN_MIN_CONFIDENCE:
            return False
        if self._is_low_signal_input(text) or self._identity_like_text(text):
            return False
        if len(text.strip()) < 2 or len(text.strip()) > 40:
            return False
        if prediction.intent == "math" and self._safe_eval_expression(text) is None:
            return False
        return True

    def _learn_from_interaction(self, text: str, prediction: Prediction) -> None:
        if not self._should_auto_learn(text, prediction):
            return
        bucket = self.learned_data.setdefault(prediction.intent, [])
        if text in bucket:
            return
        bucket.append(text)
        self.new_samples_since_save += 1
        self._refit_with_learned_data()
        if self.new_samples_since_save >= AUTO_SAVE_EVERY:
            self.persist_learned_data()
            self.new_samples_since_save = 0

    @staticmethod
    def _parse_learn_command(text: str) -> str | None:
        raw = text.strip()
        prefixes = ["学习 ", "上网学习 ", "联网学习 ", "learn "]
        for prefix in prefixes:
            if raw.lower().startswith(prefix.lower()):
                return raw[len(prefix):].strip()
        return None

    def _try_answer_with_web_knowledge(self, text: str) -> str | None:
        best = self.web_knowledge.find_best_relevant(text)
        if not best:
            return None
        topic, summary, score, missing = best
        if score < 0.7:
            return None
        if missing:
            return f"我知道“{topic}”的基础信息，但你提到的关键词覆盖还不够（缺少: {', '.join(missing[:3])}）。你可以让我继续联网深挖。"
        return f"我之前上网学过“{topic}”：{summary}"

    @staticmethod
    def _is_personal_identity_question(text: str) -> bool:
        patterns = ["我是谁", "我叫什么", "我算什么", "我是什么"]
        lowered = text.lower().replace(" ", "")
        return any(p in lowered for p in patterns)

    @staticmethod
    def _is_non_topic_prompt(text: str) -> bool:
        lowered = text.strip().lower().replace(" ", "")
        prompts = {"你去查啊", "你去搜啊", "去查", "去搜", "帮我查", "查一下"}
        return lowered in prompts

    @staticmethod
    def _canonical_entity(text: str) -> str:
        lowered = text.lower()
        alias = {
            "honor of king": "honor of kings",
            "honor of kings": "honor of kings",
            "王者荣耀": "honor of kings",
            "arena of valor": "honor of kings",
            "我的世界": "minecraft",
            "minecraft": "minecraft",
        }
        for k, v in alias.items():
            if k in lowered or k in text:
                return v
        return ""

    def _try_compare_entities_question(self, text: str) -> str | None:
        lowered = text.lower()
        cues = ["same as", "一样吗", "是不是", "是否是", "同一个"]
        if not any(c in lowered for c in cues):
            return None
        canonical = self._canonical_entity(text)
        if canonical == "honor of kings" and ("王者荣耀" in text or "honor" in lowered):
            return "是的，Honor of Kings 基本就是《王者荣耀》的国际英文名/对应版本体系。"
        return None

    @staticmethod
    def _extract_knowledge_topic(text: str) -> str:
        alias_map = {
            "我的世界": "minecraft",
            " mc ": " minecraft ",
            "王者荣耀": "王者荣耀",
        }
        stripped = f" {text.strip()} "
        lowered = stripped.lower()
        for k, v in alias_map.items():
            lowered = lowered.replace(k, f" {v} ")

        topic = re.sub(r"[，。！？,.!?]", " ", lowered.strip())
        topic = re.sub(r"^(请问|你知道|能说下|介绍下|告诉我)", "", topic).strip()
        topic = re.sub(r"(是啥|是什么|啥意思|是什么东西|怎么样|资料|介绍|有哪些)$", "", topic).strip()
        return re.sub(r"\s+", " ", topic)

    @staticmethod
    def _extract_reasoning_snippets(summary: str) -> list[str]:
        parts = re.split(r"[。；;!?！？]", summary)
        snippets: list[str] = []
        for part in parts:
            text = part.strip()
            if not text or len(text) < 8:
                continue
            if any(word in text for word in THINKING_CUE_WORDS):
                snippets.append(text)
        return snippets[:3]

    def _reasoning_hint_for_query(self, text: str) -> str | None:
        if not self.web_learning_enabled:
            return None

        best = self.web_knowledge.find_best_relevant(text)
        if best:
            _, summary, score, _ = best
            if score >= 0.45:
                snippets = self._extract_reasoning_snippets(summary)
                if snippets:
                    return f"我会参考已学到的思考方法：{snippets[0]}。"

        # Fallback: if query matching is weak, still reuse any learned reasoning pattern
        # so web-learned knowledge can influence later conversations.
        for item in self.web_knowledge.data.values():
            summary = item.get("summary", "")
            if not isinstance(summary, str):
                continue
            snippets = self._extract_reasoning_snippets(summary)
            if snippets:
                return f"我会参考已学到的思考方法：{snippets[0]}。"
        return None

    @staticmethod
    def _is_reasoning_search_request(text: str) -> bool:
        lowered = text.lower().replace(" ", "")
        cues = ["ai的逻辑思考", "ai思考逻辑", "逻辑思考逻辑", "思考框架", "推理逻辑", "思维链路"]
        return any(c in lowered for c in cues)

    @staticmethod
    def _is_replicate_request(text: str) -> bool:
        lowered = text.lower()
        return any(c in lowered for c in REPLICATE_CUE_WORDS)

    @staticmethod
    def _is_thinking_request(text: str) -> bool:
        lowered = text.lower()
        return any(c in lowered for c in THINK_REQUEST_CUES)

    @staticmethod
    def _is_capability_gap_request(text: str) -> bool:
        lowered = text.lower().replace(" ", "")
        return any(c in lowered for c in CAPABILITY_GAP_CUES)

    @staticmethod
    def _extract_goal(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^(请|请你|帮我|麻烦你|我想|我需要)", "", cleaned)
        cleaned = re.sub(r"(怎么做|怎么办|怎么学|如何|请分析|请思考)$", "", cleaned)
        cleaned = cleaned.strip(" ，。！？,.!?")
        return cleaned or "当前问题"

    def _extract_user_name(self, text: str) -> None:
        stripped = text.strip()
        if stripped.startswith("学习回答") or stripped.startswith("记住回答"):
            return
        m = re.match(r"^(?:你好[，,\s]*)?我是\s*([A-Za-z0-9_一-鿿]{2,20})", stripped)
        if not m:
            return
        name = m.group(1).strip()
        if name:
            self.user_name = name

    def _propose_two_options(self, goal: str) -> tuple[str, str]:
        if any(k in goal for k in ["学", "学习", "路线", "入门", "AI", "ai"]):
            return (
                "方案A（稳）：先补基础（Python/数学），再做2个小项目，最后上模型实战。",
                "方案B（快）：直接做一个小助手项目，边做边补短板，2周一迭代。",
            )
        if any(k in goal for k in ["产品", "项目", "开发", "写", "代码"]):
            return (
                "方案A（质量优先）：先定义需求与验收标准，再分模块开发。",
                "方案B（速度优先）：先做最小可用版本（MVP），再按反馈迭代。",
            )
        return (
            "方案A（保守）：先拆小目标，优先解决确定性最高的部分。",
            "方案B（激进）：先做最关键一步验证可行性，再补全细节。",
        )

    def _build_thinking_response(self, text: str) -> str:
        goal = self._extract_goal(text)
        reasoning_hint = self._reasoning_hint_for_query(text)
        plan_a, plan_b = self._propose_two_options(goal)
        lines = [
            self._emotion_prefix(text),
            f"1) 目标定义：先明确“{goal}”的结果标准。",
            "2) 关键约束：时间/能力/资源分别有什么限制。",
            f"3) 方案对比：{plan_a}",
            f"   - {plan_b}",
            "4) 执行与复盘：先做最小一步，观察结果后再调整。",
        ]
        if reasoning_hint:
            lines.append(reasoning_hint)
        lines.append("如果你愿意，我可以继续把这题拆成 3~5 条你今天就能做的动作。")
        lines.append(self._follow_up_question(text))
        return "\n".join(lines)

    def _capability_upgrade_answer(self) -> str:
        return (
            "你这个目标是对的：想接近更强模型，核心不是一句话，而是“能力栈升级”。\n"
            "我可以按这条路线让它明显变聪明：\n"
            "1) 检索升级：接入更稳定多源搜索与更强重排。\n"
            "2) 记忆升级：长期记住你的偏好、项目、历史决策。\n"
            "3) 推理升级：每次回答都走‘目标-约束-方案-验证’。\n"
            "4) 评估升级：加自动测试题，持续打分迭代。\n"
            "你现在就可以让我先从第3步开始，先把你当前目标拆成可执行计划。"
        )

    @staticmethod
    def _detect_emotion(text: str) -> str | None:
        lowered = text.lower()
        for mood, words in EMOTION_CUES.items():
            if any(w in lowered for w in words):
                return mood
        return None

    def _emotion_prefix(self, text: str) -> str:
        mood = self._detect_emotion(text)
        prefix_name = f"{self.user_name}，" if self.user_name else ""
        if mood == "stressed":
            return f"{prefix_name}我听出来你现在压力很大，我们先把问题压缩到可执行的一步。"
        if mood == "sad":
            return f"{prefix_name}我在，你不是一个人。我们先把眼前最小可控的一步找出来。"
        if mood == "angry":
            return f"{prefix_name}我理解你现在很烦，我们先不空谈，直接做可落地方案。"
        if mood == "positive":
            return f"{prefix_name}你这个状态很好，我们可以趁势把计划做得更扎实。"
        return f"{prefix_name}我不只给结论，我先把思考过程展开："

    def _follow_up_question(self, text: str) -> str:
        goal = self._extract_goal(text)
        if any(k in goal for k in ["学习", "学", "路线", "入门", "AI", "ai"]):
            return "你可以补充：你每天可投入多少小时、你更偏工程还是理论？"
        if any(k in goal for k in ["产品", "项目", "开发", "上线"]):
            return "你可以补充：目标用户是谁、你希望几周内上线第一版？"
        return "你可以补充：你的时间上限、最想优先解决的结果是什么？"

    def _emotion_coaching_response(self, text: str) -> str | None:
        mood = self._detect_emotion(text)
        if not mood:
            return None
        if mood == "positive":
            return "听到这个好消息真替你开心！要不要我们趁热打铁，把下一步目标拆成今天可执行的 3 步？"
        return self._build_thinking_response(text)

    def _replicate_with_reasoning(self, text: str) -> str:
        base = re.sub(r"(复刻|照着|按这个思路|按这个逻辑|模仿|套用)", "", text, flags=re.IGNORECASE).strip(" ：:，,。")
        target = base if base else "你的问题"

        hint = self._reasoning_hint_for_query(text)
        if hint:
            plan = hint.replace("我会参考已学到的思考方法：", "").strip("。")
            return (
                f"可以，我按这个逻辑复刻回答：\n"
                f"1) 明确目标：先定义“{target}”要解决什么。\n"
                f"2) 拆解步骤：把问题拆成可执行的小步骤。\n"
                f"3) 逐步验证：每一步都检查是否贴合目标。\n"
                f"参考思路：{plan}。"
            )

        return (
            "可以复刻，但你需要先让我学到一个明确的思考模板。\n"
            "你可以先说：学习 AI 思考框架，然后再发：复刻 + 你的问题。"
        )

    def _try_web_lookup_for_query(self, text: str, prediction: Prediction) -> Tuple[str | None, bool]:
        if not self.web_learning_enabled:
            return None, False
        if prediction.confidence >= WEB_LOOKUP_CONFIDENCE_THRESHOLD:
            return None, False
        if self._is_personal_identity_question(text):
            return "这个问题更像是在表达自我困惑，我可以陪你梳理想法；也可以问我具体知识问题。", True
        if self._identity_like_text(text):
            return None, False
        if self._is_non_topic_prompt(text):
            return "你可以直接说完整主题，例如：'minecraft mods 是什么' 或 '红楼梦主要人物'。", True

        topic = self._extract_knowledge_topic(text)
        if len(topic) < 2 or self._is_low_signal_input(topic):
            return None, False

        best = self.web_knowledge.find_best_relevant(topic)
        if best:
            matched_topic, summary, score, missing = best
            if score >= 0.75 and len(missing) <= 1:
                return f"我在线知识库里有相关内容：{summary}", True

        ok, msg = self.web_knowledge.learn_topic(topic)
        if not ok:
            return f"我尝试联网查“{topic}”但暂时没结果。你可以补充更多关键词（例如版本/玩法/平台）。", True

        learned = self.web_knowledge.find_relevant(topic)
        if not learned:
            return msg, True
        _, summary = learned
        return f"我刚联网查到：{summary}", True

    @staticmethod
    def _parse_teach_response_command(text: str) -> Tuple[str, str] | None:
        raw = text.strip()
        prefixes = ["学习回答 ", "记住回答 ", "teach ", "teach-answer "]
        body = ""
        for prefix in prefixes:
            if raw.lower().startswith(prefix.lower()):
                body = raw[len(prefix):].strip()
                break
        if not body:
            return None
        if "=>" not in body:
            return None
        q, a = body.split("=>", 1)
        q, a = q.strip(), a.strip()
        if not q or not a:
            return None
        return q, a

    def _find_memorized_response(self, text: str) -> str | None:
        query = text.strip()
        if not query:
            return None
        if query in self.response_memory:
            return self.response_memory[query]

        best_answer = None
        best_score = 0.0
        for q, a in self.response_memory.items():
            score = SequenceMatcher(None, query, q).ratio()
            if score > best_score:
                best_score = score
                best_answer = a
        if best_score >= 0.9:
            return best_answer
        return None

    def _remember_response(self, question: str, answer: str, prediction: Prediction, force: bool = False) -> None:
        q = question.strip()
        a = answer.strip()
        if not q or not a:
            return
        if len(q) < 2 or len(q) > 80 or len(a) > 600:
            return
        if self._is_low_signal_input(q):
            return
        if not force:
            if prediction.confidence < RESPONSE_MEMORY_MIN_CONFIDENCE:
                return
            if prediction.intent in {"unknown", "goodbye"}:
                return

        if q not in self.response_memory and len(self.response_memory) >= MAX_RESPONSE_MEMORY:
            oldest = next(iter(self.response_memory))
            self.response_memory.pop(oldest, None)
        self.response_memory[q] = a
    def _think_before_answer(self, text: str) -> dict:
        """Internal thinking step before answer/search (not exposed verbosely)."""
        return {
            "normalized": self._normalize_text(text),
            "is_identity": self._identity_like_text(text),
            "is_prompt_only": self._is_non_topic_prompt(text),
            "candidate_topic": self._extract_knowledge_topic(text),
        }

    def reply(self, user_text: str) -> Tuple[str, Prediction]:
        thought = self._think_before_answer(user_text)
        normalized_text = thought["normalized"]

        self.last_user_text = normalized_text
        self._extract_user_name(normalized_text)

        teach_pair = self._parse_teach_response_command(normalized_text)
        if teach_pair is not None:
            q, a = teach_pair
            self._remember_response(q, a, Prediction(intent="recommend", confidence=1.0), force=True)
            self.persist_response_memory()
            return "我记住了。以后你问这个问题，我会优先按你教的方式回答。", Prediction(intent="recommend", confidence=1.0)

        memorized = self._find_memorized_response(normalized_text)
        if memorized:
            return memorized, Prediction(intent="recommend", confidence=0.95)

        emotion_answer = self._emotion_coaching_response(normalized_text)
        if emotion_answer:
            pred = Prediction(intent="recommend", confidence=0.86)
            self._remember_response(normalized_text, emotion_answer, pred)
            return emotion_answer, pred

        if self._is_capability_gap_request(normalized_text):
            answer = self._capability_upgrade_answer()
            pred = Prediction(intent="recommend", confidence=0.92)
            self._remember_response(normalized_text, answer, pred)
            return answer, pred

        compare_answer = self._try_compare_entities_question(normalized_text)
        if compare_answer:
            return compare_answer, Prediction(intent="recommend", confidence=0.9)

        learn_topic = self._parse_learn_command(normalized_text)
        if learn_topic is not None:
            if not self.web_learning_enabled:
                return (
                    "当前未开启联网学习。请用 --web-learn 启动后再输入：学习 人工智能。",
                    Prediction(intent="unknown", confidence=0.0),
                )
            ok, msg = self.web_knowledge.learn_topic(learn_topic)
            conf = 1.0 if ok else 0.0
            return msg, Prediction(intent="recommend", confidence=conf)

        if self._is_reasoning_search_request(normalized_text):
            if not self.web_learning_enabled:
                return (
                    "你要的“AI 逻辑思考”可以做，但要先开启联网学习：python main.py --web-learn",
                    Prediction(intent="unknown", confidence=0.0),
                )
            ok, _ = self.web_knowledge.learn_topic("AI 思考框架")
            if not ok:
                return (
                    "我已尝试联网学习 AI 思考逻辑，但这次没拿到稳定结果。你可以再试：学习 AI 思考框架。",
                    Prediction(intent="recommend", confidence=0.6),
                )
            return (
                "我已联网学习 AI 思考逻辑。你现在可以直接说“复刻 + 你的问题”，我会按学到的思路回答。",
                Prediction(intent="recommend", confidence=0.9),
            )

        if self._is_replicate_request(normalized_text):
            answer = self._replicate_with_reasoning(normalized_text)
            pred = Prediction(intent="recommend", confidence=0.88)
            self._remember_response(normalized_text, answer, pred)
            return answer, pred

        remembered = self._try_answer_with_web_knowledge(normalized_text)
        if remembered:
            pred = Prediction(intent="recommend", confidence=0.75)
            self._remember_response(normalized_text, remembered, pred)
            return remembered, pred
        bayes_pred = self.classifier.predict(normalized_text)
        prediction = self._route_intent(normalized_text, bayes_pred)
        intent = prediction.intent

        web_answer, _ = self._try_web_lookup_for_query(normalized_text, prediction)
        if web_answer:
            reasoning_hint = self._reasoning_hint_for_query(normalized_text)
            if reasoning_hint and "我会参考已学到的思考方法" not in web_answer:
                web_answer = f"{web_answer}\n{reasoning_hint}"
            pred = Prediction(intent="recommend", confidence=0.8)
            self._remember_response(normalized_text, web_answer, pred)
            return web_answer, pred

        if intent == "greeting":
            self.last_intent = intent
            self._learn_from_interaction(normalized_text, prediction)
            answer = "你好！我能做问候、计算、天气说明和学习建议。"
            self._remember_response(normalized_text, answer, prediction)
            return answer, prediction

        if intent == "math":
            math_result = self._safe_eval_expression(normalized_text)
            self.last_intent = intent
            if math_result is None:
                return "我识别到你在问计算，但表达式太复杂或不合法。请试试：18*7 或 (12+8)/2。", prediction
            self._learn_from_interaction(normalized_text, prediction)
            answer = f"计算结果是：{math_result}"
            self._remember_response(normalized_text, answer, prediction)
            return answer, prediction

        if intent == "weather":
            self.last_intent = intent
            self._learn_from_interaction(normalized_text, prediction)
            answer = "我目前是离线版，不能实时联网查天气；你可以告诉我城市，我给你出行建议模板。"
            self._remember_response(normalized_text, answer, prediction)
            return answer, prediction

        if intent == "recommend":
            self.last_intent = intent
            self._learn_from_interaction(normalized_text, prediction)
            if self._is_thinking_request(normalized_text) or len(normalized_text) >= 10:
                answer = self._build_thinking_response(normalized_text)
                self._remember_response(normalized_text, answer, prediction)
                return answer, prediction
            base = "如果你从 0 开始：先学 Python 基础，再做 2 个小项目，然后学机器学习与深度学习。"
            reasoning_hint = self._reasoning_hint_for_query(normalized_text)
            if reasoning_hint:
                return f"{base}\n{reasoning_hint}", prediction
            self._remember_response(normalized_text, base, prediction)
            return base, prediction

        if intent == "goodbye":
            self.last_intent = intent
            answer = "好的，我们下次继续。"
            self._remember_response(normalized_text, answer, prediction)
            return answer, prediction

        if self._identity_like_text(normalized_text):
            answer = "我理解你在表达自己。你的感受是重要的。你可以告诉我你现在最想解决的一件事，我会按步骤陪你推进。"
            self._remember_response(normalized_text, answer, prediction)
            return answer, prediction

        reasoning_hint = self._reasoning_hint_for_query(normalized_text)
        if reasoning_hint:
            return f"我先不敷衍给结论。这个问题我还不够确定，但我可以按这个思路帮你：{reasoning_hint} 你可以再补一个具体目标。", prediction

        if len(normalized_text.strip()) >= 4:
            answer = self._build_thinking_response(normalized_text)
            self._remember_response(normalized_text, answer, prediction)
            return answer, prediction

        answer = "这个输入信息太少或不明确。你可以试试：'帮我算 25*4'、'推荐学习路线'、'今天北京天气'。"
        self._remember_response(normalized_text, answer, prediction)
        return answer, prediction
