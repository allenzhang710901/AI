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
    ) -> None:
        self.learned_data_path = Path(learned_data_path)
        self.auto_learn = auto_learn
        self.web_learning_enabled = web_learning_enabled
        self.web_knowledge = WebKnowledgeBase()
        self.learned_data = self._load_learned_data()
        self.classifier = classifier or NaiveBayesIntentClassifier()
        if not self.classifier.is_fitted:
            self._refit_with_learned_data()
        self.last_intent: str | None = None
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

    def persist_learned_data(self) -> bool:
        try:
            self.learned_data_path.write_text(
                json.dumps(self.learned_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            return False
        return True

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
        found = self.web_knowledge.find_relevant(text)
        if not found:
            return None
        topic, summary = found
        return f"我之前上网学过“{topic}”：{summary}"

    @staticmethod
    def _guess_web_topic(text: str) -> str:
        topic = re.sub(r"[，。！？,.!?]", " ", text).strip()
        return re.sub(r"\s+", " ", topic)

    def _try_web_lookup_for_query(self, text: str, prediction: Prediction) -> str | None:
        if not self.web_learning_enabled:
            return None
        if prediction.confidence >= WEB_LOOKUP_CONFIDENCE_THRESHOLD:
            return None
        topic = self._guess_web_topic(text)
        if len(topic) < 2 or self._is_low_signal_input(topic):
            return None

        # 先看缓存
        found = self.web_knowledge.find_relevant(topic)
        if found:
            _, summary = found
            return f"我在线知识库里有相关内容：{summary}"

        ok, _ = self.web_knowledge.learn_topic(topic)
        if not ok:
            return None
        learned = self.web_knowledge.find_relevant(topic)
        if not learned:
            return None
        _, summary = learned
        return f"我刚联网查到：{summary}"

    def reply(self, user_text: str) -> Tuple[str, Prediction]:
        normalized_text = self._normalize_text(user_text)

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

        remembered = self._try_answer_with_web_knowledge(normalized_text)
        if remembered:
            return remembered, Prediction(intent="recommend", confidence=0.75)
        bayes_pred = self.classifier.predict(normalized_text)
        prediction = self._route_intent(normalized_text, bayes_pred)
        intent = prediction.intent

        web_answer = self._try_web_lookup_for_query(normalized_text, prediction)
        if web_answer:
            return web_answer, Prediction(intent="recommend", confidence=0.8)

        if intent == "greeting":
            self.last_intent = intent
            self._learn_from_interaction(normalized_text, prediction)
            return "你好！我能做问候、计算、天气说明和学习建议。", prediction

        if intent == "math":
            math_result = self._safe_eval_expression(normalized_text)
            self.last_intent = intent
            if math_result is None:
                return "我识别到你在问计算，但表达式太复杂或不合法。请试试：18*7 或 (12+8)/2。", prediction
            self._learn_from_interaction(normalized_text, prediction)
            return f"计算结果是：{math_result}", prediction

        if intent == "weather":
            self.last_intent = intent
            self._learn_from_interaction(normalized_text, prediction)
            return "我目前是离线版，不能实时联网查天气；你可以告诉我城市，我给你出行建议模板。", prediction

        if intent == "recommend":
            self.last_intent = intent
            self._learn_from_interaction(normalized_text, prediction)
            return "如果你从 0 开始：先学 Python 基础，再做 2 个小项目，然后学机器学习与深度学习。", prediction

        if intent == "goodbye":
            self.last_intent = intent
            return "好的，我们下次继续。", prediction

        if self._identity_like_text(normalized_text):
            return "我理解你在表达自己。你也可以告诉我你现在最想解决的问题，我会尽量帮你。", prediction

        return "这个输入信息太少或不明确。你可以试试：'帮我算 25*4'、'推荐学习路线'、'今天北京天气'。", prediction
