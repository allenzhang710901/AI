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

TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+|[+\-*/()]")
KEYWORD_HINTS = {
    "greeting": ["你好", "哈喽", "在吗", "嗨", "hello", "hi"],
    "math": ["算", "计算", "多少", "加", "减", "乘", "除", "+", "-", "*", "/"],
    "weather": ["天气", "温度", "下雨", "晴", "阴", "热", "冷", "风"],
    "recommend": ["推荐", "建议", "学什么", "看什么", "怎么选"],
    "goodbye": ["再见", "拜拜", "退出", "结束", "下次聊"],
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
    """Assistant with hybrid intent routing + Naive Bayes fallback."""

    def __init__(self, classifier: NaiveBayesIntentClassifier | None = None) -> None:
        self.classifier = classifier or NaiveBayesIntentClassifier()
        if not self.classifier.is_fitted:
            self.classifier.fit(TRAINING_DATA)
        self.last_intent: str | None = None

    @classmethod
    def from_model_file(cls, model_path: str | Path) -> "SimpleChineseAIAssistant":
        classifier = NaiveBayesIntentClassifier.load(model_path)
        return cls(classifier=classifier)

    @staticmethod
    def _safe_eval_expression(text: str) -> str | None:
        expr_match = re.findall(r"[\d\s+\-*/().]+", text)
        if not expr_match:
            return None
        expr = "".join(expr_match).strip()
        if not expr or not re.fullmatch(r"[\d\s+\-*/().]+", expr):
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

    def _route_intent(self, user_text: str, bayes_pred: Prediction) -> Prediction:
        keyword_intent = self._keyword_intent(user_text)
        if keyword_intent:
            boost = 0.85 if keyword_intent != bayes_pred.intent else max(0.9, bayes_pred.confidence)
            return Prediction(keyword_intent, boost)

        nearest_intent, nearest_score = self._closest_training_intent(user_text)
        if nearest_score >= 0.62:
            return Prediction(nearest_intent, nearest_score)

        if bayes_pred.confidence < 0.24 and self.last_intent:
            return Prediction(self.last_intent, 0.35)

        return bayes_pred

    def reply(self, user_text: str) -> Tuple[str, Prediction]:
        bayes_pred = self.classifier.predict(user_text)
        prediction = self._route_intent(user_text, bayes_pred)
        intent = prediction.intent

        if intent == "greeting":
            self.last_intent = intent
            return "你好！我现在比之前更聪明一点了：能做问候、计算、天气说明和学习建议。", prediction

        if intent == "math":
            math_result = self._safe_eval_expression(user_text)
            self.last_intent = intent
            if math_result is None:
                return "我识别到你在问计算题，但没找到可计算表达式（例：18*7、(12+8)/2）。", prediction
            return f"计算结果是：{math_result}", prediction

        if intent == "weather":
            self.last_intent = intent
            return "我目前是离线版，不能实时联网查天气；但我可以先告诉你该带伞/穿衣的判断逻辑。", prediction

        if intent == "recommend":
            self.last_intent = intent
            return "如果你是从 0 开始：先学 Python 基础，再做 2 个小项目，然后学机器学习与深度学习。", prediction

        if intent == "goodbye":
            self.last_intent = intent
            return "好的，我们下次继续。", prediction

        if self.last_intent == "recommend":
            return "你可以告诉我目标（找工作/转行/比赛），我会给你更具体的学习路线。", prediction

        return "这个问题我还答不好。你可以换一种说法，或直接说：帮我算 25*4。", prediction
