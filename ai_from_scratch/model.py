"""A minimal Naive Bayes based Chinese intent assistant."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .data import TRAINING_DATA

TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+|[+\-*/()]")


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
        total_texts = sum(len(list(texts)) for texts in samples.values())
        if total_texts == 0:
            raise ValueError("No samples provided.")

        for intent, texts in samples.items():
            texts = list(texts)
            self.intent_priors[intent] = math.log(len(texts) / total_texts)
            counter = Counter()
            for text in texts:
                tokens = self.tokenize(text)
                counter.update(tokens)
                self.vocab.update(tokens)
            self.token_counts[intent] = counter
            self.total_tokens[intent] = sum(counter.values())

        self.is_fitted = True

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
    """A tiny assistant with intent recognition and rule-based responses."""

    def __init__(self) -> None:
        self.classifier = NaiveBayesIntentClassifier()
        self.classifier.fit(TRAINING_DATA)

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

    def reply(self, user_text: str) -> Tuple[str, Prediction]:
        prediction = self.classifier.predict(user_text)
        intent = prediction.intent

        if intent == "greeting":
            return "你好！很高兴见到你，我是你的从零开始 AI 助手。", prediction

        if intent == "math":
            math_result = self._safe_eval_expression(user_text)
            if math_result is None:
                return "我识别到你在问计算题，但没找到可计算的表达式（例：18*7）。", prediction
            return f"计算结果是：{math_result}", prediction

        if intent == "weather":
            return "我当前是离线示例，不能实时查天气。你可以接入天气 API 来升级我。", prediction

        if intent == "recommend":
            return "推荐你先学：Python 基础 → 数据处理 → 机器学习 → 深度学习。", prediction

        if intent == "goodbye":
            return "好的，我们下次再聊。", prediction

        return "我还在学习中，可以换一种说法再试试。", prediction
