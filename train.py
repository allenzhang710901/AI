"""Train a Naive Bayes intent model from JSON data and save it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from ai_from_scratch.data import TRAINING_DATA
from ai_from_scratch.model import NaiveBayesIntentClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练从零 AI 助手的意图模型")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="训练数据 JSON 路径（格式: {intent: [句子1, 句子2]}）。不传则使用内置数据。",
    )
    parser.add_argument("--out", type=str, default="model.json", help="输出模型文件路径")
    return parser


def load_data(data_path: str) -> Dict[str, List[str]]:
    if not data_path:
        return TRAINING_DATA

    path = Path(data_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("训练数据必须是 JSON 对象，形如 {intent: [texts]}。")

    for intent, texts in data.items():
        if not isinstance(intent, str) or not isinstance(texts, list) or not texts:
            raise ValueError(f"意图 {intent!r} 的值必须是非空字符串列表。")
        if any(not isinstance(item, str) for item in texts):
            raise ValueError(f"意图 {intent!r} 中存在非字符串样本。")

    return data


def main() -> None:
    args = build_parser().parse_args()
    train_data = load_data(args.data)

    model = NaiveBayesIntentClassifier()
    model.fit(train_data)
    model.save(args.out)

    sample_count = sum(len(v) for v in train_data.values())
    print(f"训练完成：{len(train_data)} 个意图，{sample_count} 条样本")
    print(f"模型已保存到: {args.out}")


if __name__ == "__main__":
    main()
