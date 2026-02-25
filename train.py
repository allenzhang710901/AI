"""Train a Naive Bayes intent model from JSON data and save it.

Beginner-friendly usage:
1) python train.py --wizard
2) python main.py --model my_model.json
"""

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
    parser.add_argument("--wizard", action="store_true", help="新手模式：按提示一步步生成数据并训练")
    parser.add_argument(
        "--init-data",
        type=str,
        default="",
        help="生成一个可编辑的数据模板 JSON（例如: --init-data my_data.json）",
    )
    return parser


def default_template_data() -> Dict[str, List[str]]:
    return {
        "greeting": ["你好", "在吗", "哈喽"],
        "math": ["帮我算一下", "18*7等于多少", "计算这个表达式"],
        "weather": ["今天会下雨吗", "北京天气怎么样", "温度多少"],
        "recommend": ["推荐一本书", "我该学什么", "给我建议"],
        "goodbye": ["再见", "拜拜", "下次聊"],
    }


def write_template(path: str) -> None:
    payload = default_template_data()
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已创建训练数据模板: {path}")
    print("你可以先编辑这个文件，再执行：")
    print(f"python train.py --data {path} --out my_model.json")


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


def wizard_collect_data() -> Dict[str, List[str]]:
    print("\n=== 新手训练向导 ===")
    print("每个意图至少输入 1 条样本；多条可用 | 分隔。")
    print("直接回车可使用推荐示例。\n")

    prompts = [
        ("greeting", "问候类（例如：你好|在吗|哈喽）"),
        ("math", "计算类（例如：帮我算一下|12*8 等于多少）"),
        ("weather", "天气类（例如：今天会下雨吗|北京天气）"),
        ("recommend", "推荐类（例如：推荐一本书|我该学什么）"),
        ("goodbye", "结束类（例如：再见|拜拜）"),
    ]

    template = default_template_data()
    data: Dict[str, List[str]] = {}

    for intent, tip in prompts:
        raw = input(f"{tip}\n请输入 {intent} 样本: ").strip()
        if not raw:
            data[intent] = template[intent]
            print(f"已使用默认样本 {len(data[intent])} 条。\n")
            continue

        samples = [item.strip() for item in raw.split("|") if item.strip()]
        if not samples:
            samples = template[intent]
        data[intent] = samples
        print(f"已记录 {len(samples)} 条。\n")

    return data


def train_and_save(train_data: Dict[str, List[str]], output_path: str) -> None:
    model = NaiveBayesIntentClassifier()
    model.fit(train_data)
    model.save(output_path)

    sample_count = sum(len(v) for v in train_data.values())
    print(f"训练完成：{len(train_data)} 个意图，{sample_count} 条样本")
    print(f"模型已保存到: {output_path}")
    print("下一步运行：")
    print(f"python main.py --model {output_path}")


def main() -> None:
    args = build_parser().parse_args()

    if args.init_data:
        write_template(args.init_data)
        return

    if args.wizard:
        data = wizard_collect_data()
        train_and_save(data, args.out)
        return

    train_data = load_data(args.data)
    train_and_save(train_data, args.out)


if __name__ == "__main__":
    main()
