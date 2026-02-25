"""CLI entry for the from-scratch AI assistant.

Usage:
- Interactive mode: python main.py
- One-shot mode:    python main.py --ask "18*7等于多少"
- Demo mode:        python main.py --demo
- Load model:       python main.py --model model.json --ask "你好"
"""

from __future__ import annotations

import argparse

from ai_from_scratch import SimpleChineseAIAssistant


EXIT_WORDS = {"退出", "结束", "再见", "bye", "quit", "exit", "q"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 0 开始 AI 助手")
    parser.add_argument("--ask", type=str, help="单次提问后直接返回结果")
    parser.add_argument("--demo", action="store_true", help="运行内置示例")
    parser.add_argument("--model", type=str, default="", help="可选：加载 train.py 训练出的模型文件")
    parser.add_argument("--no-auto-learn", action="store_true", help="关闭对话过程中的自动学习")
    parser.add_argument("--web-learn", action="store_true", help="开启联网学习（手动学习+低置信度<80%自动联网查询）")
    return parser


def run_interactive(assistant: SimpleChineseAIAssistant) -> None:
    print("从 0 开始 AI 助手已启动，输入内容与我对话（输入 退出/结束/q 结束）。")

    while True:
        user_text = input("你: ").strip()
        if not user_text:
            continue

        if user_text.lower() in EXIT_WORDS:
            print("AI: 好的，期待下次见面！")
            assistant.persist_learned_data()
            break

        answer, pred = assistant.reply(user_text)
        print(f"AI: {answer}")
        print(f"   (意图: {pred.intent}, 置信度: {pred.confidence:.2f})")


def run_once(assistant: SimpleChineseAIAssistant, user_text: str) -> None:
    answer, pred = assistant.reply(user_text)
    print(f"你: {user_text}")
    print(f"AI: {answer}")
    print(f"(意图: {pred.intent}, 置信度: {pred.confidence:.2f})")
    assistant.persist_learned_data()


def run_demo(assistant: SimpleChineseAIAssistant) -> None:
    examples = ["你好", "18*7等于多少", "推荐一本书", "今天北京天气", "拜拜"]
    print("正在运行 demo...\n")
    for text in examples:
        run_once(assistant, text)
        print("-" * 40)


def build_assistant(model_path: str, auto_learn: bool, web_learn: bool) -> SimpleChineseAIAssistant:
    if model_path:
        # Load trained model weights but keep auto-learning memory file behavior.
        assistant = SimpleChineseAIAssistant.from_model_file(model_path)
        assistant.auto_learn = auto_learn
        assistant.web_learning_enabled = web_learn
        return assistant
    return SimpleChineseAIAssistant(auto_learn=auto_learn, web_learning_enabled=web_learn)


def main() -> None:
    args = build_parser().parse_args()
    assistant = build_assistant(args.model, auto_learn=not args.no_auto_learn, web_learn=args.web_learn)

    if args.demo:
        run_demo(assistant)
        return

    if args.ask:
        run_once(assistant, args.ask)
        return

    run_interactive(assistant)


if __name__ == "__main__":
    main()
