"""CLI entry for the from-scratch AI assistant."""

from ai_from_scratch import SimpleChineseAIAssistant


EXIT_WORDS = {"退出", "再见", "bye", "quit", "exit"}


def main() -> None:
    assistant = SimpleChineseAIAssistant()
    print("从 0 开始 AI 助手已启动，输入内容与我对话（输入 退出 结束）。")

    while True:
        user_text = input("你: ").strip()
        if not user_text:
            continue

        if user_text.lower() in EXIT_WORDS:
            print("AI: 好的，期待下次见面！")
            break

        answer, pred = assistant.reply(user_text)
        print(f"AI: {answer}")
        print(f"   (意图: {pred.intent}, 置信度: {pred.confidence:.2f})")


if __name__ == "__main__":
    main()
