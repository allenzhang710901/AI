# 从 0 开始的 AI：一个可运行的中文命令行助手

这个项目提供了一个**不依赖云 API** 的最小 AI 示例：

- 使用纯 Python 实现一个朴素贝叶斯文本分类器。
- 用少量中文语料训练「意图识别」模型。
- 在命令行里进行对话，根据识别出的意图给出回复。

> 目标：帮助你理解“从 0 开始写一个 AI”的最小闭环：
> 数据 → 训练 → 预测 → 交互。

---

## 1. 环境要求

- Python 3.9+

无需额外第三方依赖。

---

## 2. 最快运行（更省事）

### 方式 A：一键跑 demo（推荐）

```bash
python main.py --demo
```

### 方式 B：单次提问（不用进入交互）

```bash
python main.py --ask "18*7等于多少"
```

### 方式 C：普通交互模式

```bash
python main.py
```

输入 `退出` 结束。

---

## 3. 额外便捷启动方式

Linux / macOS：

```bash
./run.sh --demo
```

Windows：

```bat
run.bat --demo
```

也支持模块方式：

```bash
python -m ai_from_scratch --demo
```

---

## 4. 项目结构

```text
.
├── ai_from_scratch
│   ├── __init__.py
│   ├── __main__.py      # 支持 python -m ai_from_scratch
│   ├── data.py          # 训练样本
│   └── model.py         # 朴素贝叶斯 + 聊天机器人逻辑
├── main.py              # CLI 入口
├── run.sh               # Linux/macOS 一键启动
├── run.bat              # Windows 一键启动
└── README.md
```

---

## 5. 你可以继续扩展的方向

1. 增加训练数据（每个意图更多样本）。
2. 加入分词与停用词处理。
3. 将规则回复改成函数调用（查天气 API、写待办等）。
4. 增加 Web 界面（Flask/FastAPI + 前端）。
5. 引入向量检索（RAG）支持你的私有知识库。
