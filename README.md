# 从 0 开始的 AI：本地中文命令行助手（可训练）

如果你是新手，**只看下面 3 步就能训练成功**。

## 新手 3 步（最简单）

1. 打开训练向导：

```bash
python train.py --wizard --out my_model.json
```

2. 按提示输入几条样本（不会就直接回车用默认示例）。

3. 用你训练出的模型聊天：

```bash
python main.py --model my_model.json
```

---

## 1. 环境要求

- Python 3.9+

---

## 2. 快速运行（不训练也能用）

```bash
python main.py --demo
python main.py --ask "18*7等于多少"
```

---

## 3. 我如何训练他（详细版）

### 3.1 方法 A：向导模式（推荐新手）

```bash
python train.py --wizard --out my_model.json
```

完成后：

```bash
python main.py --model my_model.json
```

### 3.2 方法 B：先生成模板，再编辑

先生成模板：

```bash
python train.py --init-data my_data.json
```

编辑 `my_data.json`（示例格式）：

```json
{
  "greeting": ["你好", "在吗", "哈喽"],
  "math": ["帮我算一下", "18*7等于多少", "计算这个表达式"],
  "recommend": ["推荐一本书", "我该学什么", "给我建议"]
}
```

训练并保存：

```bash
python train.py --data my_data.json --out my_model.json
```

加载模型：

```bash
python main.py --model my_model.json
```

### 3.3 方法 C：使用内置数据直接训练

```bash
python train.py --out model.json
python main.py --model model.json
```

---


### 3.5 联网学习（让它自己慢慢增长知识）

你可以开启联网学习模式：

```bash
python main.py --web-learn
```

进入对话后，直接下指令：

```text
学习 人工智能
学习 量子计算
```

助手会尝试从公开百科抓取摘要，并保存到本地 `.ai_web_knowledge.json`。
下次你问相关主题时，它会优先使用已学习到的本地知识回答。

> 注意：联网学习依赖网络可用性与公开站点返回结果。

### 3.4 常见问题（你这种情况）

- 输入 `1` 这种信息太少的内容，助手会提示你补充问题，而不是乱猜意图。
- 输入 `q` / `退出` / `结束` 可以直接退出对话。
- `在么` 这类常见口语写法会自动按问候处理。
- 输入特别长的计算式（超长数字）会被拒绝，避免异常输出。
- 像“我是.../我不是...”这类自我表达，助手会先做中性回应，不再硬套学习推荐。


## 4. 命令速查

- 交互模式：`python main.py`
- 单次提问：`python main.py --ask "你的问题"`
- 演示模式：`python main.py --demo`
- 向导训练：`python train.py --wizard --out my_model.json`
- 生成模板：`python train.py --init-data my_data.json`
- 用 JSON 训练：`python train.py --data my_data.json --out my_model.json`
- 加载模型：`python main.py --model my_model.json --ask "你好"`
- 开启联网学习：`python main.py --web-learn`（然后输入：`学习 人工智能`）

---

## 5. 项目结构

```text
.
├── ai_from_scratch
│   ├── __init__.py
│   ├── __main__.py
│   ├── data.py
│   └── model.py
├── main.py
├── train.py
├── run.sh
├── run.bat
└── README.md
```

---


## 6. 能让它自动慢慢变智能吗？

可以。现在默认开启了**自动学习（本地样本）**，也支持**联网学习（外部知识）**：

- 每次你对话后，若模型对某句判断非常有把握（高置信度），会把该句加入本地学习库。
- 本地学习库保存到 `.ai_learned_data.json`，会让意图识别更贴近你的习惯。
- 开启 `--web-learn` 后，你可用“学习 关键词”让它上网抓取知识并保存到 `.ai_web_knowledge.json`。

你可以这样验证：

```bash
python main.py
# 多聊几句问候/天气/推荐
# 退出后会看到 .ai_learned_data.json
```

如需关闭自动学习：

```bash
python main.py --no-auto-learn
```

