# 从 0 开始的 AI：本地中文命令行助手（可训练）

这个项目是一个**纯本地、零依赖云 API** 的 AI 入门示例。

- 用朴素贝叶斯做意图分类。
- 支持关键词路由 + 相似度兜底。
- 支持你自己准备数据进行训练并保存模型。

---

## 1. 环境要求

- Python 3.9+

---

## 2. 快速运行

```bash
python main.py --demo
python main.py --ask "18*7等于多少"
```

---

## 3. 我如何训练他（重点）

### 3.1 使用内置数据训练

```bash
python train.py --out model.json
```

训练完成后使用：

```bash
python main.py --model model.json --ask "你好"
```

### 3.2 用你自己的数据训练

先新建一个 JSON 文件，比如 `my_data.json`：

```json
{
  "greeting": ["你好", "在吗", "哈喽"],
  "math": ["帮我算一下", "18*7等于多少", "计算这个表达式"],
  "recommend": ["推荐一本书", "我该学什么", "给我建议"]
}
```

然后训练：

```bash
python train.py --data my_data.json --out my_model.json
```

最后加载你的模型：

```bash
python main.py --model my_model.json
```

---

## 4. 命令速查

- 交互模式：`python main.py`
- 单次提问：`python main.py --ask "你的问题"`
- 演示模式：`python main.py --demo`
- 模型训练：`python train.py --data my_data.json --out my_model.json`
- 加载模型：`python main.py --model my_model.json --ask "你好"`

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
