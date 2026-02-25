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
# Windows 也可以直接 run.bat（默认开启联网学习）
```

进入对话后，直接下指令：

```text
学习 人工智能
学习 量子计算
```

助手会尝试从公开网络源抓取摘要（优先百科，失败时走备用搜索摘要），并保存到本地 `.ai_web_knowledge.json`。
下次你问相关主题时，它会优先使用已学习到的本地知识回答。

> 注意：联网学习依赖网络可用性与公开站点返回结果。
> 已开启时，当助手对当前问题置信度 < 80% 会自动尝试联网查询。


### 3.6 稳定性与权限（针对“秒崩溃”）

如果你遇到“秒崩溃”，现在版本会自动兜底：

- `--model` 指向不存在/损坏文件时，不会崩溃，会自动回退到内置模型。
- 读写学习文件失败时，会给出提示，不会直接退出。

权限方面通常只需要：

1. **网络权限**：用于 `--web-learn` 联网抓取摘要（HTTPS）。
2. **目录写权限**：用于保存 `.ai_learned_data.json` / `.ai_web_knowledge.json` / `.ai_response_memory.json`。
3. **无需管理员权限**（一般情况）。


### 3.7 启动即长时间联网“扫库”

如果你希望每次启动都先花很长时间联网学习，再开始对话：

```bash
python main.py --web-learn --startup-sync-seconds 1800
```

- `--startup-sync-seconds`：启动时联网学习时长（秒），可设很大。
- `--seed-topics`：逗号分隔的种子主题（可选）。不填时会自动用内置全局主题池探索，不需要你手动给主题。
- 启动日志会显示 `learned / refreshed_existing / tried / expanded_from_cache`，你能看到是否真的在持续搜索，而不是秒结束。
- 深度同步的最大尝试次数会随 `--startup-sync-seconds` 自动放大（默认下限 200），长时运行不会很快触顶停止。
- Windows 可直接双击 `run.bat`（默认已带 300 秒启动深度学习）。

> 注意：严格意义上“全网、无限平台、无限时间”在工程上不可行。当前实现是“多源+扩展+可长时间运行”的可控方案。


### 3.8 一键 AI 全域思考学习（30000 秒）

如果你要它重点学习“AI 的说话方式、思考逻辑、推理流程”等大范围主题：

```bash
python main.py --ai-all-sync
```

- 该模式会自动开启联网学习（等价于开启 `--web-learn`）。
- 默认深度学习时长是 `30000` 秒（可再叠加 `--startup-sync-seconds` 覆盖）。
- 会自动注入 AI 相关种子主题（思维链、RAG、Agent、提示工程、对齐等），无需手动写主题。

### 3.4 常见问题（你这种情况）

- Windows 下用 `run.bat` 启动默认会带 `--web-learn`，会保持联网学习模式。
- 联网学习触发规则：手动输入“学习 关键词”，或当前回答置信度 < 80% 时自动联网尝试。
- 知识问法优化：像“我的世界是啥 / minecraft是啥 / 王者荣耀是啥”会优先抽取主题联网查询。
- 回答策略优化：如果你问题里有多个关键词，助手会看“匹配覆盖率”；只命中其中一个词时，会明确说“只懂基础，不是全懂”。
- 每次回答前会先做内部“思考步骤”（主题抽取、意图判断、关系判断）再决定是否搜索。
- 新增“思路复用”：如果联网学到的摘要里包含“步骤/方法/思路”等内容，后续聊天会把这些思路作为回答参考，而不只是复述词条。
- 新增“复刻思路”用法：你可以先说“查查AI的逻辑思考逻辑”或“学习 AI 思考框架”，再说“复刻 + 你的问题”，助手会按学到的步骤化逻辑回答。
- 新增“思考式回答”模式：当你输入“分析/思考/怎么办/方案”等问题时，助手会按“目标→约束→方案→复盘”输出，不再只给死板结论。
- 新增“情绪感知回复”：当你表达焦虑/难过/愤怒/开心时，助手会先共情，再给可执行下一步，而不是只回模板句。
- 新增“能力升级回答”模式：当你说“太傻/死板/想更聪明/想接近ChatGPT”时，助手会给出结构化升级路线（检索/记忆/推理/评估）。
- 关系问句优化：如 “is honor of king same as 王者荣耀” 会优先走实体关系判断。
- 若你只说“你去查啊”这类无主题命令，助手会要求你补全主题，而不是瞎查。
- 对“我是谁”这类自我身份问题，不再误查电影词条，会给出更合适的人类化回应。
- 一般不需要额外系统权限；只需本机可联网、Python 可发起 HTTPS 请求。
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
- 教它固定回答：`学习回答 问题 => 你的答案`（会持久化）
- 稳定性自检：`python main.py --doctor`
- 启动深度联网学习：`python main.py --web-learn --startup-sync-seconds 1800`
- 一键AI全域学习30000秒：`python main.py --ai-all-sync`

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
├── run_ai_all.sh
├── run_ai_all.bat
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

