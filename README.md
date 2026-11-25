# PaperQA

回答论文问题的 RAG 系统，用 [QASPER](https://allenai.org/data/qasper) 评测。
现在是 baseline：向量检索 top-5，Writer 回答并标注引用了哪几段。后面会一项一项往上加功能，
每加一项都跑一次评测，结果见下面的[消融表](#消融表)。

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
```

```bash
uv run python -m paperqa.ingest
```

第一条命令安装依赖（Linux/Windows 上装的是 CPU 版 torch，见 `pyproject.toml`）。第二条命令会下载
QASPER（约 11MB），把 dev 集的 281 篇论文切块、向量化，存进 `data/chroma/`。在 4 核 CPU 上大约要 10 分钟，
只需跑一次。

只评测检索，不调 LLM，不花钱：

```bash
uv run python eval/run_eval.py --name dense_retrieval --retrieval-only
```

完整评测（检索 + Writer + LLM-as-judge）需要 Anthropic API key，在 [Claude Console](https://platform.claude.com/)
里创建。代码先读 `PAPERQA_ANTHROPIC_API_KEY`，没有再读 `ANTHROPIC_API_KEY`：

```bash
export PAPERQA_ANTHROPIC_API_KEY=sk-ant-...
```

先用便宜的 Haiku 5.5 跑 10 题，确认流程能通：

```bash
uv run python eval/run_eval.py --name smoke_haiku --model claude-haiku-5-5 --judge-model claude-haiku-5-5 --n 10
```

没有 API 额度、但有 Claude 订阅时，加 `--backend claude-code`：LLM 调用改走 Claude Code 的无头模式
（`claude -p`），用的是 Claude Code 的登录，计入订阅额度而不是 API 额度。需要本机装了 Claude Code 并已登录：

```bash
uv run python eval/run_eval.py --name smoke_haiku_cc --backend claude-code --model claude-haiku-5-5 --judge-model claude-haiku-5-5 --n 10
```

两个后端的结果不能混在一起比，原因见下面[评测](#评测)一节。

再跑 100 题（默认模型是 `claude-opus-5-5`，用 `--model` 和 `--judge-model` 换）：

```bash
uv run python eval/run_eval.py --name baseline_dense
```

把所有结果打印成一张表：

```bash
uv run python eval/compare.py
```

## 流程

```
问题 ──> 向量检索（只在这篇论文里检索，取 top-5）──> Writer（LLM）──> {answer, citations}
                                                                     │
                     评测：Recall@k、Answer F1、Evidence F1、LLM-as-judge、延迟、token
```

- **数据**：直接用 QASPER 官方 JSON（v0.3）。论文已经按章节和段落解析好，不需要处理 PDF。
  HuggingFace 上的 `allenai/qasper` 依赖数据集脚本，新版 `datasets` 已经不支持，所以没用它。
- **切块**：一个段落一个 chunk，开头拼上章节名，比如 `[Method > Training] We train ...`。摘要和每个
  图表标题也各算一个 chunk，因为 QASPER 有一部分证据就是表格（dev 里有 253 条）。元数据里有
  `paper_id`、`section_name`、`paragraph_idx`。
- **向量**：`BAAI/bge-small-en-v1.5`（sentence-transformers），存在 Chroma 里，用余弦相似度。
- **检索**：QASPER 的每个问题只针对一篇论文，所以检索时按 `paper_id` 过滤。有三种检索器，用
  `--retriever` 选：
  - `dense`（默认）：向量检索。
  - `bm25`：关键词检索。每篇论文单独建一个 BM25 索引，自己实现，没有额外依赖。
  - `hybrid`：两路各取前 20 名，用 RRF 融合（分数 = Σ 1/(60 + 名次)）。
- **Writer**：`claude-opus-5-5`，effort 设为 `medium`，用 structured outputs 返回
  `{"answer": ..., "citations": [...]}`。按 QASPER 的风格要求答案尽量短；资料里找不到答案就回答
  `Unanswerable`。开启了 server-side fallback：模型因安全策略拒答时，API 会自动换一个模型重试。

## 评测

默认从 dev 集的可回答问题里按 `seed=42` 抽 100 个。所有标注者都标为 unanswerable 的问题会跳过
（dev 里有 60 个）。参数相同的两次运行用的是完全相同的题目。

| 指标 | 含义 | 花钱吗 |
|---|---|---|
| Recall@k | 标注的证据段落有多少比例出现在 top-k 里。按标注者分别算，取最大值 | 不花 |
| Hit@k | 至少一条证据段落出现在 top-k 里 | 不花 |
| Answer F1 | QASPER 官方的 token F1，多个标注取最大值 | 不花（但要先有 Writer 的答案） |
| Evidence F1 | Writer 引用的段落和标注证据的 F1（官方定义） | 不花（同上） |
| Judge acc | LLM-as-judge 判断答案和任意一个参考答案意思是否一致 | 花 |
| 延迟 / token | 检索 + Writer 的平均耗时；Writer 的输入和输出 token（输出包含 thinking） | – |

每次运行会存到 `eval/results/<name>.json`，里面有配置、汇总指标，以及每个问题检索到的 chunk、答案、
得分和 judge 的理由，方便逐题排查。

常用参数：`--retriever dense|bm25|hybrid`、`--n 20`（先拿少量题试跑）、`--n 0`（全量）、`--no-judge`（只算官方 F1，省掉 judge 的调用）、`--effort low`、
`--model`、`--judge-model`、`--workers`（并发请求数，默认 4）、`--backend api|claude-code`。

**两个 LLM 后端**（都在 `src/paperqa/llm.py`，结果文件的 `config.backend` 会记录用的是哪个）：

| 后端 | 怎么调 | 计费 | 注意 |
|---|---|---|---|
| `api`（默认） | Anthropic SDK | API key | – |
| `claude-code` | 每次调用起一个 `claude -p` 进程 | Claude Code 登录（订阅额度） | Claude Code 每次请求会自带约 1.5–2k token 的上下文；带 schema 的调用要两轮模型请求；没有 server-side fallback |

同样的 10 题、同样的模型（Haiku 5.5）和 prompt，两个后端的结果（`smoke_haiku` 和 `smoke_haiku_cc`）：

| 后端 | Answer F1 | Judge acc | 答案平均词数 | Writer token 输入 / 输出 | Writer + judge 总 token 输入 / 输出 |
|---|---|---|---|---|---|
| `api` | 0.478 | 0.500 | 12.3 | 1191 / 132 | 16,746 / 2,335 |
| `claude-code` | 0.318 | 0.600 | 15.4 | 2906 / 266 | 51,226 / 5,561 |

`claude-code` 后端在几道短答案题上多写了解释（比如 `api` 答 "Energy"，它答 "Energy (highest GARCH(1,1) R² of
0.44; ...)"），judge 仍判对，但按词重合算的 Answer F1 掉了。只有 10 题，具体数字不可靠，但足以说明：
**一张消融表里的所有行必须用同一个后端**，不同后端之间不能比 Answer F1、延迟和 token。

完整跑一次 100 题是 200 次 LLM 调用（Writer 和 judge 各 100 次）。Writer 的 prompt 平均约 850 token。
按 Opus 5.5 每百万 token 输入 $4、输出 $20 估算，一次大约 1–2 美元；thinking 用多少 token 不固定，
所以实际花费会有浮动。

### 评测本身的检查

指标算错了，后面的数字就都没有意义，所以下面几项都核对过：

- dev 集里所有文本证据都能和某个 chunk 精确匹配，只有 129 条只写了章节标题（如 "Datasets"）的证据
  匹配不到；253 条表格/图片证据全部能匹配到图表标题。
- 用另一份独立代码，直接从原始 JSON 重算 Recall@5，100 题逐题结果完全一致。
- Chroma 按 `paper_id` 过滤后的 top-10 和暴力精确搜索的结果一致（抽查 200 题，只有分数完全相同的
  chunk 互换了位置）。
- 官方 F1、Recall@k、证据匹配都有单元测试（`tests/`）。

**样本噪声**：100 题的样本噪声很大，见下面检索对比里的例子。检索评测不花钱，所以**比较检索方法一律用
全量 dev**（`--n 0`，945 题）。100 题样本只用来算要调 LLM 的指标。

判断两个配置的差距是不是真的，用配对 bootstrap：

```bash
uv run python eval/paired_diff.py eval/results/A.json eval/results/B.json
```

## 消融表

### 检索（全量 dev，945 道可回答题，不调 LLM）

| 检索器 | Recall@1 | Recall@5 | Recall@10 | Hit@5 | 延迟 (ms) |
|---|---|---|---|---|---|
| dense | 0.205 | 0.553 | 0.722 | 0.657 | 26 |
| bm25 | 0.158 | 0.481 | 0.639 | 0.577 | 3 |
| hybrid（RRF） | 0.223 | 0.569 | 0.734 | 0.673 | 31 |

复现（每条约 30 秒）：`uv run python eval/run_eval.py --name dense_all --retriever dense --retrieval-only --n 0`，
把 `dense` 换成 `bm25` 或 `hybrid` 即可。全量结果文件每个约 2MB，没有提交到 git。

结论：

- **hybrid 比 dense 的 Recall@5 高 1.6 个点，但这个差距不显著**（95% 置信区间 −0.9 到 +4.2 个点）。
  RRF 的参数用的是常见默认值（各取 20 名，k=60），没有在 dev 上调过。
- **bm25 单独用明显比 dense 差**（Recall@5 低 7.1 个点，区间 −10.4 到 −3.9）。
- **两路是互补的**：dense 和 bm25 至少有一路在 top-5 里命中证据的题目占 77.7%，而 hybrid 的 Hit@5 只有
  67.3%。也就是说候选里有证据，只是等权重的 RRF 没把它排进前 5。Recall@10（0.73）远高于 Recall@5（0.57）
  也说明同一件事。所以下一步更值得做的是对候选做重排序，而不是继续调融合。
- **100 题样本会误导**：在 seed=42 的 100 题上，hybrid 的 Recall@5 是 0.634，dense 是 0.521，看起来提升
  11 个点，而且在这 100 题内部做 bootstrap 也显示"显著"。全量数据说明这只是这批题碰巧对 hybrid 有利。

### 端到端（dev，n=100，seed=42）

| 配置 | Recall@5 | Hit@5 | Answer F1 | Evidence F1 | Judge acc | 延迟 (s) | Writer token 输入 / 输出 |
|---|---|---|---|---|---|---|---|
| baseline_dense_cc_sonnet（dense top-5 + Writer） | 0.521 | 0.592 | 0.387 | 0.418 | 0.610 | 1.71 | 3048 / 143 |

这张表统一用 Sonnet 5.5 做 Writer 和 judge，走 `claude-code` 后端（订阅额度）：

```bash
uv run python eval/run_eval.py --name <配置名> --backend claude-code --model claude-sonnet-5-5 --judge-model claude-sonnet-5-5
```

`baseline_dense_cc_sonnet` 是正式 baseline，0 报错，Writer 和 judge 合计 524,107 输入 / 27,783 输出 token。

这张表里的 Recall@5 只是这 100 题上的值，比较检索方法请看上面的全量表。

## 目录结构

```
src/paperqa/   config / qasper（数据和切块）/ embedding / ingest / retrieve / agents / llm / metrics
eval/          run_eval.py、compare.py、results/
tests/         单元测试（uv run pytest）
data/          QASPER 原始数据和 Chroma 索引（不提交到 git）
```

开发规则见 [CLAUDE.md](CLAUDE.md)。

## Roadmap

1. [x] Baseline：向量检索 + Writer
2. [x] 混合检索（BM25 + 向量，RRF 融合）：已实现，全量 dev 上相比 dense 没有显著提升
3. [ ] Grader 和重查循环
4. [ ] Router 和子问题拆解（加上自己造的跨论文问题）
5. [ ] FastAPI + SSE 和 Streamlit 界面
6. [ ] PDF 上传解析（PyMuPDF），Docker 打包
