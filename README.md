# PaperQA

回答论文问题的 RAG 系统，用 [QASPER](https://allenai.org/data/qasper) 评测。
现在有 baseline（向量检索 top-5，Writer 回答并标注引用了哪几段）和可选的 Grader 重查循环。功能一项一项
往上加，每加一项都跑一次评测，结果见下面的[消融表](#消融表)。

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

完整评测（检索 + Writer + LLM-as-judge）默认走 Claude Code 的无头模式（`claude -p`）：用的是 Claude Code 的
登录，计入 Claude 订阅额度，不需要 API key。需要本机装了 Claude Code 并已登录。

先用便宜的 Haiku 5.5 跑 10 题，确认流程能通（试跑结果放在 `eval/results/smoke/`，不进消融表）：

```bash
uv run python eval/run_eval.py --name smoke_haiku_cc --model claude-haiku-5-5 --judge-model claude-haiku-5-5 --n 10 --out-dir eval/results/smoke
```

再跑 100 题（默认 Writer 和 judge 都是 `claude-sonnet-5-5`，用 `--model` 和 `--judge-model` 换）：

```bash
uv run python eval/run_eval.py --name baseline_dense_cc_sonnet
```

想改用 Anthropic API key 计费，加 `--backend api`。key 在 [Claude Console](https://platform.claude.com/) 里创建，
代码先读 `PAPERQA_ANTHROPIC_API_KEY`，没有再读 `ANTHROPIC_API_KEY`：

```bash
export PAPERQA_ANTHROPIC_API_KEY=sk-ant-...
```

```bash
uv run python eval/run_eval.py --name baseline_dense_api --backend api
```

两个后端的结果不能混在一起比，原因见下面[评测](#评测)一节。

工作区有未提交的代码改动时，`run_eval.py` 会拒绝运行（结果文件要记录确切的 commit）；调试时加 `--allow-dirty`。

把所有结果打印成一张表：

```bash
uv run python eval/compare.py
```

## 流程

```
问题 ──> 向量检索（只在这篇论文里检索，取 top-5）──> [Grader] ──够──> Writer（LLM）──> {answer, citations}
                                          ▲            │                                │
                                          └─ 不够：换一个查询再检索，把新段落追加进去      │
                                    评测：Recall@k、Answer F1、Evidence F1、LLM-as-judge、延迟、token
```

Grader 是可选的（`--grader`），不加就是 baseline：检索完直接交给 Writer。

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
- **Writer**：默认 `claude-sonnet-5-5`，effort 设为 `medium`，用 structured outputs 返回
  `{"answer": ..., "citations": [...]}`。按 QASPER 的风格要求答案尽量短；资料里找不到答案就回答
  `Unanswerable`。`api` 后端开启了 server-side fallback（模型因安全策略拒答时，API 会自动换一个模型重试）；
  `claude-code` 后端没有。
- **Grader 和重查循环**（`--grader`）：Writer 之前先让 Grader 判断检索到的段落里有没有回答所需的具体信息。
  不够的话，Grader 写一个新的检索查询（用论文里可能出现的说法，而不是照抄问题），取这个查询的 top-5 里
  还没出现过的段落追加进去。最多重查 `--max-requeries` 轮（默认 1），所以 Writer 最多看到
  5 × (1 + 轮数) 个段落。Grader 调用失败就保持第一次的检索结果。Grader 的模型用 `--grader-model` 指定。

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
| Context recall | 实际送进 Writer 的段落覆盖了多少标注证据。没有 Grader 时等于 Recall@5 | 不花 |
| 延迟 / token | 检索 + Grader + Writer 的平均耗时；Writer 和 Grader 的输入和输出 token（输出包含 thinking） | – |

每次运行会存到 `eval/results/<name>.json`，里面有配置、汇总指标，以及每个问题检索到的 chunk、答案、
得分和 judge 的理由，方便逐题排查。

常用参数：`--retriever dense|bm25|hybrid`、`--n 20`（先拿少量题试跑）、`--n 0`（全量）、`--no-judge`（只算官方 F1，省掉 judge 的调用）、`--effort low`、
`--model`、`--judge-model`、`--workers`（并发请求数，默认 4）、`--backend api|claude-code`。

**两个 LLM 后端**（都在 `src/paperqa/llm.py`，结果文件的 `config.backend` 会记录用的是哪个）：

| 后端 | 怎么调 | 计费 | 注意 |
|---|---|---|---|
| `api` | Anthropic SDK | API key | – |
| `claude-code`（默认） | 每次调用起一个 `claude -p` 进程 | Claude Code 登录（订阅额度） | Claude Code 每次请求会自带约 1.5–2k token 的上下文；带 schema 的调用要两轮模型请求；没有 server-side fallback |

同样的 10 题、同样的模型（Haiku 5.5）和 prompt，两个后端的结果（`eval/results/smoke/` 下的 `smoke_haiku` 和 `smoke_haiku_cc`）：

| 后端 | Answer F1 | Judge acc | 答案平均词数 | Writer token 输入 / 输出 | Writer + judge 总 token 输入 / 输出 |
|---|---|---|---|---|---|
| `api` | 0.478 | 0.500 | 12.3 | 1191 / 132 | 16,746 / 2,335 |
| `claude-code` | 0.318 | 0.600 | 15.4 | 2906 / 266 | 51,226 / 5,561 |

`claude-code` 后端在几道短答案题上多写了解释（比如 `api` 答 "Energy"，它答 "Energy (highest GARCH(1,1) R² of
0.44; ...)"），judge 仍判对，但按词重合算的 Answer F1 掉了。只有 10 题，具体数字不可靠，但足以说明：
**一张消融表里的所有行必须用同一个后端**，不同后端之间不能比 Answer F1、延迟和 token。

完整跑一次 100 题是 200 次 LLM 调用（Writer 和 judge 各 100 次），加 Grader 再多 100 次。实测 baseline 一次
合计约 52 万输入 / 2.8 万输出 token（Sonnet 5.5，`claude-code` 后端），按 Sonnet 5.5 的 API 标价（每百万 token
输入 $2、输出 $10）折算约 1.3 美元，实际记在订阅额度上；Grader 版约 88 万 / 6 万 token。

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

| 配置 | Recall@5 | Context recall | Answer F1 | Evidence F1 | Judge acc | 延迟 (s) | LLM token 输入 / 输出 |
|---|---|---|---|---|---|---|---|
| baseline_dense_cc_sonnet（dense top-5 + Writer） | 0.521 | 0.521 | 0.387 | 0.418 | 0.610 | 1.71 | 3048 / 143 |
| grader_dense_cc_sonnet（+ Grader，最多重查 1 轮） | 0.521 | 0.712 | 0.416 | 0.523 | 0.690 | 3.25 | 6590 / 468 |
| baseline_dense_k10_cc_sonnet（dense top-10 + Writer，对照） | 0.521 | 0.780 | 0.404 | 0.570 | 0.690 | 1.61 | 3980 / 153 |

Recall@5 是第一次检索的值，两行相同；Context recall 是实际送进 Writer 的段落的证据召回。LLM token 是 Writer 加
Grader 每题的平均值，不含 judge。

这张表统一用 Sonnet 5.5 做 Writer 和 judge，走 `claude-code` 后端（订阅额度）：

```bash
uv run python eval/run_eval.py --name <配置名>
```

`baseline_dense_cc_sonnet` 是正式 baseline，0 报错，Writer 和 judge 合计 524,107 输入 / 27,783 输出 token。

#### Grader 和重查循环

`grader_dense_cc_sonnet`：Writer 和 judge 是 Sonnet 5.5，Grader 是 Haiku 5.5（`--grader --grader-model
claude-haiku-5-5`），0 报错。Writer、Grader、judge 合计 879,295 输入 / 60,274 输出 token。

和 baseline 的配对 bootstrap（`eval/paired_diff.py`，同样的 100 题）：

| 指标 | Grader − baseline | 95% 置信区间 | 结论 |
|---|---|---|---|
| Judge acc | +0.080 | +0.010 到 +0.160 | 有提升 |
| Evidence F1 | +0.105 | +0.035 到 +0.178 | 有提升 |
| Answer F1 | +0.029 | −0.018 到 +0.079 | 不显著 |

- **提升集中在重查的题上。** Grader 对 44 题判定"不够"并重查，这 44 题的 Judge acc 从 0.409 升到 0.568，
  证据召回从 0.293 升到 0.730。另外 56 题没有重查，Judge acc 是 0.768 对 0.786，基本不变。
- **"Unanswerable" 从 17 题降到 7 题**（抽样的全是可回答题）。
- **Grader 的判断大体靠谱但有漏判**：判"够了"的题里，证据确实在 top-5 的有 42/55；判"不够"的题里，
  证据其实已经在 top-5 的有 16/43。
- **代价**：每题多一次 Grader 调用，LLM token 约为 baseline 的 2.2 倍，延迟从 1.71s 升到 3.25s。

**对照：不带 Grader，直接把 top-10 交给 Writer（`baseline_dense_k10_cc_sonnet`，`--k 10`）。** Grader 版的提升
到底来自"改写的查询更准"还是"段落变多了"，要和它比才知道：

| 比较（同样的 100 题） | Judge acc 差值 | 95% 置信区间 | Context recall |
|---|---|---|---|
| top-10 − baseline（top-5） | +0.080 | +0.000 到 +0.160 | 0.780 对 0.521 |
| Grader − baseline（top-5） | +0.080 | +0.010 到 +0.160 | 0.712 对 0.521 |
| Grader − top-10 | +0.000 | −0.060 到 +0.060 | 0.712 对 0.780 |

- **Grader 没有比"直接取 top-10"更好。** 两者 Judge acc 都是 0.690；按 Grader 是否重查分组看也一样
  （重查的 44 题都是 0.568，没重查的 56 题都是 0.786）。Answer F1、Evidence F1 的差距也都不显著。
- **top-10 更便宜也更快**：每题 3980 / 153 token、1.61s，Grader 版是 6590 / 468 token、3.25s，而且 top-10
  给 Writer 的证据更全（Context hit 0.878 对 0.776，这个差距是显著的）。
- **结论：Grader 重查循环保留为可选项（`--grader`），不进默认流程。** 它的收益来自给了 Writer 更多段落，
  这件事把 `--k` 调大就能做到。瓶颈在排序（Recall@10 0.78，Recall@5 0.52），下一步更值得试的是重排序。
- 留有余地的地方：100 题样本下，"top-10 比 top-5 好"的区间下界正好是 0，严格按规则还不算显著；
  要确认需要更大的样本。

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
3. [x] Grader 和重查循环：已实现并评测。Judge acc 0.610 → 0.690，但直接取 top-10 同样是 0.690 且更便宜，
   所以 Grader 只作为可选项，不进默认流程
4. [ ] Router 和子问题拆解（加上自己造的跨论文问题）
5. [ ] FastAPI + SSE 和 Streamlit 界面
6. [ ] PDF 上传解析（PyMuPDF），Docker 打包
