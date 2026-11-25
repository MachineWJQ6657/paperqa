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

完整评测（检索 + Writer + LLM-as-judge）需要 Anthropic API key：

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

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
- **检索**：QASPER 的每个问题只针对一篇论文，所以检索时按 `paper_id` 过滤。
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

常用参数：`--n 20`（先拿少量题试跑）、`--no-judge`（只算官方 F1，省掉 judge 的调用）、`--effort low`、
`--model`、`--judge-model`、`--workers`（并发请求数，默认 4）。

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

**样本噪声**：同样是 dense 检索，100 题样本的 Recall@5 是 0.521，dev 全部 945 道可回答题是 0.553。
所以用 100 题对比配置时，差距在 5 个点以内很可能只是噪声。方法稳定后用 `--n 0` 跑全量。

## 消融表

全部基于 dev、n=100、seed=42。

| 配置 | Recall@5 | Hit@5 | Answer F1 | Evidence F1 | Judge acc | 延迟 (s) | Writer token 输入 / 输出 |
|---|---|---|---|---|---|---|---|
| dense_retrieval（只评检索） | 0.521 | 0.592 | – | – | – | 0.025 | – |
| baseline_dense（dense top-5 + Writer） | 待跑 | | | | | | |

同一次检索的 Recall@k 曲线：R@1 = 0.209，R@3 = 0.407，R@5 = 0.521，R@10 = 0.780。

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
2. [ ] 混合检索（BM25 + 向量，RRF 融合）
3. [ ] Grader 和重查循环
4. [ ] Router 和子问题拆解（加上自己造的跨论文问题）
5. [ ] FastAPI + SSE 和 Streamlit 界面
6. [ ] PDF 上传解析（PyMuPDF），Docker 打包
