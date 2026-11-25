# PaperQA：论文问答 RAG

## 项目目标

做一个回答论文问题的 RAG 系统，用 QASPER 评测。先跑通"基础 RAG + 评测"，拿到 baseline 数字，
然后一项一项往上加功能。每加一项都要有评测数字，最后形成一张消融表。

## 目录结构

```
src/paperqa/
  config.py      默认配置（路径、模型名），都能用环境变量覆盖
  qasper.py      下载/加载 QASPER、切块、从标注里提取 gold 答案和证据
  embedding.py   sentence-transformers 封装（建索引和检索共用）
  ingest.py      切块 -> 向量化 -> 存进 Chroma
  retrieve.py    检索：dense（向量）、bm25、hybrid（RRF 融合）
  agents.py      Writer（后面加 Router / Grader）
  llm.py         所有 LLM 调用都走这里，记录 token 和耗时
  metrics.py     官方 Answer F1 / Evidence F1、Recall@k、LLM-as-judge
eval/
  run_eval.py    一条命令跑评测，结果存到 eval/results/<配置名>.json
  compare.py     把所有结果打印成消融表
  paired_diff.py 两个结果文件的配对 bootstrap，判断差距是否显著
  results/       每次评测的结果（要提交到 git）
tests/           单元测试（不联网、不调 LLM）
data/            QASPER 原始数据和 Chroma 索引（不提交）
```

## 常用命令

```bash
uv sync                                                      # 安装依赖
uv run python -m paperqa.ingest                              # 下载 QASPER dev 并建索引
uv run python eval/run_eval.py --name <配置名> --retriever hybrid --retrieval-only --n 0  # 只评检索，全量，免费
uv run python eval/run_eval.py --name <配置名>                  # 完整评测（调用 LLM，花钱）
uv run python eval/compare.py                                # 打印消融表
uv run pytest                                                # 单元测试
```

## 规则

1. **每加一个功能都要跑一次评测。** 结果用新的配置名存到 `eval/results/`，并在 README 的消融表里加一行。
   没有评测数字的功能不算完成。
   - 检索指标不花钱，一律用全量 dev（`--retrieval-only --n 0`）比较。100 题样本噪声太大：hybrid 在
     seed=42 的 100 题上比 dense 高 11 个点，全量上只高 1.6 个点而且不显著。
   - 要调 LLM 的指标用 `--split dev --n 100 --seed 42`。
   - 说"B 比 A 好"之前，先跑 `eval/paired_diff.py A.json B.json`，置信区间不含 0 才算数。
2. **一次只做一项功能**，按下面 Roadmap 的顺序来，不要一次把整个系统写完。
3. **不要悄悄改评测。** 问题抽样（`eval/run_eval.py`）、指标（`src/paperqa/metrics.py`）、gold 证据匹配
   （`src/paperqa/qasper.py`）一旦改动，旧结果就不能再比了。真要改，先改或加测试，提交说明里写清楚，
   并重跑 baseline。
4. 所有 LLM 调用都通过 `src/paperqa/llm.py`，这样 token 和耗时才能统一记录。
5. 改了切块逻辑，要更新 `config.CHUNKING_VERSION`，然后用 `--rebuild` 重建索引。
6. 带 LLM 的评测会花钱。调试时先用 `--retrieval-only` 或小的 `--n`，跑完整评测前先跟用户确认。
7. 改完代码跑 `uv run pytest`。

## 评测约定

- **样本**：从 dev 集的可回答问题里（至少一位标注者给了答案）按 seed 随机抽 100 个。所有标注者都标为
  unanswerable 的问题直接跳过（dev 里有 60 个）。
- **gold 证据**：证据文本归一化空白后，和 chunk 原文精确匹配。`FLOAT SELECTED:` 开头的证据去掉前缀后匹配
  图表标题。只写了章节标题的证据（如 "Datasets"）匹配不到任何 chunk，直接丢弃。
- **Recall@k**：每位标注者的证据段落有多少比例出现在 top-k 里，取各标注者中的最大值。只统计有可匹配
  证据的问题。**Hit@k**：任意一条证据出现在 top-k 里就算命中。
- **Answer F1 / Evidence F1**：和 QASPER 官方 evaluator 一致，取各标注者中的最大值。Writer 失败算 0 分。
- **Judge acc**：LLM-as-judge，和任意一个参考答案意思一致就算对。Writer 失败算错。
- **延迟** = 检索 + Writer（不含 judge）。**Token** = Writer 的输入和输出（输出包含 thinking token）。

## Roadmap

1. [x] Baseline：向量检索 top-5，Writer 回答并标注引用
2. [x] 混合检索（BM25 + 向量，RRF 融合）：全量 dev 上相比 dense 没有显著提升，默认检索器仍是 dense
3. [ ] Grader 和重查循环
4. [ ] Router 和子问题拆解（加上自己造的跨论文问题）
5. [ ] FastAPI + SSE 和 Streamlit 界面
6. [ ] PDF 上传解析（PyMuPDF），Docker 打包
