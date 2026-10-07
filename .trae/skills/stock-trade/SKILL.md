---
name: stock-trade
description: Score A-share candidates from computed indicator cards and produce stock picks. Use when the user asks to 选股, 跑选股流程, or 用数据打分选股, or to run prepare_review_features, fetch_kline, or preselect. Do not use for general stock questions.
---

# StockTradebyZ 数据选股流程

用「指标卡 + 你自己打分」替代原来的读图打分（`trae_review.py` + 本地 llama.cpp）。**不依赖图片，也不依赖任何本地模型**——你是多模态/推理模型，直接读算好的指标 JSON 来评分。

## 运行约定

- 项目根目录：`d:\code\StockTradebyZ`（所有命令都在这里执行）。
- 必须使用项目虚拟环境的解释器：`.\.venv\Scripts\python.exe`。
- 候选股规模由 B1/砖型图 + 5 条公式（含 `max_per_formula` 截断）决定，通常 20~40 只，逐只打分可控。

## 股票池

- 成分股来源：**沪深300 + 中证500**（`pipeline/stocklist_hs300_zz500.csv`）。
- **排除科创板**（688 开头，`star`）与**北交所**（4/8 开头、`.BJ`，`bj`）；创业板（300/301）保留。
- 板块排除由 [config/fetch_kline.yaml](config/fetch_kline.yaml) 的 `exclude_boards: ["star", "bj"]` 控制，在 `pipeline/fetch_kline.py` 读取 stocklist 时生效（800 只 → 709 只）。
- 注意：`pipeline.cli preselect` 是直接扫描 `data/raw/*.csv`，**不再按板块过滤**。若 `data/raw` 里残留被排除板块的旧 CSV（如 `688*.csv`），初选仍会把它们算进来。切换股票池后需先清理 `data/raw` 中对应文件。

## 选股公式（f1~f5）

初选除 B1 / 砖型图外，还内置 5 条公式：**各自独立、命中任一即入选、取并集**。
实现在 `pipeline/Selector.py` 的 `FormulaSelector`，参数见
[config/rules_preselect.yaml](config/rules_preselect.yaml) 的 `formulas` 段。

| 公式 | KDJ | MACD（DIF/DEA） | BBI | 涨跌幅 | 其他 |
|---|---|---|---|---|---|
| f1 | J<12 | DIF>0 或 DEA>0 | BBI>60日线 | -2%~2% | — |
| f2 | J<20 | DIF>0 且 DEA>0；(DIF<0.2 或 DEA<0.2)；DIF<DEA | BBI>60日线 | -2%~2% | — |
| f3 | J<30 | DIF<0 或 DEA<0；DIF>DEA | BBI>5日线 | — | — |
| f4 | 日线 J<10 | — | — | — | 周线 J<0 |
| f5 | — | 周线 DIF 上穿 DEA；(周线 DIF>0 或 周线 DEA>0) | 周线 BBI>60日线 | — | — |

口径说明：

- KDJ 阈值一律指 **J 值**（与 B1 的 `j_threshold` 口径一致）。
- DIF/DEA 是**价格单位**的绝对值，f2 的 `0.2` 为绝对阈值；高价股可能天然不满足，可用 `f2.dif_dea_abs_max` 调整或改成相对值。
- BBI = (MA3+MA6+MA12+MA24)/4；「60日线」「5日线」取**日线**均线。
- f5 的「周线」指标按**周线 OHLC**（ISO 周，每周最后交易日）计算后前推到日线；金叉指最近 `cross_lookback_weeks` 根周线内出现上穿（默认 1 根）。选股日若在周中，最后一根周线是未走完的当周。
- **非ST / 非北交所 / 非科创** 由股票池天然满足：池内已排除科创板与北交所，且沪深300+中证500 当前无 ST 成分股。

命中结果写入候选的 `strategy` 字段（`f1`~`f5`；同一只命中多条时取最靠前者），完整命中列表在 `extra.formulas`。

**候选规模控制**：公式阈值较松，2026-09-30 并集曾达 95 只。`formulas.max_per_formula` 会在**每条公式内按成交额降序取前 N 只**再取并集（默认 f1~f3/f5 各 5 只、f4 取 10 只 → 约 30 只；未列出的公式不截断）。命中多条公式的股票，只要任一公式入选即保留。需更松/更紧时改这个映射即可；设为 `{}` 关闭截断。

## 流程

### 步骤 1-3 一条命令跑完（行情 → 初选 → 指标卡）

```bash
.\.venv\Scripts\python.exe run_all.py
```

依次执行：拉取行情（AkShare，免 token）→ 量化初选 → 生成指标卡，然后打印「等待 AI 打分」的提示并结束。
已有最新行情时可加 `--skip-fetch` 跳过下载。

指标卡输出：

- `data/review/<pick_date>/features/<code>.json` — 单只指标卡
- `data/review/<pick_date>/features/features_all.json` — 合并文件（含 `pick_date`、`count`、`features` 数组）

每张指标卡带 `strategy`（来源公式，如 `f1`）与 `formulas`（完整命中列表），打分时据此判断该股属于哪条逻辑家族。

### 步骤 4 由你（Trae）打分 —— 本步骤不用跑命令

对 `features_all.json` 里的每只股票：

1. 按下面的「指标 → 维度映射」对 4 个维度各打 1~5 分
2. **按 `strategy`/`formulas` 调整侧重**：5 条公式是不同逻辑家族——f1/f2 是零轴附近回踩、f3 是反转启动、f4 是超卖抄底（左侧）、f5 是以周线结构为准。具体要求见 [../agent/prompt.md](../../agent/prompt.md) 的「按公式类型调整评分侧重」，避免用「右侧主升」的尺子误杀左侧机会（尤其 f4）
3. 计算 `total_score`（保留 1 位小数）
4. 用 `Write` 写出 `data/review/<pick_date>/<code>.json`（schema 见下）

### 步骤 5 汇总并查看推荐

```bash
.\.venv\Scripts\python.exe run_all.py --start-from 4
```

执行 `aggregate` 生成两份输出并打印推荐表格（格式与项目原流程完全一致）：

- `suggestion.json` — 机器可读的推荐结果（推荐项含 `strategy`/`formulas`，并有 `by_formula` 分公式统计）
- `suggestion.md` — 人工可读的 markdown 复盘报告（分公式统计 + 推荐表 + 未达标表 + 每只股的维度得分与研判理由）

### 分步命令（可选，调试用）

```bash
.\.venv\Scripts\python.exe -m pipeline.fetch_kline
.\.venv\Scripts\python.exe -m pipeline.cli preselect
.\.venv\Scripts\python.exe agent/prepare_review_features.py features
.\.venv\Scripts\python.exe agent/prepare_review_features.py aggregate
```

### 步骤 6 汇报

读取 `suggestion.json`，把 `recommendations` 汇总成表格回复用户：排名、代码、公式、总分、信号、研判、备注；再按 `by_formula` 给出各公式的命中数/推荐数。若无达标股票，说明当前 `min_score_threshold` 门槛值。

汇报时一并给出复盘报告路径 `data/review/<pick_date>/suggestion.md`（已由 `aggregate` 自动生成，含完整评分理由，无需你再手写 md）。

## 打分标准（源自 [../agent/prompt.md](../../agent/prompt.md)）

4 个维度，各 1~5 分：

```
total_score = trend_structure*0.20 + price_position*0.20 + volume_behavior*0.30 + previous_abnormal_move*0.30
```

判定规则：

- **BUY**：`total_score` ≥ 4.0
- **WATCH**：3.2 ≤ `total_score` < 4.0
- **FAIL**：`total_score` < 3.2
- 特殊规则：`volume_behavior` = 1 → 必须 FAIL

## 指标 → 维度映射

用 `features_all.json` 里的字段。以下是启发式映射，可结合多字段综合判断；**信息不足时下调评分**。

### 1 trend_structure（趋势结构）

- `trend.ma_bull` = true 且 `trend.ma20_slope_5d_pct` > 0 → 4~5
- `trend.ma_bull` = false，但 `trend.ma20_slope_5d_pct` > 0 或 `trend.price_vs_ma20_pct` > 0 → 3
- 均线纠缠、`trend.ma20_slope_5d_pct` ≈ 0 → 2
- `trend.ma20_slope_5d_pct` < 0 且 `trend.price_vs_ma20_pct` < 0 → 1
- 可参考 `weekly.ma_bull`、`trend.days_close_above_ma20_20d` 微调

### 2 price_position（价格位置）

- `position.range_pos_pct_250` 偏低（< 40）、`position.pct_from_high_60` 显示刚突破、`position.gain_60d_pct` 不大 → 5
- 中位突破（`range_pos_pct_250` 40~70）→ 3~4
- 接近前高（`pct_from_high_60` 接近 0）→ 2
- `range_pos_pct_250` 很高或 `gain_60d_pct` 很大、价格远离均线 → 1

### 3 volume_behavior（量价行为）

- `volume.vol5_vs_vol20` > 1 且 `volume.up_down_vol_ratio` > 1.2 且 `volume.max_vol_is_bullish` = true 且 `volume.last_bear_vol_ratio` ≤ 0.6 → 5
- 部分满足 → 3~4
- `volume.up_down_vol_ratio` < 1 或 `volume.max_vol_is_bullish` = false → 2
- `volume.last_bear_vol_ratio` > 1.5（放量阴线）或明显下跌放量 → 1

### 4 previous_abnormal_move（前期异动）

- `abnormal.breakout_20d` = true 且 `abnormal.swing_gain_under_50` = true → 5
- 有放量突破但结构一般 → 3~4
- 普通上涨（`abnormal.max_gain_60d_pct` 平淡） → 2
- `abnormal.swing_gain_pct` > 100 → 1

## 输出 schema（`data/review/<pick_date>/<code>.json`）

严格按 [../agent/prompt.md](../../agent/prompt.md) 的格式：

```json
{
  "trend_reasoning": "趋势分析理由",
  "position_reasoning": "位置分析理由",
  "volume_reasoning": "量价分析理由",
  "abnormal_move_reasoning": "异动分析理由",
  "signal_reasoning": "信号分析理由",
  "scores": {
    "trend_structure": 5,
    "price_position": 4,
    "volume_behavior": 4,
    "previous_abnormal_move": 3
  },
  "total_score": 4.1,
  "signal_type": "trend_start",
  "verdict": "BUY",
  "comment": "一句中文交易员点评",
  "code": "600410"
}
```

- `total_score` 按上面权重计算，保留 1 位小数
- `signal_type` ∈ `trend_start` / `rebound` / `distribution_risk`
- `verdict` ∈ `BUY` / `WATCH` / `FAIL`
- `code` 必填（`aggregate` 依赖它做汇总）

## 配置

- [config/fetch_kline.yaml](config/fetch_kline.yaml) — 抓取区间、股票池、排除板块、输出目录、并发数
- [config/rules_preselect.yaml](config/rules_preselect.yaml) — 初选规则与阈值
- [config/gemini_review.yaml](config/gemini_review.yaml) — `candidates` / `output_dir` / `prompt_path` / `suggest_min_score`（推荐门槛）

## 注意事项

- `run_all.py` 的步骤 1-3 可自动执行；**步骤 4（AI 打分）必须由你完成**，否则 `--start-from 4` 汇总时会因缺少 `<code>.json` 而报错。
- 本 skill 不依赖图片，也不依赖本地 llama.cpp；`dashboard/export_kline_charts.py` 与 `agent/trae_review.py` 属于旧流程，无需调用。
- 若某次初选候选超过 20 只，先提示用户收窄候选池，再分批打分，避免单轮过长。
- `aggregate` 只汇总候选清单中存在的 `<code>.json`，缺失的会以 `[WARN]` 跳过；若推荐数异常，先确认是否每只股都写了评分文件。
- AkShare 无需 token，抓取失败多为网络或东财限流，可 `pip install akshare --upgrade`。