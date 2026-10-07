"""
prepare_review_features.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
为「Trae 直接分析数据」准备指标特征卡，替代原来的读图打分（trae_review.py + llama.cpp）。

子命令：
    features   读候选清单，从 data/raw/<code>.csv 计算每只股票的指标，写入
               data/review/<pick_date>/features/<code>.json
               以及合并文件 features_all.json（供 Trae 一次性读取）
    aggregate  读取 Trae 打好的 data/review/<pick_date>/<code>.json，
               按项目现有逻辑汇总为 suggestion.json 与 suggestion.md（复盘报告）

用法：
    python agent/prepare_review_features.py features
    python agent/prepare_review_features.py aggregate

配置：config/gemini_review.yaml（candidates / output_dir / suggest_min_score）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "gemini_review.yaml"
RAW_DIR = ROOT / "data" / "raw"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_reviewer import BaseReviewer  # noqa: E402


def _r(x, nd: int = 4):
    """安全四舍五入：None / NaN / 非数值一律返回 None。"""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(x), nd)


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_features(code: str, df: pd.DataFrame) -> dict:
    """从单只股票的日线 DataFrame 计算评分所需的指标特征卡。"""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    last_close = float(close.iloc[-1])

    ma5, ma10 = close.rolling(5).mean(), close.rolling(10).mean()
    ma20, ma60 = close.rolling(20).mean(), close.rolling(60).mean()

    trend = {
        "ma5": _r(ma5.iloc[-1]),
        "ma10": _r(ma10.iloc[-1]),
        "ma20": _r(ma20.iloc[-1]),
        "ma60": _r(ma60.iloc[-1]),
        "ma_bull": bool(ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]) if n >= 60 else None,
        "price_vs_ma20_pct": _r((last_close - ma20.iloc[-1]) / ma20.iloc[-1] * 100) if n >= 20 else None,
        "ma20_slope_5d_pct": _r((ma20.iloc[-1] - ma20.iloc[-6]) / ma20.iloc[-6] * 100) if n >= 26 else None,
        "days_close_above_ma20_20d": int((close.iloc[-20:] > ma20.iloc[-20:]).sum()) if n >= 20 else None,
    }

    win = df.iloc[-250:]
    hi, lo = float(win["high"].max()), float(win["low"].min())
    high_60 = float(high.iloc[-60:].max()) if n >= 60 else float(high.max())
    low_60 = float(low.iloc[-60:].min()) if n >= 60 else float(low.min())
    position = {
        "high_60": _r(high_60),
        "low_60": _r(low_60),
        "pct_from_high_60": _r((last_close - high_60) / high_60 * 100),
        "range_pos_pct_250": _r((last_close - lo) / (hi - lo) * 100) if hi > lo else None,
        "gain_20d_pct": _r((last_close / close.iloc[-21] - 1) * 100) if n >= 21 else None,
        "gain_60d_pct": _r((last_close / close.iloc[-61] - 1) * 100) if n >= 61 else None,
    }

    d = df.copy()
    d["chg"] = d["close"].diff()
    vol_ma20 = vol.rolling(20).mean().iloc[-1]
    w20 = d.iloc[-20:]
    up_vol = w20.loc[w20["chg"] > 0, "volume"].mean()
    down_vol = w20.loc[w20["chg"] < 0, "volume"].mean()
    mx = w20["volume"].idxmax()
    volume = {
        "vol_ma5": _r(vol.rolling(5).mean().iloc[-1]),
        "vol_ma20": _r(vol_ma20),
        "vol5_vs_vol20": _r(vol.rolling(5).mean().iloc[-1] / vol_ma20) if vol_ma20 else None,
        "up_down_vol_ratio": _r(up_vol / down_vol) if down_vol and not pd.isna(down_vol) else None,
        "last_bear_vol_ratio": None,
        "max_vol_is_bullish": bool(w20.loc[mx, "close"] > w20.loc[mx, "open"]),
        "max_vol_date": str(w20.loc[mx, "date"])[:10],
    }
    bears = d[d["chg"] < 0]
    if len(bears) and bears.index[-1] - 1 >= 0:
        i = bears.index[-1]
        volume["last_bear_vol_ratio"] = _r(vol.iloc[i] / vol.iloc[i - 1])

    prev_high20 = float(high.iloc[-21:-1].max()) if n >= 21 else None
    swing = (high_60 - low_60) / low_60 * 100 if low_60 else None
    abnormal = {
        "max_gain_60d_pct": _r((d["chg"] / close.shift(1) * 100).iloc[-60:].max()),
        "breakout_20d": bool(
            prev_high20 is not None
            and last_close > prev_high20
            and vol.iloc[-1] > 1.5 * vol_ma20
        ),
        "swing_gain_pct": _r(swing),
        "swing_gain_under_50": bool(swing is not None and swing < 50),
    }

    wk = (
        df.assign(dt=pd.to_datetime(df["date"]))
        .set_index("dt")["close"]
        .resample("W")
        .last()
        .dropna()
    )
    wma5, wma10, wma20 = wk.rolling(5).mean(), wk.rolling(10).mean(), wk.rolling(20).mean()
    weekly = {
        "ma_bull": bool(wma5.iloc[-1] > wma10.iloc[-1] > wma20.iloc[-1]) if len(wk) >= 20 else None,
        "close_vs_weekly_ma20_pct": _r((wk.iloc[-1] - wma20.iloc[-1]) / wma20.iloc[-1] * 100) if len(wk) >= 20 else None,
    }

    return {
        "code": code,
        "date": str(df["date"].iloc[-1])[:10],
        "bars": n,
        "close": _r(last_close),
        "trend": trend,
        "position": position,
        "volume": volume,
        "abnormal": abnormal,
        "weekly": weekly,
    }


def _load_candidates(cfg: dict) -> tuple[str, list[dict]]:
    cand_file = _resolve(cfg.get("candidates", "data/candidates/candidates_latest.json"))
    with open(cand_file, encoding="utf-8") as f:
        data = json.load(f)
    return data["pick_date"], list(data.get("candidates", []))


def _formula_of(cand: dict) -> dict:
    """从候选条目提取来源公式标签（strategy + 完整命中列表）。"""
    strategy = cand.get("strategy", "") or ""
    formulas = (cand.get("extra") or {}).get("formulas") or ([strategy] if strategy else [])
    return {"strategy": strategy, "formulas": formulas}


def run_features(cfg: dict) -> None:
    pick_date, candidates = _load_candidates(cfg)
    codes = [c["code"] for c in candidates]
    meta = {c["code"]: _formula_of(c) for c in candidates}
    out_dir = _resolve(cfg.get("output_dir", "data/review")) / pick_date / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    cards, missing = [], []
    for code in codes:
        csv_path = RAW_DIR / f"{code}.csv"
        if not csv_path.exists():
            missing.append(code)
            continue
        card = compute_features(code, pd.read_csv(csv_path))
        card.update(meta.get(code, {}))
        with open(out_dir / f"{code}.json", "w", encoding="utf-8") as f:
            json.dump(card, f, ensure_ascii=False, indent=2)
        cards.append(card)

    with open(out_dir / "features_all.json", "w", encoding="utf-8") as f:
        json.dump(
            {"pick_date": pick_date, "count": len(cards), "features": cards},
            f, ensure_ascii=False, indent=2,
        )

    print(f"[INFO] pick_date={pick_date}，候选 {len(codes)} 只，生成指标卡 {len(cards)} 只")
    if missing:
        print(f"[WARN] 缺少行情 CSV，已跳过：{missing}")
    print(f"[INFO] 合并文件：{out_dir / 'features_all.json'}")


DIMENSIONS = [
    ("trend_structure", "趋势结构"),
    ("price_position", "价格位置"),
    ("volume_behavior", "量价行为"),
    ("previous_abnormal_move", "前期异动"),
]

REASONING_FIELDS = [
    ("trend_reasoning", "趋势"),
    ("position_reasoning", "位置"),
    ("volume_reasoning", "量价"),
    ("abnormal_move_reasoning", "异动"),
    ("signal_reasoning", "信号"),
]


def _score_of(r: dict) -> float:
    """安全取总分，非法值一律按 0 处理。"""
    try:
        return float(r.get("total_score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _fmt_score(x, nd: int = 1) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _formula_label(r: dict) -> str:
    """把结果的 strategy / formulas 渲染成简短标签，如 "f1"、"f1+f5"。"""
    formulas = r.get("formulas") or []
    if not formulas and r.get("strategy"):
        formulas = [r["strategy"]]
    return "+".join(formulas) if formulas else "—"


def _formula_tokens(r: dict) -> list[str]:
    """取一条结果的来源标签列表（公式名或策略名）。"""
    tokens = list(r.get("formulas") or [])
    if not tokens and r.get("strategy"):
        tokens = [r["strategy"]]
    return [str(t) for t in tokens]


def _formula_stats(results: list[dict], rec_codes: set) -> list[tuple[str, int, int]]:
    """按单个公式统计 (公式, 命中数, 推荐数)，f1~f5 优先排序，其余按名称。"""
    seen: dict[str, list[int]] = {}
    for r in results:
        for token in _formula_tokens(r) or ["—"]:
            bucket = seen.setdefault(token, [0, 0])
            bucket[0] += 1
            if r.get("code") in rec_codes:
                bucket[1] += 1

    def _order(token: str) -> tuple[int, str]:
        return (0, f"{int(token[1:]):02d}") if token[:1] == "f" and token[1:].isdigit() else (1, token)

    return [(token, cnt[0], cnt[1]) for token, cnt in sorted(seen.items(), key=lambda kv: _order(kv[0]))]


def render_markdown(pick_date: str, results: list[dict], suggestion: dict) -> str:
    """把评分结果渲染成 markdown 复盘报告。"""
    min_score = suggestion.get("min_score_threshold", 0)
    recs = suggestion.get("recommendations", [])
    rec_codes = {r.get("code") for r in recs}

    # 明细顺序：先推荐（按分数降序），再未达标（按分数降序）
    ordered = sorted(
        results,
        key=lambda r: (r.get("code") in rec_codes, _score_of(r)),
        reverse=True,
    )

    lines = [
        f"# A股选股报告 — {pick_date}",
        "",
        f"- 选股日期：{pick_date}",
        f"- 评分总数：{suggestion.get('total_reviewed', len(results))} 只",
        f"- 推荐门槛：total_score ≥ {min_score}",
        f"- 推荐数量：{len(recs)} 只",
        "",
        "## 分公式统计",
        "",
    ]

    stats = _formula_stats(results, rec_codes)
    if stats:
        lines += ["| 公式 | 命中数 | 推荐数 |", "| --- | --- | --- |"]
        for label, hit, rec in stats:
            lines.append(f"| {label} | {hit} | {rec} |")
    else:
        lines.append("无。")

    lines += ["", "## 推荐标的", ""]

    if recs:
        lines += [
            "| 排名 | 代码 | 公式 | 总分 | 信号 | 研判 | 备注 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in recs:
            lines.append(
                f"| {r.get('rank', '?')} | {r.get('code', '?')} | {_formula_label(r)} | "
                f"{_fmt_score(r.get('total_score'))} | {r.get('signal_type', '')} | "
                f"{r.get('verdict', '')} | {r.get('comment', '')} |"
            )
    else:
        lines.append(f"本次无标的达到推荐门槛（total_score ≥ {min_score}）。")

    excluded = [r for r in ordered if r.get("code") not in rec_codes]
    lines += ["", "## 未达标标的", ""]
    if excluded:
        lines += [
            "| 代码 | 公式 | 总分 | 信号 | 研判 | 备注 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in excluded:
            lines.append(
                f"| {r.get('code', '?')} | {_formula_label(r)} | {_fmt_score(r.get('total_score'))} | "
                f"{r.get('signal_type', '')} | {r.get('verdict', '')} | {r.get('comment', '')} |"
            )
    else:
        lines.append("无。")

    lines += ["", "## 评分明细", ""]
    for r in ordered:
        code = r.get("code", "?")
        tag = "推荐" if code in rec_codes else "未达标"
        lines += [
            f"### {code} — {_fmt_score(r.get('total_score'))} 分 · "
            f"{r.get('verdict', '')} · {r.get('signal_type', '')} · 公式 {_formula_label(r)}（{tag}）",
            "",
        ]
        if r.get("comment"):
            lines += [f"> {r['comment']}", ""]

        scores = r.get("scores", {}) or {}
        lines += ["| 维度 | 得分 |", "| --- | --- |"]
        for key, label in DIMENSIONS:
            lines.append(f"| {label} | {scores.get(key, '—')} |")
        lines.append("")

        for key, label in REASONING_FIELDS:
            if r.get(key):
                lines.append(f"- **{label}**：{r[key]}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_aggregate(cfg: dict) -> None:
    pick_date, candidates = _load_candidates(cfg)
    codes = [c["code"] for c in candidates]
    meta = {c["code"]: _formula_of(c) for c in candidates}
    out_root = _resolve(cfg.get("output_dir", "data/review"))
    out_dir = out_root / pick_date
    min_score = float(cfg.get("suggest_min_score", 4.0))

    results, missing = [], []
    for code in codes:
        fp = out_dir / f"{code}.json"
        if not fp.exists():
            missing.append(code)
            continue
        with open(fp, encoding="utf-8") as f:
            r = json.load(f)
        r.setdefault("code", code)
        # 候选清单里的公式标签优先（评分文件通常不含），缺失时回填
        for k, v in (meta.get(code) or {}).items():
            if not r.get(k):
                r[k] = v
        results.append(r)

    if not results:
        print("[ERROR] 没有找到任何 <code>.json 评分结果，无法汇总。")
        sys.exit(1)

    reviewer = BaseReviewer({
        "prompt_path": _resolve(cfg.get("prompt_path", "agent/prompt.md")),
        "kline_dir": _resolve(cfg.get("kline_dir", "data/kline")),
        "output_dir": out_root,
    })
    suggestion = reviewer.generate_suggestion(pick_date, results, min_score)

    with open(out_dir / "suggestion.json", "w", encoding="utf-8") as f:
        json.dump(suggestion, f, ensure_ascii=False, indent=2)

    md_path = out_dir / "suggestion.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(pick_date, results, suggestion))

    print(f"[INFO] 已读取评分 {len(results)} 只，推荐门槛 score >= {min_score}")
    if missing:
        print(f"[WARN] 缺少评分文件，已跳过：{missing}")
    print(f"[INFO] 推荐 {len(suggestion['recommendations'])} 只，写入 {out_dir / 'suggestion.json'}")
    print(f"[INFO] 复盘报告：{md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="为 Trae 数据打分准备指标 / 汇总推荐")
    parser.add_argument("command", choices=["features", "aggregate"])
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.command == "features":
        run_features(cfg)
    else:
        run_aggregate(cfg)


if __name__ == "__main__":
    main()