from __future__ import annotations

import datetime as dt
import functools
import logging
import random
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd
import akshare as ak
import requests
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --------------------------- pandas 兼容补丁 --------------------------- #
# 部分第三方库仍使用 fillna(method='ffill'/'bfill')，在 pandas 2.2+ 中已移除该参数。
# 此补丁将旧式调用自动转发到 ffill()/bfill()，无需降级 pandas。
import pandas as _pd

_orig_fillna = _pd.DataFrame.fillna

def _patched_fillna(self, value=None, *, method=None, axis=None, inplace=False, limit=None, **kwargs):
    if method is not None:
        if method == "ffill":
            result = self.ffill(axis=axis, inplace=inplace, limit=limit)
        elif method == "bfill":
            result = self.bfill(axis=axis, inplace=inplace, limit=limit)
        else:
            raise ValueError(f"Unsupported fillna method: {method}")
        return result
    return _orig_fillna(self, value, axis=axis, inplace=inplace, limit=limit, **kwargs)

_pd.DataFrame.fillna = _patched_fillna  # type: ignore[method-assign]

_orig_series_fillna = _pd.Series.fillna

def _patched_series_fillna(self, value=None, *, method=None, axis=None, inplace=False, limit=None, **kwargs):
    if method is not None:
        if method == "ffill":
            result = self.ffill(axis=axis, inplace=inplace, limit=limit)
        elif method == "bfill":
            result = self.bfill(axis=axis, inplace=inplace, limit=limit)
        else:
            raise ValueError(f"Unsupported fillna method: {method}")
        return result
    return _orig_series_fillna(self, value, axis=axis, inplace=inplace, limit=limit, **kwargs)

_pd.Series.fillna = _patched_series_fillna  # type: ignore[method-assign]

# --------------------------- 全局日志配置 --------------------------- #
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "data" / "logs"

def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _PROJECT_ROOT) -> Path:
    """将配置中的路径统一解析为绝对路径：相对路径基于项目根目录。"""
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)

def _default_log_path() -> Path:
    today = dt.date.today().strftime("%Y-%m-%d")
    return _DEFAULT_LOG_DIR / f"fetch_{today}.log"

def setup_logging(log_path: Optional[Path] = None) -> None:
    """初始化日志：同时输出到 stdout 和指定文件。"""
    if log_path is None:
        log_path = _default_log_path()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )

logger = logging.getLogger("fetch_from_stocklist")

# --------------------------- 限流/封禁处理配置 --------------------------- #
COOLDOWN_SECS = 600
BAN_PATTERNS = (
    "访问频繁", "请稍后", "超过频率", "频繁访问",
    "too many requests", "429",
    "forbidden", "403",
    "max retries exceeded"
)

class RateLimiter:
    """速率限制器，严格控制请求间隔为固定时间。"""
    def __init__(self, interval_seconds=1.2):
        self.interval = interval_seconds
        self.last_request_time = 0
        self.lock = None
        try:
            import threading
            self.lock = threading.Lock()
        except ImportError:
            pass
    
    def wait(self):
        """严格控制每次请求间隔为指定时间。"""
        if self.lock:
            self.lock.acquire()
        
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.interval:
            wait_time = self.interval - elapsed
            logger.debug(f"速率限制：等待 {wait_time:.2f} 秒")
            time.sleep(wait_time)
            current_time = time.time()
        
        self.last_request_time = current_time
        
        if self.lock:
            self.lock.release()

# 创建全局速率限制器实例，严格控制间隔 1.2 秒
rate_limiter = RateLimiter(interval_seconds=1.2)

def _looks_like_ip_ban(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(pat in msg for pat in BAN_PATTERNS)

class RateLimitError(RuntimeError):
    """表示命中限流/封禁，需要长时间冷却后重试。"""
    pass

def _cool_sleep(base_seconds: int) -> None:
    jitter = random.uniform(0.9, 1.2)
    sleep_s = max(1, int(base_seconds * jitter))
    logger.warning("疑似被限流/封禁，进入冷却期 %d 秒...", sleep_s)
    time.sleep(sleep_s)

# --------------------------- 历史K线（多数据源，固定 qfq） --------------------------- #
def set_api(session=None) -> None:
    """兼容旧接口：AkShare 无需注入会话，保留此函数以免外部调用报错。"""
    return


_KLINE_COLUMNS = ["date", "open", "close", "high", "low", "volume"]


def _to_market_symbol(code: str) -> str:
    """把 6 位代码转成带市场前缀的形式（腾讯/新浪接口需要）。"""
    s = str(code).zfill(6)
    if s.startswith("6"):
        return f"sh{s}"
    if s.startswith(("0", "3")):
        return f"sz{s}"
    if s.startswith(("4", "8")):
        return f"bj{s}"
    return f"sh{s}"


def _normalize_ohlcv(df: pd.DataFrame, volume_in_shares: bool = False) -> pd.DataFrame:
    """统一成 date/open/close/high/low/volume，成交量口径统一为「手」（与东财一致）。

    volume_in_shares=True 时，输入成交量单位为「股」，需除以 100 换成「手」。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=_KLINE_COLUMNS)
    df = df[_KLINE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if volume_in_shares:
        df["volume"] = df["volume"] / 100.0
    return (df.dropna(subset=["date"])
              .drop_duplicates(subset="date")
              .sort_values("date")
              .reset_index(drop=True))


def _get_kline_em(code: str, start: str, end: str) -> pd.DataFrame:
    """源1：东方财富（AkShare stock_zh_a_hist），成交量单位：手。"""
    symbol = str(code).zfill(6)
    try:
        rate_limiter.wait()
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
    except Exception as e:
        if _looks_like_ip_ban(e):
            raise RateLimitError(str(e)) from e
        raise

    if df is None or df.empty:
        return pd.DataFrame(columns=_KLINE_COLUMNS)

    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
    })
    return _normalize_ohlcv(df)


_TX_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_TX_MAX_COUNT = 640       # 单次请求返回上限，超过上限会被静默截断
_TX_WINDOW_YEARS = 2      # 每次请求覆盖的年份数（2 年约 490 个交易日，安全小于上限）


def _get_kline_tx(code: str, start: str, end: str) -> pd.DataFrame:
    """源2：腾讯行情（按时间窗分片请求），成交量单位：手。"""
    symbol = _to_market_symbol(code)
    start_dt = dt.datetime.strptime(str(start), "%Y%m%d").date()
    end_dt = dt.datetime.strptime(str(end), "%Y%m%d").date()

    frames = []
    with requests.Session() as sess:
        sess.headers.update({"User-Agent": "Mozilla/5.0"})
        for year in range(start_dt.year, end_dt.year + 1, _TX_WINDOW_YEARS):
            win_end = min(year + _TX_WINDOW_YEARS - 1, end_dt.year)
            rate_limiter.wait()
            resp = sess.get(
                _TX_URL,
                params={"param": f"{symbol},day,{year}-01-01,{win_end}-12-31,{_TX_MAX_COUNT},qfq"},
                timeout=20,
            )
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data")
            if not isinstance(data, dict):
                raise ValueError(f"腾讯接口返回异常：{payload.get('msg')}")
            node = data.get(symbol) or {}
            rows = node.get("qfqday") or node.get("day") or []
            if rows:
                frames.append(pd.DataFrame([r[:6] for r in rows], columns=_KLINE_COLUMNS))

    if not frames:
        return pd.DataFrame(columns=_KLINE_COLUMNS)

    df = _normalize_ohlcv(pd.concat(frames, ignore_index=True))
    mask = (df["date"] >= pd.Timestamp(start_dt)) & (df["date"] <= pd.Timestamp(end_dt))
    return df[mask].reset_index(drop=True)


def _get_kline_sina(code: str, start: str, end: str) -> pd.DataFrame:
    """源3：新浪行情（AkShare stock_zh_a_daily），成交量原始单位：股。"""
    symbol = _to_market_symbol(code)
    rate_limiter.wait()
    df = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust="qfq")
    return _normalize_ohlcv(df, volume_in_shares=True)


# 顺序即优先级：某源连续 3 次失败后降级到末位，成功后提升为首选，
# 避免每只股票都在已失效的源上反复消耗重试等待。
SOURCES = [
    ("em", _get_kline_em),
    ("tx", _get_kline_tx),
    ("sina", _get_kline_sina),
]
_source_lock = threading.Lock()


def _source_snapshot() -> list[tuple[str, object]]:
    with _source_lock:
        return list(SOURCES)


def _promote_source(name: str) -> None:
    with _source_lock:
        for i, (n, _) in enumerate(SOURCES):
            if n == name and i > 0:
                SOURCES.insert(0, SOURCES.pop(i))
                logger.info("数据源 %s 可用，已提升为首选源", name)
                return


def _demote_source(name: str) -> None:
    with _source_lock:
        for i, (n, _) in enumerate(SOURCES):
            if n == name:
                if i < len(SOURCES) - 1:
                    SOURCES.append(SOURCES.pop(i))
                    logger.warning("数据源 %s 连续 3 次失败，已降级到末位", name)
                return


def get_latest_date_from_csv(csv_path: Path) -> Optional[pd.Timestamp]:
    """从现有CSV文件获取最新交易日期。"""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["date"])
        if df.empty:
            return None
        dates = pd.to_datetime(df["date"], errors="coerce")
        dates = dates.dropna()
        if dates.empty:
            return None
        return dates.max()
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _trading_calendar() -> Optional[pd.DatetimeIndex]:
    """获取 A 股交易日历（AkShare 新浪源）。失败返回 None，由调用方回退。"""
    try:
        cal = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(cal["trade_date"], errors="coerce").dropna()
        return pd.DatetimeIndex(dates.dt.normalize().unique()).sort_values()
    except Exception as e:
        logger.warning("获取交易日历失败，回退为按自然日判断：%s", e)
        return None


def latest_trading_day(today: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """返回 <= 今天的最近交易日（含今天）。日历不可用时回退为今天。"""
    today = (today or pd.Timestamp.today()).normalize()
    cal = _trading_calendar()
    if cal is None or cal.empty:
        return today
    idx = int(cal.searchsorted(today, side="right")) - 1
    return cal[idx] if idx >= 0 else today


def validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df

# --------------------------- 读取 stocklist.csv & 过滤板块 --------------------------- #

def _filter_by_boards_stocklist(df: pd.DataFrame, exclude_boards: set[str]) -> pd.DataFrame:
    ts = df["ts_code"].astype(str).str.upper()
    num = ts.str.extract(r"(\d{6})", expand=False).str.zfill(6)
    mask = pd.Series(True, index=df.index)

    if "gem" in exclude_boards:
        mask &= ~((ts.str.endswith(".SZ")) & num.str.startswith(("300", "301")))
    if "star" in exclude_boards:
        mask &= ~((ts.str.endswith(".SH")) & num.str.startswith(("688",)))
    if "bj" in exclude_boards:
        mask &= ~((ts.str.endswith(".BJ")) | num.str.startswith(("4", "8")))

    return df[mask].copy()


def load_codes_from_stocklist(stocklist_csv: Path, exclude_boards: set[str]) -> List[str]:
    df = pd.read_csv(stocklist_csv)    
    df = _filter_by_boards_stocklist(df, exclude_boards)
    codes = df["symbol"].astype(str).str.zfill(6).tolist()
    codes = list(dict.fromkeys(codes))  # 去重保持顺序
    logger.info("从 %s 读取到 %d 只股票（排除板块：%s）",
                stocklist_csv, len(codes), ",".join(sorted(exclude_boards)) or "无")
    return codes

# --------------------------- 单只抓取（增量更新） --------------------------- #
_OVERLAP_DAYS = 7      # 增量抓取时多取几天重叠数据，用于校验复权基准
_ADJUST_DRIFT_TOL = 0.01  # 重叠日期收盘价相对偏差中位数超过 1% 判定为前复权基准漂移


def _detect_adjust_drift(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> bool:
    """比对新旧数据重叠日期的收盘价，判断前复权基准是否已发生变化。

    前复权会随最新价整体重算，除权后同一历史日期的价格会变，此时增量拼接
    会在接缝处产生断层，必须丢弃历史数据重新全量抓取。
    """
    old = existing_df[["date", "close"]].copy()
    old["date"] = pd.to_datetime(old["date"], errors="coerce")
    merged = old.merge(new_df[["date", "close"]], on="date", suffixes=("_old", "_new"))
    if merged.empty:
        return False

    old_close = pd.to_numeric(merged["close_old"], errors="coerce")
    new_close = pd.to_numeric(merged["close_new"], errors="coerce")
    valid = old_close.notna() & new_close.notna() & (old_close > 0)
    if not valid.any():
        return False

    rel_diff = ((new_close[valid] - old_close[valid]).abs() / old_close[valid]).median()
    return bool(rel_diff > _ADJUST_DRIFT_TOL)


def fetch_one(
    code: str,
    start: str,
    end: str,
    out_dir: Path,
    target_date: Optional[pd.Timestamp] = None,
):
    csv_path = out_dir / f"{code}.csv"
    # 目标日期：<= 今天的最近交易日。本地数据已包含该日则无需拉取。
    if target_date is None:
        target_date = latest_trading_day()
    target_date = pd.Timestamp(target_date).normalize()
    latest_existing_date = get_latest_date_from_csv(csv_path)

    if latest_existing_date is not None:
        latest_existing_date = latest_existing_date.normalize()
        if latest_existing_date >= target_date:
            logger.debug("%s 已是最新交易日数据（最新日期: %s），跳过", code, latest_existing_date.date())
            return
        # 多取几天重叠数据，供复权基准漂移检测使用
        fetch_start = (latest_existing_date - pd.Timedelta(days=_OVERLAP_DAYS)).strftime("%Y%m%d")
    else:
        fetch_start = start

    for name, fetch in _source_snapshot():
        for attempt in range(1, 4):
            try:
                new_df = fetch(code, fetch_start, end)
                if new_df.empty:
                    logger.debug("%s [%s] 无新数据，跳过", code, name)
                    _promote_source(name)
                    return

                if latest_existing_date is not None:
                    existing_df = pd.read_csv(csv_path)
                    if _detect_adjust_drift(existing_df, new_df):
                        logger.warning("%s [%s] 检测到前复权基准漂移，丢弃历史数据改为全量重抓", code, name)
                        new_df = fetch(code, start, end)
                        if new_df.empty:
                            logger.error("%s [%s] 全量重抓返回空数据，保留原有文件", code, name)
                            return
                    else:
                        existing_df["date"] = pd.to_datetime(existing_df["date"])
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        combined_df = combined_df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
                        new_df = combined_df

                new_df = validate(new_df)
                new_df.to_csv(csv_path, index=False)
                logger.debug("%s [%s] 更新完成（%s → %s）", code, name,
                            new_df["date"].min().date() if not new_df.empty else "N/A",
                            new_df["date"].max().date() if not new_df.empty else "N/A")
                _promote_source(name)
                return
            except Exception as e:
                if _looks_like_ip_ban(e):
                    logger.error("%s [%s] 第 %d 次抓取疑似被封禁，沉睡 %d 秒", code, name, attempt, COOLDOWN_SECS)
                    _cool_sleep(COOLDOWN_SECS)
                else:
                    silent_seconds = 10
                    logger.info("%s [%s] 第 %d 次抓取失败，%d 秒后重试：%s", code, name, attempt, silent_seconds, e)
                    time.sleep(silent_seconds)
        logger.warning("%s 数据源 [%s] 连续 3 次失败，切换下一个源", code, name)
        _demote_source(name)

    logger.error("%s 所有数据源均抓取失败，已跳过！", code)



# --------------------------- 配置加载 --------------------------- #
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "fetch_kline.yaml"

def _load_config(config_path: Path = _CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("已加载配置文件：%s", config_path.resolve())
    return cfg


# --------------------------- 主入口 --------------------------- #
def main(log_path: Optional[Path] = None):
    # ---------- 读取 YAML 配置 ---------- #
    cfg = _load_config()

    # ---------- 日志路径（优先参数，其次 YAML，最后默认值） ---------- #
    if log_path is None:
        cfg_log = cfg.get("log")
        log_path = _resolve_cfg_path(cfg_log) if cfg_log else _default_log_path()
    setup_logging(log_path)
    logger.info("日志文件：%s", Path(log_path).resolve())

    # ---------- 日期解析 ---------- #
    raw_start = str(cfg.get("start", "20190101"))
    raw_end   = str(cfg.get("end",   "today"))
    start = dt.date.today().strftime("%Y%m%d") if raw_start.lower() == "today" else raw_start
    end   = dt.date.today().strftime("%Y%m%d") if raw_end.lower()   == "today" else raw_end

    out_dir = _resolve_cfg_path(cfg.get("out", "./data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 从 stocklist.csv 读取股票池 ---------- #
    stocklist_path = _resolve_cfg_path(cfg.get("stocklist", "./pipeline/stocklist.csv"))
    exclude_boards = set(cfg.get("exclude_boards") or [])
    codes = load_codes_from_stocklist(stocklist_path, exclude_boards)

    if not codes:
        logger.error("stocklist 为空或被过滤后无代码，请检查。")
        sys.exit(1)

    # ---------- 目标最新交易日：本地数据已含该日则跳过拉取 ---------- #
    target_date = latest_trading_day()
    logger.info("目标最新交易日：%s（按交易日历，节假日/周末自动回退）", target_date.date())

    logger.info(
        "开始抓取 %d 支股票 | 数据源:AkShare(日线,qfq) | 日期:%s → %s | 排除:%s",
        len(codes), start, end, ",".join(sorted(exclude_boards)) or "无",
    )

    # ---------- 多线程抓取（增量更新，已是最新交易日的股票自动跳过） ---------- #
    workers = int(cfg.get("workers", 8))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                fetch_one,
                code,
                start,
                end,
                out_dir,
                target_date,
            )
            for code in codes
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            pass

    logger.info("全部任务完成，数据已保存至 %s", out_dir.resolve())

if __name__ == "__main__":
    main()
