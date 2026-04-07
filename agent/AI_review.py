"""
trae_review.py
~~~~~~~~~~~~~~~~
使用 TRAE 内置的 AI 对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构。

用法：
    python agent/trae_review.py
    python agent/trae_review.py --config config/gemini_review.yaml

配置：
    默认读取 config/gemini_review.yaml。

输出：
    ./data/review/{pick_date}/{code}.json   每支股票的评分 JSON
    ./data/review/{pick_date}/suggestion.json  汇总推荐建议
"""

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from base_reviewer import BaseReviewer

# ────────────────────────────────────────────────
# 内置 Prompt（极简版）
# ────────────────────────────────────────────────
DEFAULT_PROMPT = """
专业波段交易员根据日线图判断波段爆发潜力。只根据图中可见信息分析。

## 评分标准

### 1️⃣ 趋势结构
5 分：均线刚多头，短期上拐上穿长期，间距合理
4 分：多头排列，沿均线上行
3 分：均线偏多但不流畅
2 分：均线交叉趋势不清
1 分：空头排列向下

### 2️⃣ 价格位置
5 分：中低位刚突破，空间大压力轻
4 分：中位突破向前高
3 分：临近前高
2 分：接近历史高位
1 分：高位过热远离均线

### 3️⃣ 量价行为
5 分：上涨放量首阴缩量 (≤阳半),最大量在涨段
4 分：上涨放量回调缩量
3 分：量价中性
2 分：上涨无放量回调不缩
1 分：放量大阴，最大量在跌 K

### 4️⃣ 前期异动
5 分：异常放量阳突破，涨幅<50%
4 分：明显放量阳，涨幅<50%
3 分：放量不突出，涨幅<50%
2 分：普通上涨或涨幅 50-100%
1 分：涨幅>100% 或放量大阴出货

## 权重
trend:0.20 | pos:0.20 | vol:0.30 | abnormal:0.30

## 判定
PASS≥4.0 | WATCH 3.2-4.0 | FAIL<3.2 | vol=1→必 FAIL

## 信号
trend_start | rebound | distribution_risk

## JSON 输出
{"trend_reasoning":"string","position_reasoning":"string","volume_reasoning":"string","abnormal_move_reasoning":"string","signal_reasoning":"string(周线 + 量价 + 异动 + 风险)","scores":{"trend_structure":1-5,"price_position":1-5,"volume_behavior":1-5,"previous_abnormal_move":1-5},"total_score":1.0-5.0,"signal_type":"trend_start/rebound/distribution_risk","verdict":"BUY/WATCH/FAIL","comment":"一句中文点评"}
"""


def _find_available_port(start_port: int = 11434, max_attempts: int = 100) -> int:
    """查找可用的端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError("无法找到可用端口")


def _restart_ollama(model_name: str = "qwen3-vl:4b") -> int:
    """重启 Ollama 服务，返回可用的端口"""
    import platform

    system = platform.system()

    # 停止现有 Ollama 进程
    if system == "Windows":
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "ollama"], capture_output=True)

    time.sleep(2)

    # 找到可用端口
    port = _find_available_port()

    # 设置环境变量并启动 Ollama（启用 Vulkan 加速）
    env = {**subprocess.os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_VULKAN": "1"}

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True if system != "Windows" else False,
    )

    # 等待 Ollama 启动
    max_wait = 30
    for _ in range(max_wait):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(("127.0.0.1", port))
                print(f"Ollama 已启动，端口：{port}")
                break
        except (OSError, socket.timeout):
            time.sleep(1)
    else:
        raise RuntimeError("Ollama 启动失败")

    # 预热模型：先检查模型列表，然后加载
    print(f"正在预热模型 {model_name}（这可能需要几分钟）...")
    base_url = f"http://127.0.0.1:{port}"
    
    # 步骤 1: 检查模型列表
    try:
        tags_response = requests.get(f"{base_url}/api/tags", timeout=10)
        if tags_response.status_code == 200:
            tags_data = tags_response.json()
            models = tags_data.get("models", [])
            model_names = [m.get("name", "") for m in models]
            print(f"已安装模型：{', '.join(model_names)}")
            if model_name not in model_names:
                print(f"警告：模型 {model_name} 未找到")
                return port
    except Exception as e:
        print(f"获取模型列表失败：{e}")
    
    # 步骤 2: 发送预热请求
    warmup_url = f"{base_url}/api/generate"
    warmup_payload = {
        "model": model_name,
        "prompt": "hi",
        "stream": False
    }
    for i in range(180):
        try:
            response = requests.post(warmup_url, json=warmup_payload, timeout=30)
            if response.status_code == 200:
                print("模型已加载完成")
                return port
            elif response.status_code == 404:
                error_msg = response.json().get("error", "Unknown error")
                print(f"模型 {model_name} 不存在：{error_msg}")
                return port
            else:
                print(f"模型加载中... ({i+1}s) 状态码：{response.status_code}")
        except Exception:
            print(f"模型加载中... ({i+1}s) 等待模型响应")
        time.sleep(1)
    
    print("模型预热超时，继续执行...")
    return port


# ────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "gemini_review.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # 路径参数（相对路径默认基于项目根目录）
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # 模型参数
    "request_delay": 5,
    "skip_existing": False,
    "suggest_min_score": 4.0,
    # Ollama 配置
    "ollama_url": "http://localhost:11434/api/generate",
    "ollama_model": "qwen3-vl:4b",
}


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _ROOT) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    cfg_path = config_path or _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")

    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = {**DEFAULT_CONFIG, **raw}

    # BaseReviewer 依赖这些路径字段为 Path 对象
    cfg["candidates"] = _resolve_cfg_path(cfg["candidates"])
    cfg["kline_dir"] = _resolve_cfg_path(cfg["kline_dir"])
    cfg["output_dir"] = _resolve_cfg_path(cfg["output_dir"])
    cfg["prompt_path"] = _resolve_cfg_path(cfg["prompt_path"])

    return cfg


class TraeReviewer(BaseReviewer):
    _ollama_restarted = False
    _timeout_count = 0
    _max_timeout_before_restart = 3

    def __init__(self, config):
        # 使用内置 prompt，不从文件加载
        self.config = config
        self.prompt = DEFAULT_PROMPT.strip()
        self.kline_dir = Path(config["kline_dir"])
        self.output_dir = Path(config["output_dir"])
        
        if not TraeReviewer._ollama_restarted:
            print("正在重启 Ollama 服务...")
            model_name = config.get("ollama_model", "qwen3-vl:4b")
            port = _restart_ollama(model_name)
            config["ollama_url"] = f"http://localhost:{port}/api/generate"
            TraeReviewer._ollama_restarted = True
        self.ollama_url = config.get("ollama_url", "http://localhost:11434/api/generate")
        self.ollama_model = config.get("ollama_model", "qwen3-vl:4b")

    def _restart_ollama_if_needed(self):
        """检查是否需要重启 Ollama，并在需要时重启"""
        TraeReviewer._timeout_count += 1
        if TraeReviewer._timeout_count >= TraeReviewer._max_timeout_before_restart:
            print(f"\n检测到 {TraeReviewer._timeout_count} 次超时，正在重启 Ollama 服务...")
            model_name = self.ollama_model
            port = _restart_ollama(model_name)
            self.ollama_url = f"http://localhost:{port}/api/generate"
            TraeReviewer._timeout_count = 0
            print("Ollama 已重启，继续分析...\n")

    def review_stock(self, code: str, day_chart: Path, prompt: str) -> dict:
        """
        调用本地 Ollama 模型，对单支股票进行图表分析，返回解析后的 JSON 结果。
        使用完整的 prompt.md 内容。
        """
        import base64
        
        # 读取图片数据并转换为 base64
        with open(day_chart, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        
        # 构建请求数据 - 将股票代码和完整 prompt 合并
        full_prompt = f"股票代码：{code}\n\n{prompt}\n\n请直接输出 JSON，不要任何其他内容。"
        
        payload = {
            "model": self.ollama_model,
            "prompt": full_prompt,
            "images": [base64_image],
            "stream": False
        }
        
        # 发送请求到 Ollama API
        try:
            print("正在分析...", end="", flush=True)
            response = requests.post(self.ollama_url, json=payload, timeout=600)
            response.raise_for_status()
            print("完成")
            
            # 解析响应
            result = response.json()
            if "response" in result:
                # 提取 JSON 响应
                import re
                json_text = result["response"]
                # 尝试从响应中提取 JSON
                code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", json_text)
                if code_block:
                    json_text = code_block.group(1)
                start = json_text.find("{")
                end = json_text.rfind("}") + 1
                if start != -1 and end != 0:
                    result = json.loads(json_text[start:end])
                    result["code"] = code
                    return result
                else:
                    # 如果解析失败，打印原始响应以便调试
                    print(f"\nJSON 解析失败，原始响应：{json_text[:500]}...")
                    return {
                        "total_score": 0,
                        "verdict": "出错",
                        "signal_type": "其他",
                        "comment": f"Ollama 分析：出错 - 无法解析 JSON",
                        "code": code
                    }
            else:
                print(f"\n响应格式错误：{result}")
                return {
                    "total_score": 0.0,
                    "verdict": "出错",
                    "signal_type": "出错",
                    "comment": f"Ollama 分析：出错 - 响应格式错误",
                    "code": code
                }
        except requests.exceptions.Timeout as e:
            print(f"\n分析失败：请求超时 - {e}")
            self._restart_ollama_if_needed()
        except Exception as e:
            print(f"\n分析失败：{e}")
            if "timeout" in str(e).lower():
                self._restart_ollama_if_needed()
        
        # 如果失败，返回默认结果
        return {
            "total_score": 0.0,
            "verdict": "出错",
            "signal_type": "其他",
            "comment": f"Ollama 分析出错",
            "code": code
        }


def main():
    parser = argparse.ArgumentParser(description="TRADE 内置 AI 图表分析")
    parser.add_argument("--config", type=Path, help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    reviewer = TraeReviewer(config)
    reviewer.run()


if __name__ == "__main__":
    main()
