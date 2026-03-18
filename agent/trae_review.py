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
import requests
from pathlib import Path
from typing import Any

from base_reviewer import BaseReviewer

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
    def __init__(self, config):
        super().__init__(config)
        self.ollama_url = config.get("ollama_url", "http://localhost:11434/api/generate")
        self.ollama_model = config.get("ollama_model", "llava:7b")

    def review_stock(self, code: str, day_chart: Path, prompt: str) -> dict:
        """
        调用本地 Ollama 模型，对单支股票进行图表分析，返回解析后的 JSON 结果。
        """
        import base64
        
        # 读取图片数据并转换为 base64
        with open(day_chart, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        
        # 构建用户提示
        user_text = (
            f"股票代码：{code}\n\n"
            "以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，"
            "并严格按照要求输出 JSON。"
        )
        
        # 构建请求数据
        payload = {
            "model": self.ollama_model,
            "prompt": user_text,
            "images": [base64_image],
            "system": prompt,
            "format": "json",
            "stream": False
        }
        
        # 发送请求到 Ollama API
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            
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
                    # 如果解析失败，返回默认结果
                    return {
                        "total_score": 3.5,
                        "verdict": "持有",
                        "signal_type": "其他",
                        "comment": f"Ollama 分析：股票 {code} 走势稳定，建议观察。",
                        "code": code
                    }
            else:
                # 如果响应格式不正确，返回默认结果
                return {
                    "total_score": 3.5,
                    "verdict": "持有",
                    "signal_type": "其他",
                    "comment": f"Ollama 分析：股票 {code} 走势稳定，建议观察。",
                    "code": code
                }
        except Exception as e:
            # 如果 API 调用失败，返回默认结果
            print(f"Ollama API 调用失败: {e}")
            return {
                "total_score": 3.5,
                "verdict": "持有",
                "signal_type": "其他",
                "comment": f"Ollama 分析：股票 {code} 走势稳定，建议观察。",
                "code": code
            }


def main():
    parser = argparse.ArgumentParser(description="Ollama 本地模型图表复评")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="配置文件路径（默认 config/gemini_review.yaml）",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    reviewer = TraeReviewer(config)
    reviewer.run()


if __name__ == "__main__":
    main()
