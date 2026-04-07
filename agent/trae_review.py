# ────────────────────────────────────────────────
# 内置 Prompt（总体版）
# ────────────────────────────────────────────────
DEFAULT_PROMPT = """
专业波段交易员根据日线图判断波段爆发潜力。只根据图中可见信息分析。用中文输出。

## 评分标准

### 1️⃣ 趋势结构
5 分：均线刚多头，短期上拐上穿长期，间距合理
4 分：均线多头排列，沿均线上行
3 分：均线偏多但不流畅
2 分：均线交叉趋势不清
1 分：空头排列向下

### 2️⃣ 价格位置
5 分：中低位刚突破，上方空间大
4 分：中位突破向前高
3 分：临近前高
2 分：接近历史高位
1 分：高位过热远离均线

### 3️⃣ 量价行为
5 分：上涨放量首阴缩量，最大量在涨段
4 分：上涨放量回调缩量
3 分：量价中性
2 分：上涨无放量回调不缩
1 分：放量大阴在跌 K

### 4️⃣ 前期异动
5 分：异常放量阳突破，涨幅<50%
4 分：明显放量阳，涨幅<50%
3 分：放量不突出，涨幅<50%
2 分：普通上涨或涨幅 50-100%
1 分：涨幅>100% 或放量大阴出货

## 权重
trend:0.20 | pos:0.20 | vol:0.30 | abnormal:0.30
计算公式：total_score = trend*0.20 + pos*0.20 + vol*0.30 + abnormal*0.30

## 判定
PASS≥4.0 | WATCH 3.2-4.0 | FAIL<3.2 | vol=1→必 FAIL

## 信号
主升启动 | 跌后反弹 | 出货风险

## JSON 输出
{"trend_reasoning":"string","position_reasoning":"string","volume_reasoning":"string","abnormal_move_reasoning":"string","signal_reasoning":"string","scores":{"trend_structure":1-5,"price_position":1-5,"volume_behavior":1-5,"previous_abnormal_move":1-5},"total_score":1.0-5.0,"signal_type":"trend_start/rebound/distribution_risk","verdict":"BUY/WATCH/FAIL","comment":"一句中文点评"}
"""


# ────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from base_reviewer import BaseReviewer

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "gemini_review.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # 路径参数
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # 模型参数
    "request_delay": 5,
    "skip_existing": False,
    "suggest_min_score": 4.0,
    # llama-cpu 配置
    "llama_cli": "llama-vulkan/llama-mtmd-cli.exe",
    "model_path": "models/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
    "mmproj_path": "models/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
}


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _ROOT) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    cfg = DEFAULT_CONFIG.copy()

    if config_path.exists():
        import yaml
        with open(config_path, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    else:
        print(f"[WARN] 未找到配置文件 {config_path}，使用默认配置。")

    # 路径解析
    for key in ["candidates", "kline_dir", "output_dir", "prompt_path", "llama_cli", "model_path", "mmproj_path"]:
        if key in cfg:
            cfg[key] = _resolve_cfg_path(cfg[key])

    return cfg


class TraeReviewer(BaseReviewer):
    def __init__(self, config):
        # 使用内置 prompt，不从文件加载
        self.config = config
        self.prompt = DEFAULT_PROMPT.strip()
        self.kline_dir = Path(config["kline_dir"])
        self.output_dir = Path(config["output_dir"])
        self.llama_cli = Path(config["llama_cli"])
        self.model_path = Path(config["model_path"])
        self.mmproj_path = Path(config["mmproj_path"])
        
        # 验证模型文件是否存在
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")
        
        # 验证 mmproj 文件是否存在
        if not self.mmproj_path.exists():
            raise FileNotFoundError(f"mmproj 文件不存在：{self.mmproj_path}")
        
        # 验证 llama 客户端是否存在
        if not self.llama_cli.exists():
            raise FileNotFoundError(f"llama 客户端不存在：{self.llama_cli}")

    def review_stock(self, code: str, day_chart: Path, prompt: str) -> dict:
        """
        调用 llama-cpu 对单支股票进行图表分析，返回解析后的 JSON 结果。
        """
        import tempfile
        
        # 将图片保存到临时文件（避免命令行参数过长）
        temp_image = None
        try:
            # 创建临时图片文件
            fd, temp_image = tempfile.mkstemp(suffix='.jpg')
            import os
            os.close(fd)
            
            # 复制图片到临时文件
            with open(temp_image, 'wb') as f:
                with open(day_chart, 'rb') as src:
                    f.write(src.read())
            
            # 构建完整 prompt
            full_prompt = f"股票代码：{code}\n\n{prompt}\n\n请直接输出 JSON，不要任何其他内容。"
            
            print("正在分析...", end="", flush=True)
            
            # 调用 llama
            result = self._call_llama(temp_image, full_prompt)
            
            print("完成")
            
            if result:
                result["code"] = code
                return result
            else:
                return self._get_default_result(code)
            
        except Exception as e:
            print(f"\n分析失败：{e}")
            return self._get_default_result(code)
        finally:
            # 清理临时文件
            if temp_image and Path(temp_image).exists():
                try:
                    Path(temp_image).unlink()
                except:
                    pass
    
    def _call_llama(self, image_path: str, prompt: str) -> dict:
        """调用 llama 进行单次分析，返回 JSON 结果"""
        # 构建 llama 命令行参数
        cmd = [
            str(self.llama_cli),
            "-m", str(self.model_path),
            "--mmproj", str(self.mmproj_path),
            "-p", prompt,
            "--image", image_path,
            "--n-predict", "512",  # 每步只需要较短输出
            "--temp", "0.3",  # 降低温度提高稳定性
            "-ngl", "99",
            "--json-schema", "{}",
        ]
        
        try:
            # 执行 llama 命令（使用 UTF-8 编码避免解码错误）
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # 替换无法解码的字符
                timeout=180,  # 3 分钟超时
                cwd=str(_ROOT)
            )
            
            if result.returncode != 0:
                return {}
            
            # 解析输出
            output = result.stdout
            
            # 打印调试信息
            print(f"\n[DEBUG] 原始输出：{output[:200]}...")
            
            import re
            json_text = output
            
            # 尝试从输出中提取 JSON
            code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", json_text)
            if code_block:
                json_text = code_block.group(1)
            
            start = json_text.find("{")
            end = json_text.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(json_text[start:end])
            else:
                return {}
                
        except Exception:
            return {}
    
    def _get_default_result(self, code: str) -> dict:
        """返回默认结果"""
        return {
            "total_score": 0.0,
            "verdict": "出错",
            "signal_type": "其他",
            "comment": f"分析出错",
            "code": code
        }


def main():
    config = load_config()
    reviewer = TraeReviewer(config)
    reviewer.run()


if __name__ == "__main__":
    main()
