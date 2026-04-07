import requests
import base64
import time
from pathlib import Path

url = "http://localhost:11434/api/generate"

kline_path = Path("data/kline/2026-03-18/600089_day.jpg")
if not kline_path.exists():
    kline_path = Path("data/kline/2026-03-18/600089_day.png")

if kline_path.exists():
    with open(kline_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    user_text = """股票代码：600089

你是一名专业波段交易员，请根据这张日线图进行分析。

分析要点：
1. 趋势结构（均线排列与斜率）
2. 价格位置（是否处于突破位或高位）
3. 量价行为（上涨放量、回调缩量）
4. 历史异动（是否有主力建仓痕迹）

请输出 JSON 格式：
{
  "trend_reasoning": "趋势分析",
  "position_reasoning": "位置分析",
  "volume_reasoning": "量价分析",
  "abnormal_move_reasoning": "异动分析",
  "signal_reasoning": "信号分析",
  "scores": {"trend_structure": 3, "price_position": 3, "volume_behavior": 3, "previous_abnormal_move": 3},
  "total_score": 3.0,
  "signal_type": "trend_start",
  "verdict": "WATCH",
  "comment": "一句中文交易员点评"
}

直接输出 JSON，不要其他内容。"""
    
    payload = {
        "model": "qwen3-vl:4b",
        "prompt": user_text,
        "images": [image_data],
        "stream": False
    }
    
    print(f"发送请求... (prompt 长度：{len(user_text)})")
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=300)
        elapsed = time.time() - start_time
        print(f"状态码：{response.status_code}")
        print(f"耗时：{elapsed:.2f}秒")
        result = response.json()
        if "response" in result:
            resp_text = result["response"]
            print(f"响应长度：{len(resp_text)}")
            print(f"\n响应内容:\n{resp_text[:800]}")
        else:
            print(f"完整响应：{result}")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"错误：{e}")
        print(f"耗时：{elapsed:.2f}秒")
else:
    print(f"找不到测试图片：{kline_path}")
