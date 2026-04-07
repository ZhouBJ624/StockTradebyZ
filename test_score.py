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
1. 趋势结构（均线排列与斜率）：5 分=刚进入多头，4 分=健康上升，3 分=偏多但不理想，2 分=混乱，1 分=空头
2. 价格位置（是否处于突破位或高位）：5 分=中低位刚突破，4 分=中位突破，3 分=接近前高，2 分=接近历史高位，1 分=明显高位
3. 量价行为（上涨放量、回调缩量）：5 分=上涨放量回调明显缩量，4 分=整体健康，3 分=中性，2 分=偏弱，1 分=恶化
4. 历史异动（是否有主力建仓痕迹）：5 分=明确建仓痕迹，4 分=明显放量阳线，3 分=一定放量，2 分=普通上涨，1 分=出货迹象

请输出 JSON 格式，根据图表实际情况给出每个维度的评分（1-5 分）：
{
  "trend_reasoning": "趋势分析理由",
  "position_reasoning": "位置分析理由",
  "volume_reasoning": "量价分析理由",
  "abnormal_move_reasoning": "异动分析理由",
  "signal_reasoning": "信号分析理由",
  "scores": {"trend_structure": 分数，"price_position": 分数，"volume_behavior": 分数，"previous_abnormal_move": 分数},
  "total_score": 总分 (1-5 分，保留 1 位小数),
  "signal_type": "trend_start 或 rebound 或 distribution_risk",
  "verdict": "BUY 或 WATCH 或 FAIL",
  "comment": "一句中文交易员点评"
}

要求：
- 必须根据图表实际情况给出真实评分，不要默认给 3 分
- 如果图表显示强势，可以给 4-5 分；如果显示弱势，可以给 1-2 分
- 直接输出 JSON，不要其他内容。"""
    
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
            print(f"\n响应内容:\n{resp_text}")
        else:
            print(f"完整响应：{result}")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"错误：{e}")
        print(f"耗时：{elapsed:.2f}秒")
else:
    print(f"找不到测试图片：{kline_path}")
