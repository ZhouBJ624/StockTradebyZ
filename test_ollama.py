import requests
import base64
from pathlib import Path

# 测试 Ollama API
url = "http://localhost:11434/api/generate"

# 测试带图片的请求（模拟 trae_review.py 的用法）
print("测试带图片的 API（模拟 trae_review.py）...")
kline_path = Path("data/kline/2026-03-18/600089_day.jpg")
if not kline_path.exists():
    kline_path = Path("data/kline/2026-03-18/600089_day.png")

if kline_path.exists():
    with open(kline_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    # 简化版 prompt
    user_text = (
        f"股票代码：600089\n\n"
        f"请分析这张日线图，并以 JSON 格式输出以下字段：\n"
        f"- total_score: 总分（1-5 分）\n"
        f"- verdict: 建议（买入/持有/卖出）\n"
        f"- signal_type: 信号类型\n"
        f"- comment: 简短评论\n\n"
        f"直接输出 JSON，不要其他内容。"
    )
    
    payload = {
        "model": "qwen3-vl:4b",
        "prompt": user_text,
        "images": [image_data],
        "stream": False
    }
    
    print(f"发送请求... (prompt 长度：{len(user_text)})")
    try:
        response = requests.post(url, json=payload, timeout=120)
        print(f"状态码：{response.status_code}")
        result = response.json()
        print(f"响应 keys: {result.keys()}")
        if "response" in result:
            resp_text = result["response"]
            print(f"响应长度：{len(resp_text)}")
            print(f"响应前 1000 字符：{resp_text[:1000]}")
        else:
            print(f"完整响应：{result}")
    except Exception as e:
        print(f"错误：{e}")
else:
    print(f"找不到测试图片：{kline_path}")
