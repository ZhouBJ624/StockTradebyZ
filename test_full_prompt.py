import requests
import base64
import time
from pathlib import Path

url = "http://localhost:11434/api/generate"

# 读取完整的 prompt.md
prompt_path = Path("agent/prompt.md")
with open(prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()

kline_path = Path("data/kline/2026-03-18/600089_day.jpg")
if not kline_path.exists():
    kline_path = Path("data/kline/2026-03-18/600089_day.png")

if kline_path.exists():
    with open(kline_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    user_text = (
        f"股票代码：600089\n\n"
        f"以下是该股票的日线图，请严格按照 prompt 中的分析顺序和评分标准进行分析，"
        f"先完成推理，再给出评分，最后输出 JSON。"
    )
    
    payload = {
        "model": "qwen3-vl:8b",
        "prompt": f"{system_prompt}\n\n{user_text}",
        "images": [image_data],
        "stream": False
    }
    
    print(f"发送请求... (system_prompt 长度：{len(system_prompt)}, user_text 长度：{len(user_text)})")
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
            print(f"\n响应内容:\n{resp_text[:1500]}")
        else:
            print(f"完整响应：{result}")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"错误：{e}")
        print(f"耗时：{elapsed:.2f}秒")
else:
    print(f"找不到测试图片：{kline_path}")
