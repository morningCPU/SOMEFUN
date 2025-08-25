import os
import json
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
import google.generativeai as genai
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("NewsServerGemini")

# ---------------- 配置 Gemini ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ 请在 .env 文件中设置 GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

# ---------------- 配置 Serper ----------------
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/news"


# ---------------- 工具1：搜索 Google 新闻 ----------------
@mcp.tool()
async def search_google_news(keyword: str) -> str:
    if not SERPER_API_KEY:
        return "❌ 未配置 SERPER_API_KEY"

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": keyword}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(SERPER_URL, headers=headers, json=payload)
            data: Dict[str, Any] = resp.json()
    except Exception as e:
        return f"❌ Serper 请求异常：{e}"

    if "news" not in data or not data["news"]:
        return "🔍 未获取到任何新闻结果"

    articles = [
        {"title": item.get("title", ""), "desc": item.get("snippet", ""), "url": item.get("link", "")}
        for item in data["news"][:5]
    ]
    # 返回 JSON 字符串，客户端可解析
    return json.dumps(articles, ensure_ascii=False)


# ---------------- 工具2：情感分析 ----------------
@mcp.tool()
async def analyze_sentiment(text: str, filename: str = "") -> str:
    """
    使用 Gemini 对新闻文本进行情感分析，并生成 Markdown 报告
    """
    prompt = (
        "请对以下新闻内容进行情绪倾向分析，并说明原因；"
        "语言请使用中文，结构清晰简洁：\n\n"
        f"{text}"
    )
    response = await asyncio.to_thread(gemini_model.generate_content, prompt)
    result = response.text.strip()

    # 生成报告
    markdown = f"""# 舆情分析报告
**分析时间：** {datetime.now():%Y-%m-%d %H:%M:%S}
---
## 📥 原始文本
{text}
---
## 📊 分析结果
{result}
"""
    output_dir = "./sentiment_reports"
    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = f"sentiment_{datetime.now():%Y%m%d_%H%M%S}.md"
    file_path = os.path.abspath(os.path.join(output_dir, filename))

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return file_path


# ---------------- 工具3：发送带附件的邮件 ----------------
@mcp.tool()
async def send_email_with_attachment(
    to: str,
    subject: str,
    body: str,
    attachment_path: str,
) -> str:
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 465))
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")

    if not all([smtp_server, sender, password]):
        return "❌ 邮箱配置缺失"
    if not os.path.exists(attachment_path):
        return f"❌ 附件不存在: {attachment_path}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.set_content(body)

    with open(attachment_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="octet-stream",
            filename=os.path.basename(attachment_path),
        )

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.send_message(msg)
        return f"✅ 邮件已发送至 {to}"
    except Exception as e:
        return f"❌ 发送失败：{e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
