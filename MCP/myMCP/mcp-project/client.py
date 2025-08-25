import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from datetime import datetime
from typing import List, Optional

import aiofiles
from dotenv import load_dotenv
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


class MCPClientGemini:
    def __init__(self) -> None:
        self.exit_stack = AsyncExitStack()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("MODEL", "gemini-1.5-flash")
        if not self.api_key:
            raise ValueError("❌ 未找到 Gemini API Key")
        genai.configure(api_key=self.api_key)
        self.client = genai
        self.session: Optional[ClientSession] = None

    async def connect_to_server(self, server_script_path: str) -> None:
        is_py = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_py or is_js):
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")
        command = "python" if is_py else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        tools = (await self.session.list_tools()).tools
        print("\n已连接到服务器，可用工具:", [t.name for t in tools])

    async def process_query(self, query: str) -> str:
        tools_resp = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_resp.tools
        ]

        # 简化逻辑：优先使用 search_google_news + analyze_sentiment
        if "search_google_news" in [t["function"]["name"] for t in available_tools]:
            news_json = await self.session.call_tool("search_google_news", {"keyword": query})
            news_list = json.loads(news_json.content[0].text)
            news_text = "\n".join([f"{i+1}. {n['title']}\n{n['desc']}\n{n['url']}" for i, n in enumerate(news_list)])
            sentiment_file = await self.session.call_tool("analyze_sentiment", {"text": news_text})
            final_text = f"📄 新闻内容：\n{news_text}\n\n📊 舆情分析报告路径：{sentiment_file.content[0].text}"
        else:
            final_text = "❌ 未找到 search_google_news 工具"

        # 保存记录
        safe_name = re.sub(r'[\\/:*?"<>|]', "", query)[:20]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{ts}.txt"
        output_dir = "./llm_outputs"
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(f"🗣 用户提问：{query}\n\n🤖 模型回复：\n{final_text}\n")
        print(f"📄 对话记录已保存：{path}")
        return final_text

    async def chat_loop(self) -> None:
        print("\n🤖 Gemini-MCP 客户端已启动！输入 'quit' 退出")
        while True:
            try:
                query = input("\n你: ").strip()
                if query.lower() == "quit":
                    break
                answer = await self.process_query(query)
                print(f"\n🤖 AI: {answer}")
            except Exception as e:
                print(f"\n⚠️ 发生错误: {e}")

    async def cleanup(self) -> None:
        await self.exit_stack.aclose()


async def main() -> None:
    server_script_path = "E:/library/SOMEFUN/MCP/myMCP/mcp-project/server.py"
    client = MCPClientGemini()
    try:
        await client.connect_to_server(server_script_path)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
