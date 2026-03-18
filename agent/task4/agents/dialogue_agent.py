"""
对话Agent模块
负责处理日常对话和一般信息问题
"""

from typing import Optional
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.deepseek import DeepSeek

class DialogueAgent:
    """对话Agent，擅长进行日常对话和提供一般信息"""
    def __init__(self, llm):
        self.name = "对话Agent"
        self.llm = llm
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
        self.chat_engine = None
        self.initialize()
    
    def initialize(self):
        """初始化对话Agent"""
        self.chat_engine = SimpleChatEngine.from_defaults(
            llm=self.llm,
            memory=self.memory
        )
        print(f"{self.name} 初始化完成")
    
    async def process_query(self, query: str) -> str:
        """处理日常对话和一般信息问题"""
        try:
            # 使用简单聊天引擎回答问题
            response = await self.chat_engine.achat(query)
            return str(response)
        except Exception as e:
            print(f"{self.name} 处理出错: {str(e)}")
            return f"{self.name} 无法回答此问题: {str(e)}"