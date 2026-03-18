"""
总结Agent模块
负责对所有Agent的响应进行总结并给出最终回答
"""

from typing import Optional
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.deepseek import DeepSeek

class SummaryAgent:
    """总结Agent，负责整合各Agent的回答并生成最终回复"""
    def __init__(self, llm):
        self.name = "总结Agent"
        self.llm = llm
        self.chat_engine = None
        self.initialize()
    
    def initialize(self):
        """初始化总结Agent"""
        self.chat_engine = SimpleChatEngine.from_defaults(
            llm=self.llm
        )
        print(f"{self.name} 初始化完成")
    
    async def generate_summary(self, query: str, agent_type: str, response: str) -> str:
        """整合Agent的回答并生成最终回复"""
        try:
            # 根据Agent类型生成不同的总结提示
            if agent_type == "knowledge":
                prompt = f"""
                你是一个专业的知识总结助手。请将以下与多元分析学相关的回答进行整理，使其更加清晰、专业：
                用户问题：{query}
                Agent回答：{response}
                
                要求：
                1. 保持专业性和准确性
                2. 确保逻辑清晰，层次分明
                3. 保留关键信息
                """
            else:
                prompt = f"""
                你是一个友好的对话助手。请将以下回答进行整理，使其更加自然、流畅：
                用户问题：{query}
                Agent回答：{response}
                
                要求：
                1. 保持友好、自然的语气
                2. 确保回答清晰易懂
                3. 保留原始信息
                """
            
            # 生成总结后的回答
            summary = await self.chat_engine.achat(prompt)
            return str(summary)
        except Exception as e:
            print(f"{self.name} 处理出错: {str(e)}")
            # 如果总结Agent出错，直接返回原始回答
            return response