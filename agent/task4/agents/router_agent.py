"""
路由Agent模块
负责判断任务类型并将任务路由到对应的业务Agent
"""

from typing import Optional
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.deepseek import DeepSeek

class RouterAgent:
    """路由Agent，负责分析用户问题并选择合适的业务Agent"""
    def __init__(self, llm):
        self.name = "路由Agent"
        self.llm = llm
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
        self.chat_engine = None
        self.initialize()
    
    def initialize(self):
        """初始化路由Agent"""
        self.chat_engine = SimpleChatEngine.from_defaults(
            llm=self.llm,
            memory=self.memory
        )
        print(f"{self.name} 初始化完成")
    
    async def route_task(self, query: str) -> str:
        """分析用户问题并返回任务类型"""
        try:
            # 使用LLM分析问题类型
            prompt = f"""
            请分析用户的问题，并判断应该由哪个Agent来处理：
            用户问题：{query}
            
            可选Agent类型：
            1. calculate - 数学计算任务，如加减乘除等
            2. knowledge - 与多元分析学相关的所有问题，包括专业内容、书籍信息（作者、出版社等）、概念解释等
            3. dialogue - 日常对话、聊天或一般信息问题
            4. summary - 对某一段话或文本内容进行总结、归纳、概括
            
            请仅返回Agent类型的标识符（calculate、knowledge 或 dialogue），不要添加其他任何内容。
            """
            
            response = await self.chat_engine.achat(prompt)
            agent_type = str(response).strip().lower()
            
            # 验证返回的Agent类型是否有效
            if agent_type not in ["calculate", "knowledge", "dialogue", "summary"]:
                # 默认使用dialogue类型
                return "dialogue"
            
            return agent_type
        except Exception as e:
            print(f"{self.name} 处理出错: {str(e)}")
            return "dialogue"