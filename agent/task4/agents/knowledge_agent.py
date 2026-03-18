"""
知识检索Agent模块
负责处理与多元分析学相关的专业问题
"""

from typing import Optional
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.indices.base import BaseIndex
from llama_index.llms.deepseek import DeepSeek

class KnowledgeAgent:
    """知识检索Agent，擅长回答与多元分析学相关的专业问题"""
    def __init__(self, llm, index: BaseIndex):
        self.name = "知识检索Agent"
        self.llm = llm
        self.index = index
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
        self.chat_engine = None
        self.initialize()
    
    def initialize(self):
        """初始化知识检索Agent"""
        # 优化检索参数
        self.chat_engine = self.index.as_chat_engine(
            similarity_top_k=6,  # 增加检索的文档数量
            response_mode="refine",  # 使用refine模式获得更完整的回答
            memory=self.memory,
            llm=self.llm,
            verbose=True  # 启用详细日志
        )
        print(f"{self.name} 初始化完成")
    
    async def process_query(self, query: str) -> str:
        """处理与多元分析学相关的专业问题"""
        try:
            print(f"\n{self.name} 接收到查询: {query}")
            
            # 记录查询信息，便于调试
            print(f"{self.name} 正在准备调用RAG聊天引擎...")
            
            # 调用RAG聊天引擎回答问题
            response = await self.chat_engine.achat(query)
            
            # 记录响应信息，便于调试
            print(f"{self.name} 获得RAG响应: {str(response)[:200]}")
            
            # 如果回答不包含所需信息，尝试直接查询向量存储
            if "don't know" in str(response).lower() or len(str(response).strip()) < 10:
                print(f"{self.name} 响应不完整，尝试直接查询向量存储...")
                
                # 直接获取检索结果
                query_engine = self.index.as_query_engine(
                    similarity_top_k=6,
                    response_mode="tree_summarize",
                    llm=self.llm  # 确保使用DeepSeek模型
                )
                direct_response = await query_engine.aquery(query)
                
                print(f"{self.name} 直接查询响应: {str(direct_response)[:200]}")
                return str(direct_response)
            
            return str(response)
        except Exception as e:
            print(f"{self.name} 处理出错: {str(e)}")
            return f"{self.name} 无法回答此问题: {str(e)}"