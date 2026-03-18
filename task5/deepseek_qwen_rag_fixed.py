"""
基于RAG的知识问答智能体
使用DeepSeek API和Qwen embedding模型
修改版：符合参考代码的结构和功能
"""

import os
import sys
import asyncio
import json
import time
import threading
from typing import Optional
from dotenv import load_dotenv
from tqdm import tqdm
from llama_index.llms.deepseek import DeepSeek
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool
from llama_index.core.indices.base import BaseIndex

# 常量定义
PERSIST_DIR = "./storage"
SOURCE_MD = "../多元分析学_markdown.md"

# 全局进度条变量和锁
_progress_bar = None
_progress_lock = threading.Lock()

# 安全打印函数，确保中文正常显示
def safe_print(text):
    """安全打印函数，确保中文正常显示"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 如果出现编码错误，尝试使用系统默认编码
        if hasattr(sys.stdout, 'encoding'):
            print(text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
        else:
            print(text)

# 启动进度条
def start_progress(total=100, desc="进度", unit="%"):
    """启动全局进度条"""
    global _progress_bar
    with _progress_lock:
        if _progress_bar is None:
            _progress_bar = tqdm(total=total, desc=desc, unit=unit)

# 更新进度条
def update_progress(amount=1):
    """更新全局进度条"""
    global _progress_bar
    with _progress_lock:
        if _progress_bar is not None:
            _progress_bar.update(amount)

# 完成进度条
def finish_progress():
    """完成并关闭全局进度条"""
    global _progress_bar
    with _progress_lock:
        if _progress_bar is not None:
            _progress_bar.close()
            _progress_bar = None

# 加载环境变量
def load_environment():
    """加载环境变量和API Key"""
    load_dotenv()
    # 从环境变量中获取API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # 验证API Key是否存在
    if not api_key:
        safe_print("错误: 未在.env文件中找到DEEPSEEK_API_KEY。请确保已正确配置.env文件。")
        return None
    
    return api_key

# 创建LLM实例
def create_llm(api_key):
    """创建DeepSeek LLM实例"""
    # 使用Deepseek模型
    llm = DeepSeek(
        model="deepseek-chat",
        api_key=api_key,
        temperature=0.2,  # 降低温度以获得更准确的回答
        max_tokens=1000,
        base_url="https://api.deepseek.com/v1"
    )
    return llm

# 构建或加载索引
def build_or_load_index(persist_dir: str = PERSIST_DIR, source_md: str = SOURCE_MD, embedding_model=None, force_rebuild=False) -> Optional[BaseIndex]:
    """构建或加载知识库索引"""
    os.makedirs(persist_dir, exist_ok=True)

    # 如果强制重建或没有持久化，则构建新索引
    if force_rebuild or not os.path.exists(os.path.join(persist_dir, "index_store.json")):
        try:
            safe_print("正在构建知识库...")
            docs = SimpleDirectoryReader(input_files=[source_md]).load_data()
            
            # 使用更小的块大小，以确保更精细的检索
            from llama_index.core.node_parser import SentenceSplitter
            node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            nodes = node_parser.get_nodes_from_documents(docs)
            
            index = VectorStoreIndex(
                nodes,
                embed_model=embedding_model,
                show_progress=True
            )

            # 持久化
            index.storage_context.persist(persist_dir=persist_dir)
            safe_print("知识库构建并保存成功！")
            return index
        except Exception as e:
            safe_print(f"构建知识库时出错: {str(e)}")
            return None
    
    # 如果已有持久化，直接加载
    try:
        safe_print(f"正在从磁盘加载知识库: {persist_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = VectorStoreIndex.from_vector_store(
            vector_store=storage_context.vector_store,
            storage_context=storage_context,
            embed_model=embedding_model
        )
        safe_print("知识库加载完成！")
        return index
    except Exception as e:
        safe_print(f"从磁盘加载知识库时出错: {str(e)}")
        safe_print("将尝试重新构建知识库...")
        return build_or_load_index(persist_dir, source_md, embedding_model, force_rebuild=True)

# 创建查询引擎工具
def create_query_engine_tool(index: BaseIndex, llm: Optional[object] = None) -> QueryEngineTool:
    """创建查询引擎工具"""
    query_engine = index.as_query_engine(
        similarity_top_k=8,  # 增加检索的文档数量
        response_mode="tree_summarize",  # 使用树状汇总模式，更适合复杂问题
        llm=llm or Settings.llm,
    )
    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="multivariate_analysis_rag",
        description="针对《多元分析学》教材的知识检索工具，基于RAG返回相关片段综合回答",
    )
    return tool

# 知识问答类
class RAGChat:
    def __init__(self, llm, embedding_model) -> None:
        # LLM 与记忆
        self.llm = llm
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

        # 构建或加载索引
        self.index = build_or_load_index(embedding_model=embedding_model, force_rebuild=True)
        if self.index:
            # 直接创建查询引擎
            self.query_engine = self.index.as_query_engine(
                llm=llm,
                memory=self.memory,
                similarity_top_k=8,  # 增加检索的文档数量
                response_mode="tree_summarize"  # 使用树状汇总模式，更适合复杂问题
            )
        else:
            safe_print("无法构建或加载知识库索引")
            self.query_engine = None

    async def aask(self, user_input: str) -> str:
        """异步询问问题"""
        if not self.query_engine:
            return "抱歉，知识库初始化失败，无法回答问题。"
        
        try:
            # 使用同步查询，但在异步上下文中运行
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, 
                lambda: self.query_engine.query(user_input)
            )
            return str(resp)
        except Exception as e:
            safe_print(f"回答问题时出错: {str(e)}")
            return "抱歉，回答问题时出现错误。"

# 运行对话程序（带上下文记忆功能）
async def run_chatbot(llm, embedding_model):
    """运行带上下文记忆功能的对话程序"""
    safe_print("\n===== 知识问答助手 =====")
    safe_print("输入'退出'或'quit'结束对话")
    
    # 创建RAG聊天实例
    chat = RAGChat(llm, embedding_model)
    
    if not chat.query_engine:
        safe_print("无法初始化RAG聊天实例，程序将退出。")
        return
    
    try:
        while True:
            # 获取用户输入
            user_input = input("\n您的问题: ")
            
            # 检查是否退出
            if user_input.lower() in ['退出', 'quit', 'exit']:
                safe_print("感谢使用知识问答助手，再见！")
                break
            
            # 处理用户输入
            safe_print(f"处理问题: {user_input}")
            
            # 使用异步方法获取回答
            response = await chat.aask(user_input)
            
            # 显示AI响应
            safe_print(f"\nAI回答：\n{response}")
    except EOFError:
        safe_print("\n检测到EOF错误，程序将退出。")
    except KeyboardInterrupt:
        safe_print("\n程序被用户中断，退出对话。")

# 测试嵌入模型连接
def test_embedding_model(embedding_model):
    """测试嵌入模型是否能正常工作"""
    try:
        safe_print("正在测试嵌入模型连接...")
        
        # 准备测试文本
        test_text = "这是一个测试文本，用于验证嵌入模型是否正常工作。"
        
        # 尝试获取嵌入向量
        embedding = embedding_model.get_text_embedding(test_text)
        
        # 验证嵌入向量
        if embedding and isinstance(embedding, list) and len(embedding) > 0:
            safe_print(f"✅ 嵌入模型连接成功！")
            safe_print(f"   嵌入向量维度: {len(embedding)}")
            return True
        else:
            safe_print("❌ 嵌入模型返回了无效的嵌入向量")
            return False
    except Exception as e:
        safe_print(f"❌ 测试嵌入模型连接时出错: {str(e)}")
        return False

# 创建嵌入模型
def create_embedding_model():
    """创建Ollama嵌入模型实例"""
    safe_print("\n===== 创建嵌入模型 ======")
    safe_print("使用Ollama mxbai-embed-large模型")
    
    try:
        # 使用Ollama模型
        embedding_model = OllamaEmbedding(
            model_name="mxbai-embed-large",
            base_url="http://localhost:11434",  # Ollama默认API地址
            request_timeout=120.0,  # 增加超时时间以处理较大的请求
            dimensions=1024  # 设置嵌入向量维度
        )
        
        return embedding_model
    except Exception as e:
        safe_print(f"创建嵌入模型时出错: {str(e)}")
        raise

# 主函数
async def main():
    """主函数"""
    try:
        start_progress(total=100, desc="程序启动中", unit="%")
        safe_print("程序启动中...")
        
        # 加载环境变量和API Key
        update_progress(20)
        time.sleep(0.1)
        api_key = load_environment()
        
        # 如果API Key不存在，直接返回
        if not api_key:
            safe_print("未找到有效的API Key，程序将退出。")
            finish_progress()
            return
        
        # 简单验证API Key格式
        if len(api_key) < 10:
            safe_print("警告: API Key格式可能不正确，请检查您的.env文件。")
        
        # 创建LLM实例
        update_progress(20)
        time.sleep(0.1)
        llm = create_llm(api_key)
        
        # 设置全局参数
        update_progress(10)
        time.sleep(0.1)
        Settings.llm = llm
        Settings.text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=100)
        
        # 创建Ollama Embedding模型
        update_progress(20)
        time.sleep(0.1)
        embedding_model = create_embedding_model()
        
        # 设置全局嵌入模型
        update_progress(10)
        time.sleep(0.1)
        Settings.embed_model = embedding_model
        
        # 测试Ollama连接
        update_progress(20)
        time.sleep(0.1)
        finish_progress()
        
        connection_success = test_embedding_model(embedding_model)
        
        if connection_success:
            # 运行对话程序
            await run_chatbot(llm, embedding_model)
        else:
            safe_print("无法连接到Ollama服务，程序将退出。")
            # 提供故障排除建议
            safe_print("\n故障排除建议：")
            safe_print("1. Ollama服务是否正在运行")
            safe_print("2. Qwen3-Embedding-4B模型是否已正确下载")
            safe_print("3. 您是否安装了所有必要的依赖包")
            safe_print("4. 您的网络连接是否正常")
    except Exception as e:
        try:
            finish_progress()
        except:
            pass
        
        safe_print(f"程序运行时出错: {str(e)}")
        import traceback
        safe_print(f"详细错误信息: {traceback.format_exc()}")
    finally:
        # 确保进度条完成
        try:
            finish_progress()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())