"""
多智能体系统主程序
协调路由Agent、知识检索Agent、对话Agent和总结Agent的工作流程
"""

import os
import asyncio
import time
from dotenv import load_dotenv
from tqdm import tqdm

from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# 导入所有Agent
from agents.router_agent import RouterAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.dialogue_agent import DialogueAgent
from agents.summary_agent import SummaryAgent
from agents.calculate_agent import CalculateAgent

class MultiAgentSystem:
    """多智能体系统，协调各个Agent的工作"""
    def __init__(self):
        self.llm = None
        self.embed_model = None
        self.index = None
        self.router_agent = None
        self.knowledge_agent = None
        self.dialogue_agent = None
        self.summary_agent = None
        self.calculate_agent = None
    
    def load_env(self):
        """加载环境变量"""
        try:
            load_dotenv()
            if not os.environ.get('DEEPSEEK_API_KEY'):
                print("警告：未找到DEEPSEEK_API_KEY环境变量")
            return True
        except Exception as e:
            print(f"加载环境变量失败: {str(e)}")
            return False
    
    def init_models(self):
        """初始化LLM和嵌入模型"""
        try:
            print("正在初始化模型...")
            # 初始化DeepSeek LLM
            self.llm = DeepSeek(
                model="deepseek-chat",
                api_key=os.environ.get('DEEPSEEK_API_KEY')
            )
            
            # 初始化Ollama嵌入模型
            # 使用与独立rag_llamaindex项目相同的嵌入模型，该模型已被验证可以正常工作
            self.embed_model = OllamaEmbedding(
                model_name="dengcao/Qwen3-Embedding-4B:Q8_0",
                base_url="http://localhost:11434",
                request_timeout=120.0,
                dimensions=5000
            )
            
            print("模型初始化完成")
            return True
        except Exception as e:
            print(f"初始化模型失败: {str(e)}")
            return False
    
    def load_knowledge_base(self):
        """加载或创建知识库"""
        try:
            print("正在加载知识库...")
            
            # 设置Chroma向量存储
            persist_dir = "./chroma_db"
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            chroma_collection = chroma_client.get_or_create_collection("multiagent_knowledge")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 检查是否已有知识库
            if len(chroma_collection.get()['ids']) > 0:
                print("发现已有知识库，正在加载...")
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    embed_model=self.embed_model
                )
                print("知识库加载完成")
            else:
                print("未发现知识库，正在创建...")
                # 读取文档（假设文档在./documents目录下）
                documents_dir = "./documents"
                if os.path.exists(documents_dir) and os.listdir(documents_dir):
                    # 使用tqdm显示进度
                    with tqdm(total=100, desc="正在解析文档") as pbar:
                        documents = SimpleDirectoryReader(documents_dir).load_data()
                        pbar.update(40)
                        time.sleep(1)  # 为了显示进度
                        
                        # 创建索引
                        self.index = VectorStoreIndex.from_documents(
                            documents,
                            storage_context=storage_context,
                            embed_model=self.embed_model
                        )
                        pbar.update(60)
                    print("知识库创建完成")
                else:
                    # 创建空知识库
                    self.index = VectorStoreIndex.from_vector_store(
                        vector_store,
                        embed_model=self.embed_model
                    )
                    print("警告：创建了空知识库，建议在./documents目录下添加文档")
            
            return True
        except Exception as e:
            print(f"加载知识库失败: {str(e)}")
            return False
    
    def init_agents(self):
        """初始化所有Agent"""
        try:
            print("正在初始化所有Agent...")
            
            # 初始化路由Agent
            self.router_agent = RouterAgent(self.llm)
            
            # 初始化知识检索Agent
            self.knowledge_agent = KnowledgeAgent(self.llm, self.index)
            
            # 初始化对话Agent
            self.dialogue_agent = DialogueAgent(self.llm)
            
            # 初始化总结Agent
            self.summary_agent = SummaryAgent(self.llm)
            
            # 初始化计算Agent
            self.calculate_agent = CalculateAgent(self.llm)
            
            print("所有Agent初始化完成")
            return True
        except Exception as e:
            print(f"初始化Agent失败: {str(e)}")
            return False
    
    async def run_pipeline(self, query: str) -> str:
        """运行多智能体处理流程"""
        try:
            print(f"\n用户问题: {query}")
            
            # 1. 路由Agent判断任务类型
            print("1. 路由Agent正在分析问题类型...")
            agent_type = await self.router_agent.route_task(query)
            print(f"   任务类型: {agent_type}")
            
            # 2. 对应业务Agent执行任务
            print(f"2. {self._get_agent_name(agent_type)}Agent正在处理问题...")
            if agent_type == "calculate":
                # 计算任务是同步的
                response = self.calculate_agent.process_query(query)
            elif agent_type == "knowledge":
                response = await self.knowledge_agent.process_query(query)
            elif agent_type == "summary":
                # 直接使用总结Agent处理，这里复用generate_summary方法但简化流程
                # 提取需要总结的内容（去除指令部分）
                content_to_summarize = query.replace("总结这段话：", "").replace("请总结：", "").strip()
                prompt = f"请总结以下内容：{content_to_summarize}\n要求：简明扼要，保留核心信息。"
                response = await self.summary_agent.chat_engine.achat(prompt)
                response = str(response)
            else:
                response = await self.dialogue_agent.process_query(query)
            print(f"   Agent响应获取完成")
            
            # 3. 总结Agent生成最终回答
            print("3. 总结Agent正在生成最终回答...")
            final_answer = await self.summary_agent.generate_summary(query, agent_type, response)
            print(f"   最终回答生成完成")
            
            return final_answer
        except Exception as e:
            print(f"处理流程出错: {str(e)}")
            return f"抱歉，处理您的问题时出错: {str(e)}"
    
    async def start_interactive_mode(self):
        """启动交互式对话模式"""
        print("\n=================================================================")
        print("多智能体问答系统已启动")
        print("您可以输入问题进行咨询，输入'退出'或'quit'结束对话")
        print("=================================================================")
        
        while True:
            try:
                user_query = input("\n您的问题: ")
                if user_query.lower() in ["退出", "quit"]:
                    print("感谢使用多智能体问答系统，再见！")
                    break
                
                # 运行处理流程
                final_answer = await self.run_pipeline(user_query)
                
                # 显示最终回答
                print("\n最终回答:\n")
                print(final_answer)
                print("\n" + "=" * 70)
            except KeyboardInterrupt:
                print("\n感谢使用多智能体问答系统，再见！")
                break
            except Exception as e:
                print(f"交互过程中出错: {str(e)}")
                print("请尝试重新输入问题。")
    
    def _get_agent_name(self, agent_type: str) -> str:
        """根据agent_type返回对应的中文名称"""
        name_map = {
            "calculate": "计算",
            "knowledge": "知识检索",
            "dialogue": "对话",
            "summary": "总结"
        }
        return name_map.get(agent_type, "未知")
    
    async def start(self):
        """启动多智能体系统"""
        print("多智能体系统启动中...")
        
        # 1. 加载环境变量
        if not self.load_env():
            print("无法加载环境变量，系统启动失败")
            return False
        
        # 2. 初始化模型
        if not self.init_models():
            print("无法初始化模型，系统启动失败")
            return False
        
        # 3. 加载知识库
        if not self.load_knowledge_base():
            print("知识库加载失败，但系统将继续运行")
        
        # 4. 初始化Agent
        if not self.init_agents():
            print("无法初始化Agent，系统启动失败")
            return False
        
        # 5. 启动交互式模式
        await self.start_interactive_mode()
        
        return True

async def main():
    """主函数"""
    system = MultiAgentSystem()
    await system.start()

if __name__ == "__main__":
    # 检查documents目录是否存在，不存在则创建
    if not os.path.exists("./documents"):
        os.makedirs("./documents")
        print("📁 创建了documents目录，请将需要分析的文档放入该目录")
    
    # 运行主程序
    asyncio.run(main())