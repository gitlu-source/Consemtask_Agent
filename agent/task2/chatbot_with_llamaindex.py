import os
from dotenv import load_dotenv
from llama_index.llms.deepseek import DeepSeek  # 修改导入名称为DeepSeek
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine

# 加载环境变量
def load_environment():
    load_dotenv()
    # 从环境变量中获取API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # 验证API Key是否存在
    if not api_key:
        print("错误: 未在.env文件中找到DEEPSEEK_API_KEY。请确保已正确配置.env文件。")
        print("请参考README.md文件中的配置说明。")
        return None
    
    return api_key

# 创建LLM实例
def create_llm(api_key):
    # 使用Deepseek-R1模型
    llm = DeepSeek(  # 修改类名为DeepSeek
        model="deepseek-chat",  # 使用正确的Deepseek-R1模型标识符
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000,
        base_url="https://api.deepseek.com/v1"  # 设置正确的API基础URL
    )
    return llm

# 创建带有记忆功能的聊天引擎
def create_chat_engine(llm):
    # 创建记忆缓冲区，用于存储对话历史
    memory = ChatMemoryBuffer.from_defaults(token_limit=8000)  # 调整token限制，防止请求过大
    
    # 创建系统提示词，要求展示思考过程
    system_prompt = """
    你是一个有思考能力的智能助手，请按照以下格式回应用户:
    
    【思考】
    [这里是你的思考过程，分析用户的问题，解释你的回答思路]
    
    [你的最终回答]
    """
    
    # 创建聊天引擎
    chat_engine = SimpleChatEngine.from_defaults(
        llm=llm,
        memory=memory,
        system_prompt=system_prompt
    )
    
    return chat_engine

# 运行对话程序
def run_chatbot(chat_engine):
    print("\n===== 有记忆的智能对话程序 =====")
    print("输入'退出'结束对话\n")
    
    while True:
        # 获取用户输入
        user_input = input("你: ")
        
        if user_input.lower() in ["退出", "quit", "exit"]:
            print("对话已结束，再见！")
            break
        
        # 显示"正在思考中..."的提示
        print("AI正在思考中...")
        
        # 获取AI响应
        response = chat_engine.chat(user_input)
        
        # 显示AI响应
        print(f"AI:\n{response}")

# 主函数
def main():
    try:
        # 加载环境变量和API Key
        api_key = load_environment()
        
        # 如果API Key不存在，直接返回
        if not api_key:
            return
        
        # 简单验证API Key格式
        if len(api_key) < 10:
            print("警告: API Key格式可能不正确，请检查您的.env文件。")
        
        # 创建LLM实例
        llm = create_llm(api_key)
        
        # 创建聊天引擎
        chat_engine = create_chat_engine(llm)
        
        # 运行对话程序
        run_chatbot(chat_engine)
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        print("\n请检查以下几点:")
        print("1. 您的Deepseek API Key是否正确")
        print("2. 您是否安装了所有必要的依赖包: pip install -r requirements.txt")
        print("3. 您的网络连接是否正常")
        print("4. 您的API Key是否有足够的调用额度")
        print("\n如果问题仍然存在，请尝试使用基础版本: python chatbot.py")

if __name__ == "__main__":
    main()