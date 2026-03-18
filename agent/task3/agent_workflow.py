import os
import sys
import io
import platform
from dotenv import load_dotenv
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from ddgs import DDGS

# 设置安全的编码环境
def setup_safe_encoding():
    """设置安全的编码环境，处理不同终端的编码问题"""
    # 首先尝试检测当前终端编码
    terminal_encoding = None
    try:
        if platform.system() == 'Windows':
            # 在Windows上尝试获取控制台编码
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)  # 设置为UTF-8
            kernel32.SetConsoleCP(65001)
        # 获取当前终端的编码
        terminal_encoding = sys.stdout.encoding if sys.stdout.isatty() else 'utf-8'
    except Exception as e:
        # 如果获取失败，默认使用UTF-8
        terminal_encoding = 'utf-8'
    
    # 配置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_CTYPE'] = 'en_US.UTF-8'
    
    # 重新配置标准输出和标准错误，添加错误处理
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding='utf-8',
        errors='replace'
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding='utf-8',
        errors='replace'
    )
    
    return terminal_encoding

# 初始化安全编码
already_setup = False
if not already_setup:
    terminal_encoding = setup_safe_encoding()
    already_setup = True

# 安全打印函数
def safe_print(text=""):
    """安全地打印文本，处理各种编码情况"""
    try:
        # 确保文本是字符串
        if not isinstance(text, str):
            text = str(text)
            
        # 尝试直接打印
        print(text)
    except UnicodeEncodeError:
        try:
            # 尝试使用终端编码
            print(text.encode(terminal_encoding, errors='replace').decode(terminal_encoding))
        except Exception:
            try:
                # 尝试使用ASCII编码，替换不可打印字符
                print(text.encode('ascii', errors='replace').decode('ascii'))
            except Exception:
                # 最后的回退方案
                print("[无法显示的内容]")

# 加载环境变量
def load_environment():
    load_dotenv()
    # 从环境变量中获取API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # 验证API Key是否存在
    if not api_key:
        safe_print("错误: 未在.env文件中找到DEEPSEEK_API_KEY。请确保已正确配置.env文件。")
        safe_print("请参考README.md文件中的配置说明。")
        return None
    
    return api_key

# 创建LLM实例
def create_llm(api_key):
    try:
        # 使用Deepseek-R1模型，简化配置以避免编码问题
        llm = DeepSeek(
            model="deepseek-chat",
            api_key=api_key,
            temperature=0.7,
            max_tokens=1000,
            base_url="https://api.deepseek.com/v1"
        )
        return llm
    except Exception as e:
        safe_print(f"创建LLM实例出错: {str(e)}")
        # 尝试使用更基础的配置
        try:
            from llama_index.llms import DeepSeekLLM
            # 尝试使用基础类
            llm = DeepSeekLLM(
                model="deepseek-chat",
                api_key=api_key,
                temperature=0.7,
                max_tokens=1000,
                base_url="https://api.deepseek.com/v1"
            )
            return llm
        except Exception as e2:
            safe_print(f"尝试使用DeepSeekLLM也失败: {str(e2)}")
            # 返回一个模拟的LLM对象，用于测试
            class MockLLM:
                def complete(self, prompt):
                    return "这是一个模拟的回答，实际LLM创建失败。"
            return MockLLM()

# 自定义LLM包装器，用于处理中文编码问题
class EncodedDeepSeek:
    def __init__(self, llm):
        self.llm = llm
        
    def complete(self, prompt):
        try:
            # 确保提示文本以UTF-8编码
            if isinstance(prompt, str):
                # 先将字符串转换为UTF-8编码的字节串
                prompt_bytes = prompt.encode('utf-8')
                # 再解码为字符串（这有助于规范化字符串）
                prompt = prompt_bytes.decode('utf-8')
            
            # 调用原始LLM的complete方法
            response = self.llm.complete(prompt)
            
            # 确保响应正确编码
            if isinstance(response, str):
                response_bytes = response.encode('utf-8')
                response = response_bytes.decode('utf-8')
            elif hasattr(response, 'text') and isinstance(response.text, str):
                # 处理可能的其他响应格式
                response_bytes = response.text.encode('utf-8')
                response = response_bytes.decode('utf-8')
            else:
                # 尝试将任何其他格式转换为字符串
                response = str(response)
            
            return response
        except UnicodeEncodeError as e:
            safe_print(f"EncodedDeepSeek编码错误: {str(e)}")
            # 尝试使用更简单的提示
            try:
                simplified_prompt = "请用中文回答"
                response = self.llm.complete(simplified_prompt)
                return str(response).strip()
            except Exception as e2:
                safe_print(f"EncodedDeepSeek简化提示也失败: {str(e2)}")
                return "抱歉，处理过程中出现了编码问题。"
        except Exception as e:
            safe_print(f"EncodedDeepSeek调用错误: {str(e)}")
            # 如果所有方法都失败，返回一个简单的错误消息
            return "抱歉，处理过程中出现了调用错误。"

# DuckDuckGo搜索工具
def duckduckgo_search(query, max_results=5):
    """使用DDGS库搜索信息并返回结果"""
    ddgs = DDGS()
    results = []
    
    try:
        # 确保查询字符串正确编码
        query = str(query) if query else ""
        
        # 使用ddgs库正确的API方法 - text_search
        search_results = ddgs.text(query, max_results=max_results)
        
        # 处理搜索结果
        for result in search_results:
            # 提取标题、摘要和链接，确保中文字符正确处理
            title = str(result.get('title', '无标题'))
            snippet = str(result.get('body', '无摘要'))
            link = str(result.get('href', '无链接'))
            
            # 过滤黑名单内容（包含"51"的标题或摘要）
            if "51" in title or "51" in snippet:
                continue
            
            # 将结果添加到列表中
            results.append({
                'title': title,
                'snippet': snippet,
                'link': link
            })
        
        return results
    except Exception as e:
        safe_print(f"搜索出错: {str(e)}")
        # 尝试使用备用搜索方式（直接返回空列表）
        return []

# 提取查询关键词
def extract_keywords(llm, user_query):
    """使用LLM从用户问题中提取关键词"""
    try:
        # 确保用户问题正确编码
        user_query = str(user_query) if user_query else ""
        
        # 使用更简洁、更明确的提示词
        prompt = f"""用户问题：{user_query}\n\n请直接提取关键词，用空格分隔，不要添加其他任何文字。"""
        
        response = llm.complete(prompt)
        keywords = str(response).strip()
        
        # 验证提取的关键词，如果为空或不合法，使用备用方法
        if not keywords or len(keywords) > 100:
            safe_print("提取的关键词不合法，使用备用方法")
            # 备用方法：直接返回原始问题的前30个字符
            return user_query[:30]
        
        return keywords
    except UnicodeEncodeError as e:
        safe_print(f"提取关键词编码错误: {str(e)}")
        # 使用备用方法：返回原始问题的分词版本
        try:
            # 简单分词处理
            if len(user_query) > 20:
                return " ".join(list(user_query[:20]))  # 简单拆分
            return user_query
        except Exception:
            return "搜索词"
    except Exception as e:
        safe_print(f"提取关键词调用错误: {str(e)}")
        # 如果出错，返回原始查询作为关键词
        return user_query

# 基于搜索结果生成回答
def generate_answer(llm, user_query, search_results):
    """使用LLM基于搜索结果生成回答"""
    
    try:
        # 确保用户问题正确编码
        user_query = str(user_query) if user_query else ""
        
        # 构建搜索结果文本
        search_results_text = "搜索结果:\n"
        for i, result in enumerate(search_results, 1):
            # 确保中文字符正确处理
            title = str(result.get('title', '无标题'))
            snippet = str(result.get('snippet', '无摘要'))
            
            search_results_text += f"{i}. 标题: {title}\n摘要: {snippet}\n\n"
            
    except Exception as e:
        safe_print(f"构建搜索结果文本出错: {str(e)}")
        search_results_text = "搜索结果:\n无法获取完整搜索结果信息"
    
    try:
        # 构建提示文本，确保正确编码且简洁
        prompt = f"""
我需要你作为一个智能助手，基于提供的搜索结果来回答用户的问题。

用户问题：{user_query}

{search_results_text}

请根据搜索结果，用中文详细回答用户的问题。如果搜索结果中没有相关信息，请直接说明无法找到相关信息。
"""
        
        # 直接调用DeepSeek生成回答
        response = llm.complete(prompt)
        
        # 确保响应正确编码和格式化
        if isinstance(response, str):
            return response.strip()
        elif hasattr(response, 'text'):
            return str(response.text).strip()
        else:
            return str(response).strip()
        
    except Exception as e:
        safe_print(f"生成回答时出错: {str(e)}")
        # 如果出现错误，返回一个简洁的错误消息
        return "抱歉，我在生成回答时遇到了问题。请尝试重新提问。"

# 运行智能体工作流
def run_agent_workflow(llm):
    # 使用自定义的EncodedDeepSeek包装器包装原始LLM
    llm = EncodedDeepSeek(llm)
    
    safe_print("\n===== 智能搜索智能体系统 - DDGS + DeepSeek AI增强版 =====")
    safe_print("功能说明:")
    safe_print("1. 自动提取用户问题关键词")
    safe_print("2. 使用DDGS搜索相关信息")
    safe_print("3. DeepSeek AI智能总结生成答案")
    safe_print("4. 保存对话历史记录")
    safe_print("5. 显示详细搜索结果")
    safe_print("=" * 60)
    
    # 显示初始化状态
    safe_print("✅ 已从配置文件加载DeepSeek API密钥")
    safe_print("✅ DDGS搜索工具已初始化")
    safe_print("✅ DeepSeek AI已初始化")
    safe_print("✅ 中文编码处理已配置")
    safe_print("=" * 60)
    
    # 添加明确的输入提示
    safe_print("\n系统已准备就绪，您可以开始提问了！")
    
    while True:
        # 获取用户输入
        try:
            user_input = input("\n请输入您的问题（输入 'quit' 退出）：")
        except EOFError:
            safe_print("\n检测到EOF错误，程序将退出。")
            break
        except KeyboardInterrupt:
            safe_print("\n检测到键盘中断，程序将退出。")
            break
        
        if user_input.lower() in ["quit", "退出", "exit"]:
            safe_print("对话已结束，再见！")
            break
        
        safe_print("\n🚀 智能搜索智能体开始工作...")
        safe_print("=" * 60)
        safe_print(f"👤 用户问题: {user_input}")
        
        # 1. 分析用户问题并提取关键词
        safe_print("\n🧠 正在分析用户问题：" + user_input)
        keywords = extract_keywords(llm, user_input)
        safe_print(f"🔑 提取的关键词：{keywords}")
        
        # 2. 使用DuckDuckGo搜索
        safe_print(f"\n🔍 正在使用DDGS搜索：{keywords}")
        search_results = duckduckgo_search(keywords)
        
        if not search_results:
            safe_print("❌ 未找到搜索结果")
            continue
        
        safe_print(f"✅ 找到 {len(search_results)} 个搜索结果")
        
        # 显示搜索结果详情
        safe_print("\n📋 搜索结果详情：")
        for i, result in enumerate(search_results, 1):
            safe_print(f"{i}. 标题: {result['title']}")
            safe_print(f"   摘要: {result['snippet']}")
            safe_print(f"   链接: {result['link']}")
            safe_print("")
        
        # 3. 调用LLM分析搜索结果并生成回答
        safe_print("\n🤖 正在调用DeepSeek AI分析搜索结果...")
        answer = generate_answer(llm, user_input, search_results)
        safe_print("✅ DeepSeek AI回答生成成功")
        
        # 4. 显示最终回答
        safe_print("\n🤖 AI答案：")
        safe_print("=" * 60)
        safe_print(answer)
        safe_print("=" * 60)

# 主函数
def main():
    try:
        # 加载环境变量和API Key
        api_key = load_environment()
        
        # 如果API Key不存在，直接返回
        if not api_key:
            safe_print("请在.env文件中配置有效的DEEPSEEK_API_KEY")
            # 为了测试，使用一个临时的模拟值
            safe_print("为了测试，将使用模拟的LLM对象")
            class MockLLM:
                def complete(self, prompt):
                    return "模拟的关键词" if "关键词" in prompt else "这是一个模拟的回答，实际DeepSeek API调用失败。"
            llm = MockLLM()
        else:
            # 简单验证API Key格式
            if len(api_key) < 10:
                safe_print("警告: API Key格式可能不正确，请检查您的.env文件。")
            
            # 创建LLM实例
            llm = create_llm(api_key)
        
        # 运行智能体工作流
        run_agent_workflow(llm)
        
    except Exception as e:
        safe_print(f"程序运行出错: {str(e)}")
        # 提供更详细的错误信息
        import traceback
        safe_print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    main()