# DeepSeek + Qwen RAG智能体

基于LlamaIndex框架、DeepSeek API和Qwen嵌入模型的知识问答智能体。

## 项目简介

这个项目实现了一个基于检索增强生成(RAG)的智能问答系统，使用DeepSeek大语言模型进行文本生成，Qwen嵌入模型进行向量检索，能够从本地文档中构建知识库并基于该知识库回答用户问题。

## 主要功能

- 使用DeepSeek API进行文本生成
- 集成Qwen3-Embedding-4B模型进行向量检索
- 支持从本地文件构建知识库
- 实现上下文记忆对话功能
- 支持知识库的保存和加载
- 自动检测GPU加速
- 优化的文本分块策略

## 环境配置

1. 安装依赖包：
```bash
pip install -r requirements.txt
```

2. 创建`.env`文件，添加DeepSeek API Key：
```
DEEPSEEK_API_KEY=your_api_key_here
```

3. 确保Ollama服务正在运行，并已下载Qwen3-Embedding模型：
```bash
ollama pull dengcao/Qwen3-Embedding-4B:Q8_0
```

4. 要启用GPU加速，请设置环境变量并重启Ollama：
```bash
# Windows
set OLLAMA_CUDA=1
ollama serve

# Linux/Mac
OLLAMA_CUDA=1 ollama serve
```

## 使用方法

1. 确保您有要用于构建知识库的文档文件。默认情况下，程序会尝试加载名为`多元分析学_markdown.md`的文件。

2. 运行主程序：
```bash
python deepseek_qwen_rag.py
```

3. 程序启动后，您可以输入问题进行问答交互。输入`退出`或`quit`结束对话。

## 知识库管理

- 首次运行时，程序会构建知识库并保存在`./storage`目录中
- 后续运行将直接加载已保存的知识库，提高启动速度
- 程序会优先尝试从预计算的向量化结果文件加载（如果存在）

## 故障排除

如果遇到问题，请检查以下几点：

1. 您的Deepseek API Key是否正确
2. Ollama服务是否正在运行
3. Qwen3-Embedding-4B模型是否已正确下载
4. 您是否安装了所有必要的依赖包
5. 您的网络连接是否正常

## 优化建议

- 对于大型文档，建议先进行适当的预处理以提高检索效率
- 使用GPU加速可以显著提高向量化速度
- 可以根据需要调整分块大小和检索参数以获得更好的回答质量