# 智能体方向任务提交文档

## 任务概述

本文档总结了智能体方向的六个任务（task1-task6）的完成情况、运行方式和技术要点。

## 任务详情

### Task1: 智能体工作流基础概念

**文件结构:**
- README.md: 智能体工作流基础概念文档
- requirements.txt: 依赖包列表

**主要内容:**
- 智能体定义、类型、多智能体系统
- 工作流架构、通信机制、协作模式等理论知识
- 与后续任务（task2-task5）的关联说明

**依赖包:**
- requests
- python-dotenv
- duckduckgo-search
- llama-index系列包（llama-index、llama-index-llms-deepseek、llama-index-embeddings-huggingface等）
- sentence-transformers

### Task2: 基于DeepSeek的聊天机器人

**文件结构:**
- chatbot_with_llamaindex.py: 主程序文件
- README.md: 任务说明文档（已创建）
- requirements.txt: 依赖包列表（已创建）

**主要功能:**
- 基于DeepSeek API的聊天机器人实现
- 使用LlamaIndex框架构建对话系统
- 包含记忆缓冲区和系统提示词模板

**运行方式:**
```bash
cd task2
python chatbot_with_llamaindex.py
```

**依赖包:**
- llama-index
- llama-index-llms-deepseek
- python-dotenv

**环境配置:**
- 需要配置.env文件，包含DEEPSEEK_API_KEY
- 确保网络连接正常，能够访问DeepSeek API

### Task3: 带DuckDuckGo搜索的智能代理

**文件结构:**
- agent_workflow.py: 主程序文件
- README.md: 任务说明文档
- requirements.txt: 依赖包列表（已创建）

**主要功能:**
- 实现带DuckDuckGo搜索的智能代理功能
- 工作流程：问题分析→关键词提取→DuckDuckGo搜索→答案生成
- 包含终端编码处理功能

**运行方式:**
```bash
cd task3
python agent_workflow.py
```

**依赖包:**
- llama-index
- llama-index-llms-deepseek
- python-dotenv
- duckduckgo-search

**环境配置:**
- 需要配置.env文件，包含DEEPSEEK_API_KEY
- 确保网络连接正常，能够访问DeepSeek API和DuckDuckGo搜索

### Task4: 多智能体系统实现

**文件结构:**
- MultiAgent/: 多智能体系统目录
  - requirements.txt: 依赖包列表
  - README_separate_agents.md: 分离式智能体说明文档
- README.md: 任务说明文档

**主要功能:**
- 实现多智能体系统，包含主控制程序和多个智能体模块
- 包含5种智能体类型：计算、搜索、分析、生成、协调
- 实现智能体间的通信机制和协作模式

**系统架构:**
- 主控制程序
- 智能体模块
- 向量数据库
- 文档存储

**依赖包:**
- llama-index系列包（llama-index==0.9.48、llama-index-llms-deepseek==0.1.2等）
- python-dotenv==1.0.0
- pytorch、numpy、pandas
- requests、sentence-transformers、langchain
- tqdm==4.66.4

### Task5: 带有RAG知识库的Agent实现

**文件结构:**
- rag_llamaindex/: RAG实现目录
  - deepseek_qwen_rag_fixed.py: 主程序文件
  - vectorize_markdown_file.py: 文档处理工具
  - storage/: 向量存储目录
  - requirements.txt: 依赖包列表
- README.md: 任务说明文档

**主要功能:**
- 实现RAG（检索增强生成）技术的智能体
- 支持文档预处理、向量化、查询处理和增强生成
- 包含向量存储、文档处理和模板系统等模块

**工作流程:**
1. 文档预处理
2. 向量化
3. 查询处理
4. 增强生成

**运行方式:**
```bash
# 先执行文档向量化
python vectorize_markdown_file.py

# 再运行RAG智能体
cd rag_llamaindex
python deepseek_qwen_rag_fixed.py
```

**依赖包:**
- llama-index系列包（llama-index==0.9.48、llama-index-llms-deepseek==0.1.2等）
- python-dotenv==1.0.0
- pytorch、numpy、pandas
- requests、sentence-transformers、langchain
- tqdm==4.66.4

**环境配置:**
- 需要配置.env文件，包含DEEPSEEK_API_KEY
- 确保网络连接正常，能够访问DeepSeek API

### Task6: Qwen3-14B模型脚本集合

**文件结构:**
- README.md: 任务说明文档
- requirements.txt: 依赖包列表

**主要功能:**
- 提供Qwen3-14B模型的四个核心脚本：
  - interactive_chat.py: 交互式对话
  - qwen3_inference.py: 基础推理
  - qwen3_deploy.py: 模型部署
  - memory_efficient_train.py: 内存高效LoRA微调

**依赖包:**
- transformers>=4.40.0
- torch>=2.0.0
- accelerate、bitsandbytes、peft
- sentencepiece、tiktoken、huggingface_hub
- datasets

## 环境配置通用说明

### 1. 依赖安装
每个任务目录下都有requirements.txt文件，可以使用以下命令安装依赖：
```bash
pip install -r requirements.txt
```

### 2. 环境变量配置
对于使用DeepSeek API的任务（task2、task3、task5），需要配置.env文件：
```
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 网络要求
- 确保网络连接正常，能够访问DeepSeek API
- 对于task3，还需要能够访问DuckDuckGo搜索

## 测试提交方式

### 1. 单独测试
每个任务都可以单独运行和测试：
- task2: `python chatbot_with_llamaindex.py`
- task3: `python agent_workflow.py`
- task5: 先执行`python vectorize_markdown_file.py`，再执行`cd rag_llamaindex && python deepseek_qwen_rag_fixed.py`

### 2. 功能测试
- task2: 测试基本对话功能，验证记忆和系统提示词是否生效
- task3: 测试搜索功能，验证能否正确获取和处理搜索结果
- task4: 测试多智能体协作功能，验证不同类型智能体的工作情况
- task5: 测试RAG功能，验证知识库检索和生成效果
- task6: 测试模型推理和微调功能

### 3. 性能测试
- 测试各任务的响应时间和资源占用
- 对于task5，测试不同大小知识库的检索效率
- 对于task6，测试模型在不同硬件配置下的运行情况

## 总结

本智能体方向任务涵盖了从基础概念到实际实现的完整学习路径：
1. task1提供了理论基础
2. task2实现了基础对话功能
3. task3增加了搜索能力
4. task4实现了多智能体协作
5. task5引入了RAG知识库
6. task6提供了大模型应用工具

通过这些任务，可以全面了解智能体的设计原理、实现方法和应用场景。