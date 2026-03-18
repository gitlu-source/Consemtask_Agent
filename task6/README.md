# Qwen3-14B 核心模型脚本集合

本目录包含了运行和部署Qwen3-14B对话模型的四个核心脚本，已优化精简，保留了模型运行的核心功能。

## 脚本列表

| 脚本名称 | 主要功能 | 文件大小 |
|---------|---------|---------|
| **interactive_chat.py** | 交互式对话系统，支持用户与模型进行对话交互 | ~4.9KB |
| **qwen3_inference.py** | 基础推理脚本，优化的4位量化推理，快速启动对话 | ~3.4KB |
| **qwen3_deploy.py** | 模型部署工具，支持多种量化策略，自动适配设备 | ~13.7KB |
| **memory_efficient_train.py** | 内存高效的LoRA微调脚本，支持8位量化训练 | ~6.6KB |

## 脚本功能说明

### 1. interactive_chat.py
- **功能**：提供与微调后Qwen3-14B模型的交互式对话界面
- **特点**：简洁的对话历史管理，支持控制对话轮数
- **使用方法**：`python interactive_chat.py`
- **退出方式**：输入'exit'、'quit'或'退出'

### 2. qwen3_inference.py
- **功能**：轻量级的模型推理入口，优化了内存占用
- **特点**：默认启用4位量化以节省显存，适用于资源受限环境
- **使用方法**：`python qwen3_inference.py`
- **参数**：可通过修改代码中的`max_new_tokens`调整生成内容长度

### 3. qwen3_deploy.py
- **功能**：灵活的模型部署工具，支持在线/离线模式
- **支持的量化策略**：4位量化(nf4)、8位量化、FP16、BF16
- **自动设备适配**：根据硬件环境自动选择合适的设备映射
- **使用方法**：`python qwen3_deploy.py`
- **配置**：可在代码中修改`model_config`字典调整部署参数

### 4. memory_efficient_train.py
- **功能**：内存高效的LoRA微调实现
- **特点**：8位量化加载模型，小参数LoRA配置(8 rank)
- **支持的数据集**：角色扮演数据集(默认使用BigPancake01/roleplayLLM_Chinese)
- **使用方法**：`python memory_efficient_train.py`
- **优化**：梯度检查点、adafactor优化器、梯度累积等内存优化技术

## 环境要求

在运行脚本前，请先安装依赖：

```bash
# 安装所有必要的依赖
pip install -r requirements.txt
```

主要依赖包：
- python >= 3.8
- torch >= 2.0.0
- transformers >= 4.40.0
- bitsandbytes
- datasets
- peft
- accelerate
- sentencepiece
- tiktoken
- huggingface_hub

## 使用建议

1. **快速开始对话**：直接运行`qwen3_inference.py`获取基础对话功能
2. **完整交互式体验**：使用`interactive_chat.py`进行更友好的对话交互
3. **首次部署模型**：运行`qwen3_deploy.py`下载并配置模型
4. **微调自定义模型**：使用`memory_efficient_train.py`基于自定义数据集进行微调

## 注意事项

- 所有脚本都需要模型文件支持，默认从本地路径`./qwen3-14b-finetuned`加载
- 如需从Hugging Face下载模型，请修改`qwen3_deploy.py`中的`model_id`参数
- 在显存不足的环境中，推荐使用4位量化模式运行
- 微调需要较大的内存/显存，建议在配置较高的GPU环境中运行

## 脚本特点

- **代码精简**：删除了冗余功能和示例代码
- **核心功能保留**：所有脚本都保留了运行模型的核心功能
- **执行权限**：所有Python脚本已设置执行权限
- **中文支持**：完全支持中文输入输出