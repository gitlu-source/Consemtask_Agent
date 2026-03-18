#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对指定的Markdown文件进行向量化计算
"""

import os
import sys
import time
import requests
import json
import subprocess
from llama_index.core.node_parser import SentenceSplitter

# 设置中文字体支持
sys.stdout.reconfigure(encoding='utf-8')

# 安全打印函数
def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('utf-8', 'ignore').decode('utf-8'))

def check_ollama_service():
    """检查Ollama服务是否正在运行"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except Exception as e:
        safe_print(f"Ollama服务连接失败: {str(e)}")
        return False

def get_gpu_usage():
    """检查GPU使用情况"""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            output = result.stdout
            memory_used = None
            memory_total = None
            
            for line in output.split('\n'):
                if 'MiB' in line and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        memory_part = parts[2].strip()
                        if 'MiB' in memory_part:
                            mem_values = memory_part.split('/')
                            if len(mem_values) == 2:
                                memory_used = mem_values[0].strip().replace('MiB', '')
                                memory_total = mem_values[1].strip().replace('MiB', '')
                                break
            
            if memory_used and memory_total:
                memory_percent = (float(memory_used) / float(memory_total)) * 100
                return True, f"GPU内存使用: {memory_used} MiB / {memory_total} MiB ({memory_percent:.1f}%)"
            else:
                return False, "无法提取GPU内存信息"
        else:
            return False, "无法运行nvidia-smi工具"
    except Exception as e:
        return False, f"检查GPU使用情况时出错: {str(e)}"

# 优化的分块处理函数，借鉴参考代码中的SentenceSplitter
# 根据文本特征自动调整切片大小，范围在500-1000之间
def chunk_text_optimized(text, chunk_overlap=100):
    """优化的文本分块处理，根据文本特征自动调整切片大小在500-1000之间，确保句子完整性"""
    
    # 分析文本特征，动态调整chunk_size
    # 根据句子平均长度、段落复杂度等因素调整
    sentences = text.split('.')
    avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 100
    
    # 基于句子平均长度动态调整chunk_size
    if avg_sentence_length > 200:
        chunk_size = 800  # 长句子文本使用较小的块大小
    elif avg_sentence_length < 100:
        chunk_size = 1000  # 短句子文本使用较大的块大小
    else:
        chunk_size = 900  # 适中句子长度使用中间值
    
    # 确保chunk_size在500-1000范围内
    chunk_size = max(500, min(1000, chunk_size))
    
    # 使用llama_index的SentenceSplitter进行分块
    from llama_index.core.schema import Document
    document = Document(text=text)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([document])
    
    # 提取文本块
    chunks = [node.get_content() for node in nodes]
    
    return chunks, chunk_size

def vectorize_text(text, model_name="dengcao/Qwen3-Embedding-8B:Q5_K_M", dimensions=5000):
    """对文本进行向量化计算，使用新的大模型，支持指定维度"""
    try:
        url = "http://localhost:11434/api/embeddings"
        
        # 借鉴参考代码的模型调用方式，添加内存优化参数和GPU设备选择
        payload = {
            "model": model_name,
            "prompt": text,
            "options": {
                "embedding_dim": dimensions,
                # 添加GPU设备选择和内存优化参数
                "num_gpu_layers": 32,  # 使用GPU加速的层数
                "context_length": 8192,  # 增加上下文长度以处理长文本
                # 选择GPU1（NVIDIA GeForce RTX 4060 Laptop GPU）
                "cuda_device": 1
            }
        }
        
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get("embedding", [])
            
            return True, {
                "embedding": embedding,
                "embedding_dim": len(embedding),
                "processing_time": end_time - start_time,
                "chars_processed": len(text)
            }
        else:
            return False, f"向量化请求失败，状态码: {response.status_code}, 响应内容: {response.text}"
    except Exception as e:
        return False, f"向量化计算时出错: {str(e)}"

def read_markdown_file(file_path):
    """读取Markdown文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return True, file.read()
    except Exception as e:
        return False, f"读取文件时出错: {str(e)}"

def save_embedding_results(file_path, results):
    """保存向量化结果到文件"""
    try:
        output_file = file_path.replace('.md', '_embedding.json')
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
        return True, output_file
    except Exception as e:
        return False, f"保存结果时出错: {str(e)}"

def main():
    """主函数"""
    safe_print("===== Markdown文件向量化计算 ======")
    
    # 检查Ollama服务
    if not check_ollama_service():
        safe_print("❌ Ollama服务未运行或无法连接！")
        safe_print("   请确保Ollama服务正在运行，然后重新运行此脚本。")
        return
    
    # 指定要向量化的文件路径
    file_path = "d:\\Desktop\\智能体方向\\多元分析学_markdown.md"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        safe_print(f"❌ 文件不存在: {file_path}")
        return
    
    safe_print(f"正在处理文件: {file_path}")
    
    # 读取文件内容
    read_success, file_content = read_markdown_file(file_path)
    if not read_success:
        safe_print(f"❌ {file_content}")
        return
    
    # 获取文件基本信息
    file_size = os.path.getsize(file_path) / 1024  # KB
    char_count = len(file_content)
    line_count = file_content.count('\n') + 1
    
    safe_print(f"文件信息:")
    safe_print(f"  - 文件大小: {file_size:.2f} KB")
    safe_print(f"  - 字符数量: {char_count} 字符")
    safe_print(f"  - 行数: {line_count} 行")
    
    # 显示前100个字符作为预览
    preview = file_content[:100] + ('...' if len(file_content) > 100 else '')
    safe_print(f"文件内容预览: {preview}")
    
    # 检查初始GPU使用情况
    gpu_success, gpu_message = get_gpu_usage()
    if gpu_success:
        safe_print(f"初始{gpu_message}")
    else:
        safe_print(f"⚠️ {gpu_message}")
    
    # 开始向量化计算
    safe_print("\n正在进行向量化计算...")
    safe_print("这可能需要一些时间，请耐心等待...")
    
    # 分块处理大文件
    all_embeddings = []
    total_time = 0
    
    # 使用优化的分块函数，自动调整切片大小
    chunks, actual_chunk_size = chunk_text_optimized(file_content, chunk_overlap=100)
    
    safe_print(f"文件已分为{len(chunks)}个优化的文本块，平均每块{int(char_count/len(chunks))}字符 (自动调整的块大小: {actual_chunk_size}字符)")
    
    for i, chunk in enumerate(chunks):
        safe_print(f"处理块 {i+1}/{len(chunks)} ({len(chunk)}字符)...")
        # 尝试提高向量化维度到5000
        success, result = vectorize_text(chunk, dimensions=5000)
        if success:
            all_embeddings.append(result["embedding"])
            total_time += result["processing_time"]
            safe_print(f"  ✅ 块{i+1}处理完成，嵌入维度: {result['embedding_dim']}")
        else:
            safe_print(f"❌ 处理块{i+1}时出错: {result}")
            return
    
    # 计算平均处理速度
    avg_speed = char_count / total_time if total_time > 0 else 0
    
    # 检查最终GPU使用情况
    final_gpu_success, final_gpu_message = get_gpu_usage()
    if final_gpu_success:
        safe_print(f"\n最终{final_gpu_message}")
    
    # 保存结果
    results = {
        "file_path": file_path,
        "file_size_kb": file_size,
        "char_count": char_count,
        "line_count": line_count,
        "chunks_processed": len(chunks),
        "embeddings": all_embeddings,
        "embedding_dim": len(all_embeddings[0]) if all_embeddings else 0,
        "total_processing_time": total_time,
        "avg_processing_speed": avg_speed,
        "gpu_used": gpu_success,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    save_success, save_message = save_embedding_results(file_path, results)
    if save_success:
        safe_print(f"\n✅ 向量化结果已保存到: {save_message}")
    else:
        safe_print(f"❌ {save_message}")
    
    # 显示总结
    safe_print("\n===== 向量化计算总结 =====")
    safe_print(f"总处理时间: {total_time:.2f} 秒")
    safe_print(f"平均处理速度: {avg_speed:.2f} 字符/秒")
    safe_print(f"嵌入向量维度: {results['embedding_dim']}")
    
    # 判断是否使用了GPU加速
    if avg_speed > 50:
        safe_print("✅ 根据处理速度判断，很可能已使用GPU加速")
    else:
        safe_print("⚠️ 处理速度较慢，可能未充分利用GPU加速")
    
    safe_print("\n===== 处理完成 =====")

if __name__ == "__main__":
    main()