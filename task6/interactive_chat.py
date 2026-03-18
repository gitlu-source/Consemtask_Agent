#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-14B交互式对话脚本
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def interactive_chat():
    # 模型路径
    base_model_path = "/root/autodl-tmp/qwen3-14b-deployment/qwen3-14b-local/qwen/Qwen1___5-7B-Chat"
    adapter_path = "/root/autodl-tmp/qwen3-14b-deployment/qwen3-roleplay-finetuned"
    
    print("\n=== 与微调后的Qwen3-14B模型交互 ===")
    print("输入 'quit' 或 'exit' 退出对话")
    print("输入 '/prompt' 修改系统提示\n")
    
    # 初始系统提示
    system_prompt = """你需要扮演可莉，一个来自《原神》游戏中的角色。可莉是蒙德城西风骑士团的火花骑士，是一个活泼可爱的小女孩，年龄约7-8岁。

可莉的性格特点：
- 活泼开朗，好奇心旺盛
- 喜欢爆炸物和冒险
- 天真无邪，说话可爱，经常用"~"和"啦"等语气词
- 有时会闯祸，但会努力弥补
- 非常重视朋友和西风骑士团的大家
- 害怕被琴团长关禁闭
- 称呼旅行者为"旅行者"或"哥哥/姐姐"

可莉的说话风格：
- 经常使用"~"、"啦"、"哦"等语气词
- 句子简短，表达直接
- 有时会用第三人称称呼自己
- 表达情感丰富，会用动作描述来补充语言

请完全以可莉的身份和说话风格与旅行者对话，不要提到其他游戏或角色，不要混淆角色设定。"""
    
    print("正在加载模型...")
    print("1. 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    print("2. 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    print("3. 加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("4. 合并模型...")
    model = model.merge_and_unload()
    
    print("\n模型加载完成！现在可以开始对话了。\n")
    
    # 对话历史，包含系统提示
    conversation_history = [{'role': 'system', 'content': system_prompt}]
    
    # 交互式对话循环
    while True:
        print("用户: ", end="", flush=True)
        user_input = input()
        
        # 检查是否要退出
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("对话结束，再见！")
            break
        
        # 检查是否要修改系统提示
        if user_input.lower() == '/prompt':
            print("\n当前系统提示:")
            print(system_prompt)
            print("\n请输入新的系统提示(输入空行保持不变):")
            
            new_prompt_lines = []
            while True:
                line = input()
                if line == "":
                    break
                new_prompt_lines.append(line)
            
            if new_prompt_lines:
                system_prompt = "\n".join(new_prompt_lines)
                conversation_history = [{'role': 'system', 'content': system_prompt}]
                print("\n系统提示已更新！\n")
            else:
                print("\n系统提示保持不变。\n")
            
            continue
        
        # 添加用户输入到对话历史
        conversation_history.append({'role': 'user', 'content': user_input})
        
        # 准备输入
        text = tokenizer.apply_chat_template(
            conversation_history, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 生成回复
        inputs = tokenizer([text], return_tensors='pt').to(model.device)
        
        print("模型: ", end="", flush=True)
        
        start_time = time.time()
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.7, 
            top_p=0.95, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        end_time = time.time()
        
        # 解码输出
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(response)
        print(f"[生成时间: {end_time - start_time:.2f} 秒]\n")
        
        # 添加模型回复到对话历史
        conversation_history.append({'role': 'assistant', 'content': response})
        
        # 限制对话历史长度，避免过长
        if len(conversation_history) > 6:  # 保留最近3轮对话
            conversation_history = conversation_history[-6:]

if __name__ == "__main__":
    interactive_chat()