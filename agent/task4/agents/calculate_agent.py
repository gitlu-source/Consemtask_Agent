"""
计算Agent模块
负责处理数学计算任务
"""

from typing import Optional
import re

class CalculateAgent:
    """计算Agent，擅长处理数学计算任务"""
    def __init__(self, llm=None):
        self.name = "计算Agent"
        self.llm = llm  # 计算Agent可能不需要LLM
        print(f"{self.name} 初始化完成")
    
    def can_handle(self, query: str) -> bool:
        """判断是否为数学计算任务"""
        # 检查是否包含常见的数学运算符
        math_patterns = [
            r'[\+\-××\*÷/]',  # 加减乘除运算符
            r'\d+\s*[\+\-××\*÷/]\s*\d+',  # 简单表达式
            r'(计算|求|算|等于|等于|总和|差|积|商|平方|立方|次方)',  # 关键词
            r'\d+\s*(乘以|乘|加|减|除以|除)\s*\d+',  # 中文数学表达式
        ]
        
        # 检查是否包含数字
        if not re.search(r'\d', query):
            return False
        
        # 检查是否包含数学相关模式
        for pattern in math_patterns:
            if re.search(pattern, query):
                return True
        
        return False
    
    def process_query(self, query: str) -> str:
        """处理数学计算任务"""
        try:
            print(f"   {self.name} 正在计算...")
            
            # 预处理查询，提取数学表达式
            expr = self._extract_math_expression(query)
            if not expr:
                return f"抱歉，我无法从您的问题中提取有效的数学表达式。"
            
            # 安全计算表达式
            result = self._safe_calculate(expr)
            return f"计算结果：{expr} = {result}"
        except Exception as e:
            print(f"{self.name} 处理出错: {str(e)}")
            return f"{self.name} 无法完成计算: {str(e)}"
    
    def _extract_math_expression(self, query: str) -> str:
        """从查询中提取数学表达式"""
        # 处理中文数学表达式
        query = query.replace('乘以', '*')
        query = query.replace('乘', '*')
        query = query.replace('加', '+')
        query = query.replace('减', '-')
        query = query.replace('除以', '/')
        query = query.replace('除', '/')
        query = query.replace('×', '*')
        query = query.replace('÷', '/')
        
        # 提取数字和运算符
        expr = re.findall(r'\d+(?:\.\d+)?|[\+\-\*/\(\)]', query)
        return ''.join(expr)
    
    def _safe_calculate(self, expr: str) -> float:
        """安全计算数学表达式"""
        # 仅允许基本的算术运算
        allowed_chars = set('0123456789.+-*/() ')
        if not all(c in allowed_chars for c in expr):
            raise ValueError("表达式包含不支持的字符")
        
        # 安全计算（使用eval但限制了表达式内容）
        # 注意：在实际生产环境中，应使用更安全的方法计算表达式
        result = eval(expr, {
            '__builtins__': {},
            'abs': abs,
            'pow': pow
        })
        
        # 如果结果是整数，返回整数格式
        if isinstance(result, float) and result.is_integer():
            return int(result)
        
        return result