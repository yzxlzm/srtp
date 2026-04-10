import re
from typing import Optional


class ModelSelector:
    """根据问题特征自动选择合适的模型"""

    # 需要推理能力的关键词
    REASONING_KEYWORDS = [
        "为什么", "如何", "怎么", "证明", "推导", "计算", "求解",
        "why", "how", "prove", "derive", "calculate", "solve",
        "代码", "算法", "bug", "错误", "调试", "code", "algorithm", "debug"
    ]

    # 简单问答关键词
    SIMPLE_KEYWORDS = [
        "是什么", "定义", "介绍", "什么是",
        "what is", "define", "definition", "introduce"
    ]


    MECHINE_LEARNING_KEYWORDS = [
        "机器学习", "深度学习", "神经网络", "模型", "算法", "数据", "训练", "预测", "分类", "回归", "聚类", "特征", "损失函数", "优化器", "过拟合", "欠拟合", "泛化", "交叉验证", "超参数", "参数", "梯度", "反向传播", "激活函数",
        "machine learning", "deep learning", "neural network", "model", "algorithm", "data", "train", "predict", "classification", "regression", "clustering", "feature", "loss function", "optimizer", "overfitting", "underfitting", "generalization", "cross validation", "hyperparameter", "parameter", "gradient", "backpropagation", "activation function"
    ]

    @staticmethod
    def select_model(question: str, user_model: Optional[str] = None) -> str:
        """
        自动选择模型

        Args:
            question: 用户问题
            user_model: 用户指定的模型（优先级最高）

        Returns:
            模型 key
        """
        # 用户指定模型则直接使用
        if user_model:
            return user_model

        q = question.lower()

        # 1. 检测是否需要推理能力
        if any(kw in q for kw in ModelSelector.MECHINE_LEARNING_KEYWORDS):
            return "qwen_lora"
        if any(kw in q for kw in ModelSelector.REASONING_KEYWORDS):
            # 包含数学符号或代码块
            if re.search(r'[+\-*/=<>]|\d+|```|`', question):
                return "deepseek-reasoner"
            return "deepseek-chat"

        # 2. 简单定义类问题
        if any(kw in q for kw in ModelSelector.SIMPLE_KEYWORDS) and len(question) < 50:
            return "openai-gpt-4o-mini"

        # 3. 默认使用 deepseek-chat（性价比高）
        return "deepseek-chat"
