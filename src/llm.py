from typing import Optional, Any, List
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel  
from langchain_openai import ChatOpenAI

# 引入 LangChain 自定义 Chat Model 所需的基类
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from src.app_config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)

# 可扩展的模型注册表
MODEL_REGISTRY = {
    "deepseek-chat": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
        "model": "deepseek-chat",
    },
    "openai-gpt-4o-mini": {
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
        "model": "gpt-4o-mini",
    },
    "qwen-lora": {
        "api_key": None,
        "base_url": None,
        "model": "qwen-lora",
    },
}

class LocalQwenChatModel(BaseChatModel):
    """自定义的 LangChain Chat Model，用于无缝对接本地 Qwen 模型"""
    model: Any
    tokenizer: Any

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        # 1. 将 LangChain 的 Message 格式转化为 Qwen 需要的字典格式
        formatted_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                formatted_messages.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                formatted_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                formatted_messages.append({"role": "assistant", "content": m.content})
            else:
                formatted_messages.append({"role": "user", "content": m.content})

        # 2. 应用 Chat Template
        text = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 3. 转化为 Tensor 并送入 GPU
        inputs = self.tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 4. 执行推理
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2, # 保持与 API 模型默认行为一致
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 5. 解码输出
        new_tokens = out_ids[0, inputs["input_ids"].shape[1] :]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 6. 组装成 LangChain 的标准返回格式
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "local-qwen-lora"

def pre_load_qwen_lora():
    lora_path = "./qwen_lora_ml"
    model_path = "Qwen/Qwen3.5-4B" # 之前纠正过的真实模型路径
        
    qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
    print(f"正在加载基础模型 {model_path} ...")
    base = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.float16,
            quantization_config=qconfig,
            attn_implementation="sdpa",
            trust_remote_code=True,   
        )
        
    tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
        )
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(lora_path):
            print("正在挂载 LoRA 权重...")
            local_model = PeftModel.from_pretrained(
                base,
                lora_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            print(" LoRA 权重挂载完成")
    else:
            print("LoRA 权重不存在，将使用原始基础模型")
            local_model = base

    local_model.eval()
    return local_model, tokenizer
    
    
def get_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    local_model: Optional[PeftModel] = None,
    tokenizer: Optional[AutoTokenizer] = None,
):
    if model and model in MODEL_REGISTRY:
        cfg = MODEL_REGISTRY[model]
        api_key = api_key or cfg.get("api_key")
        base_url = base_url or cfg.get("base_url")
        model_name = cfg.get("model")
        print(f"已选择模型 '{model_name}'，API Key 和 Base URL 已自动加载")
    else:
        print(f"未指定或识别的模型 '{model}'，将使用默认模型 '{DEEPSEEK_MODEL}'")
        model_name = model or DEEPSEEK_MODEL
        api_key = api_key or DEEPSEEK_API_KEY
        base_url = base_url or DEEPSEEK_BASE_URL
    
    if model_name == "qwen-lora":
        print('已选择本地 Qwen-Lora 模型进行推理')
        #  关键变动：不返回元组，而是返回封装好的 LangChain 对象
        return LocalQwenChatModel(model=local_model, tokenizer=tokenizer)
        

    if not api_key:
        raise ValueError("LLM API Key 未设置，请在 .env 中配置对应的密钥")

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.2,
    )