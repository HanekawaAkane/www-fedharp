"""
FedHARP 的工具函数。
包含随机种子设置、设备获取、模型保存与加载等辅助功能。
"""

import torch
import numpy as np
import random
from typing import Optional
import os


def set_seed(seed: int = 42):
    """
    设置随机种子以保证结果可复现。
    
    参数:
        seed: 随机种子数值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    获取可用设备 (优先使用 CUDA，否则使用 CPU)。
    
    返回:
        torch.device: 计算设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"使用设备: {device}")
    return device


def save_checkpoint(
    model_state: dict,
    lora_layers_state: dict,
    round_num: int,
    filepath: str
):
    """
    保存模型检查点。
    只保存 LoRA 的 A 矩阵状态和非 LoRA 部分（如果非 LoRA 部分也训练的话）。
    """
    checkpoint = {
        'round': round_num,
        'model_state': model_state,
        'lora_layers_state': lora_layers_state
    }
    torch.save(checkpoint, filepath)
    print(f"检查点已保存至 {filepath}")


def load_checkpoint(filepath: str):
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location='cpu')
    return checkpoint


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """计算模型参数数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_number(num: int) -> str:
    """格式化大数字（添加逗号）"""
    return f"{num:,}"


def create_directories(*dirs: str):
    """如果目录不存在则创建"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def print_model_info(model: torch.nn.Module, lora_layers: dict):
    """
    打印模型信息摘要。
    
    显示总参数量、可训练参数量以及 LoRA 参数占比。
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    lora_params = sum(
        layer.lora_A.numel() + layer.lora_B.numel()
        for layer in lora_layers.values()
    )
    
    print("\n" + "="*60)
    print("模型信息摘要")
    print("="*60)
    print(f"总参数量: {format_number(total_params)}")
    print(f"可训练参数量: {format_number(trainable_params)}")
    print(f"LoRA 参数量 (A+B): {format_number(lora_params)}")
    print(f"LoRA 层数量: {len(lora_layers)}")
    print(f"LoRA 参数占比效率: {100.0 * lora_params / total_params:.4f}%")
    print("="*60 + "\n")