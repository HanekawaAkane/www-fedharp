"""
Fed-HARP 联邦学习模拟主程序。
负责初始化环境、加载数据、构建模型以及执行联邦训练循环。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from typing import Dict, List, Optional, Set, Tuple


def parse_args(argv=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FedHARP: personalized federated LoRA with sensitivity-aware allocation")
    
    # 数据集参数
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="CIFAR-10 数据集目录")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "chestxray8"],
                        help="数据集名称 (cifar10/cifar100/chestxray8，默认: cifar10)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Non-IID Dirichlet 分布参数 (默认: 0.5)")
    parser.add_argument("--data_partition", type=str, default="dirichlet", choices=["dirichlet", "uniform"],
                        help="客户端数据划分方式: dirichlet=Non-IID, uniform=类别均衡 IID (默认: dirichlet)")
    parser.add_argument("--client_splits", type=str, default="0.9,0.1",
                        help="每个客户端本地 train/test 划分比例，例如 '0.9,0.1'")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="每个客户端使用的训练数据比例 (0-1, 如 0.01 表示 1%% 数据；默认: 1.0)")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                        help="ViT 模型名称 (默认: vit_base_patch16_224)")
    parser.add_argument("--pretrained", action="store_true",
                        help="是否使用预训练权重")
    parser.add_argument("--pretrained_source", type=str, default="local", choices=["modelscope", "local"],
                        help="预训练权重来源 (默认: modelscope)")
    parser.add_argument("--pretrained_path", type=str, default="vit",
                        help="本地 pretrained checkpoint 路径 (pretrained_source=local)")
    parser.add_argument("--ms_model_id", type=str, default=None,
                        help="ModelScope 模型 ID (pretrained_source=modelscope)")
    parser.add_argument("--ms_revision", type=str, default=None,
                        help="ModelScope revision/tag/commit (可选)")
    parser.add_argument("--ms_cache_dir", type=str, default=None,
                        help="ModelScope 缓存目录 (可选)")
    parser.add_argument("--ms_checkpoint_file", type=str, default=None,
                        help="ModelScope snapshot 内 checkpoint 相对路径 (可选)")
    parser.add_argument("--hf_endpoint", type=str, default=None,
                        help="HuggingFace Hub 端点/镜像 (如 https://hf-mirror.com)")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA 秩 (默认: 8)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha 缩放因子 (默认: 16.0)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout 率 (默认: 0.0)")

    # 联邦学习参数
    parser.add_argument("--method", type=str, default="fedharp",
                        choices=[
                            "fedharp",
                        ],
                        help="联邦方法 (默认: fedharp)")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="客户端总数 (默认: 10)")
    parser.add_argument("--num_rounds", type=int, default=1,
                        help="联邦训练轮数 (默认: 1)")
    parser.add_argument("--clients_per_round", type=int, default=None,
                        help="每轮选中的客户端数量 (默认: 全部)")
    parser.add_argument("--heterogeneity_type", type=str, default="6-3-1", choices=["uniform", "6-3-1", "1-1-1"],
                        help="客户端资源异构类型: 'uniform'(全部相同) 或 '6-3-1'(模拟高中低资源分布)")
    parser.add_argument("--allocation_ratio", type=float, default=0.5,
                        help="基础分配比例 (在 uniform 模式下使用，或作为低资源基准)")
    parser.add_argument("--aggregation_lr", type=float, default=1.0,
                        help="服务器聚合学习率 eta (默认: 1.0)")
    parser.add_argument("--struct_to_data_rounds", type=int, default=20,
                        help="全局先验到客户端敏感度反馈的过渡轮数 (默认: 20)")
    parser.add_argument("--allocation_temperature", type=float, default=1.0,
                        help="DSA softmax 温度，越大越平滑 (默认: 1.0)")
    parser.add_argument("--coverage_prior_weight", type=float, default=0.15,
                        help="DSA 覆盖先验权重，用于降低过度重复选择同一层 (默认: 0.15)")
    parser.add_argument("--sample_weight_power", type=float, default=0.5,
                        help="FedHARP 聚合中的有效样本权重指数 n_k^gamma (默认: 0.5)")
    parser.add_argument("--fedharp_matrix_mode", type=str, default="a", choices=["a", "b", "ab", "adaptive"],
                        help="FedHARP 家族共享/聚合的 LoRA 矩阵模式: a/b/ab/adaptive (默认: a)")
    parser.add_argument("--adaptive_matrix_threshold", type=float, default=1.25,
                        help="adaptive 模式下 A/B delta norm 的选择阈值 (默认: 1.25)")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批次大小 (默认: 128)")
    parser.add_argument("--client_lr", type=float, default=0.01,
                        help="客户端本地学习率 (默认: 0.01)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="每轮本地训练的 Epoch 数 (默认: 1)")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="B 矩阵预热对齐的步数 (默认: 10)")
    parser.add_argument("--warmup_lr", type=float, default=0.0001,
                        help="预热阶段的学习率 (默认: 0.0001)")
    parser.add_argument("--b_train_every", type=int, default=4,
                        help="每隔多少轮通信后触发一次 B-only 训练 (默认: 5)")
    parser.add_argument("--b_num_epochs", type=int, default=1,
                        help="B-only 阶段本地训练的 Epoch 数 (默认: 1)")
    parser.add_argument("--b_num_steps", type=int, default=3,
                        help="LFC B-only 阶段每次最多执行的 minibatch 数 (默认: 3；0 表示不限制)")
    parser.add_argument("--projection_mode", type=str, default="none",
                        choices=["none", "projection_only"],
                        help="重激活 stale block 的本地 B-factor transport；默认关闭")
    parser.add_argument("--projection_ridge", type=float, default=1e-6,
                        help="projection transport 的 Tikhonov ridge")
    
    # 评估参数
    parser.add_argument("--eval_every", type=int, default=1,
                        help="每多少轮评估一次 (默认: 1)")
    parser.add_argument("--test_batch_size", type=int, default=100,
                        help="测试批次大小 (默认: 100)")
    
    # 系统参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="使用的设备 (cuda/cpu, 默认: 自动检测)")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="是否保存模型检查点")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="检查点保存目录")

    # C0 compute-matched control 与 journal 诊断接口
    # （默认全部关闭；不传任何新参数时行为与权威源逐位一致）
    parser.add_argument("--journal_diagnostics", type=str, default=None,
                        help="诊断输出目录：启用后记录 per-round allocation trace 与 stale correction 事件（JSONL）")
    parser.add_argument("--replay_allocation_trace_path", type=str, default=None,
                        help="重放指定 full run 的 allocation trace（JSONL），保证 control 与 full 的分配/参与完全一致")
    parser.add_argument("--journal_compute_matched_control", action="store_true",
                        help="C0 compute-matched control：在与 stale 目标数量相同的非-stale 块上执行 B-only 训练")

    return parser.parse_args(args=argv)


def _strip_double_dash(argv):
    if argv is None:
        return None
    if len(argv) > 0 and argv[0] == "--":
        return argv[1:]
    return argv


def generate_client_resources(num_clients: int, heterogeneity_type: str, base_ratio: float) -> List[float]:
    """
    生成每个客户端的资源能力（即可以训练的层比例）。
    
    参数:
        num_clients: 客户端总数
        heterogeneity_type: 'uniform' 或 '6-3-1'
        base_ratio: 基础比例
        
    返回:
        List[float]: 每个客户端的分配比例列表
    """
    if heterogeneity_type == "uniform":
        return [base_ratio] * num_clients
    elif heterogeneity_type == "1-1-1":
        n_low = int(0.33 * num_clients)
        n_mid = int(0.33 * num_clients)
        n_high = num_clients - n_low - n_mid
        ratios = []
        ratios.extend([base_ratio] * n_low)           # Low
        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid) # Mid
        ratios.extend([1.0] * n_high)
        np.random.shuffle(ratios)
        return ratios
    elif heterogeneity_type == "6-3-1":
        # 模拟 Fed-HeLLo 论文中的资源分布
        # 60% 低资源 (基准比例, 如 0.25 或 0.5)
        # 30% 中资源 (基准 * 1.5 或 0.75)
        # 10% 高资源 (全量训练 1.0)
        n_low = int(0.6 * num_clients)
        n_mid = int(0.3 * num_clients)
        n_high = num_clients - n_low - n_mid
        
        # 定义不同等级的层比例
        # 注意：这里假设 base_ratio 是最低标准
        ratios = []
        ratios.extend([base_ratio] * n_low)           # Low
        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid) # Mid
        ratios.extend([1.0] * n_high)                 # High (Full Layers)
        
        # 打乱分配，使得 ID 不直接对应资源等级
        np.random.shuffle(ratios)
        return ratios
    
    return [base_ratio] * num_clients


def evaluate_global_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """在测试集上评估全局模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _dict_tensor_nbytes(d: Dict[str, torch.Tensor]) -> int:
    total = 0
    for _, v in d.items():
        total += _tensor_nbytes(v)
    return total


def _delta_A_nbytes(delta_A: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(delta_A)


def _delta_B_nbytes(delta_B: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(delta_B)


def _A_matrices_nbytes(A_matrices: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(A_matrices)


def _ab_trainable_param_numel(client_lora_layers: Dict[str, nn.Module], allocated_layers: Set[str]) -> int:
    n = 0
    for name in allocated_layers:
        layer = client_lora_layers.get(name)
        if layer is None:
            continue
        n += int(layer.lora_A.numel())
        n += int(layer.lora_B.numel())
    return n


def _b_only_trainable_param_numel(client_lora_layers: Dict[str, nn.Module], train_B_layers: Set[str]) -> int:
    n = 0
    for name in train_B_layers:
        layer = client_lora_layers.get(name)
        if layer is None:
            continue
        n += int(layer.lora_B.numel())
    return n


def _journal_write_line(path: str, obj: dict):
    """追加一行 JSONL（journal 诊断输出统一入口）。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True, sort_keys=True) + "\n")


def _journal_config_line(args) -> dict:
    """trace 文件首行：记录协议关键参数，供 control 重放前对齐审计。"""
    keys = [
        "method", "dataset", "data_partition", "alpha", "num_clients", "clients_per_round",
        "num_rounds", "heterogeneity_type", "allocation_ratio", "fedharp_matrix_mode",
        "b_train_every", "b_num_epochs", "b_num_steps", "data_fraction", "seed", "lora_rank",
        "client_lr", "num_epochs", "batch_size",
    ]
    cfg = {"type": "config", "code": "fedharp", "journal": "v1"}
    for k in keys:
        cfg[k] = getattr(args, k, None)
    return cfg


def _journal_load_trace(path: str):
    """读取 full run 的 allocation trace。返回 (config, {round: round_obj})。"""
    cfg = None
    rounds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "config":
                cfg = obj
            elif obj.get("type") == "round":
                rounds[int(obj["round"])] = obj
    if cfg is None or not rounds:
        raise SystemExit(f"[journal] trace 文件无效（缺少 config 或 round 行）: {path}")
    return cfg, rounds


def _pick_control_layers(stale_map, stale_layers, allocated_layers, layer_names, b_train_every):
    """为 compute-matched control 选择与 |stale_layers| 数量相同的非-stale 块。

    选择规则（确定性，可审计）：
      1) 优先「本轮未分配且 0 <= tau < b_train_every」的休眠块——其 B 不会被本轮
         local_train 覆写，与 full 臂在 stale 块上的修正语义最接近；按 (tau, 层名) 升序；
      2) 不足时回退到本轮已分配块（tau==0），按层名排序；
      3) 仍不足则记 shortfall（诚实报告，不得静默吞掉）。
    返回 (control_layers: Set[str], shortfall: int)。
    """
    need = len(stale_layers)
    pool_dormant = []
    pool_allocated = []
    for name in layer_names:
        if name in stale_layers:
            continue
        tau = int(stale_map.get(name, 0))
        if name in allocated_layers:
            pool_allocated.append(name)
        elif 0 <= tau < int(b_train_every):
            pool_dormant.append((tau, name))
    pool_dormant.sort()
    pool_allocated.sort()
    ordered = [name for _, name in pool_dormant] + pool_allocated
    chosen = ordered[:need]
    shortfall = need - len(chosen)
    return (set(chosen) if chosen else set()), shortfall


def vision_main(argv=None):
    """联邦学习主循环"""
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    
    # 设置随机种子
    global np
    global torch
    global nn
    global DataLoader
    global CIFAR10Dataset
    global CIFAR100Dataset
    global ChestXray8Dataset
    global FedHarpClient
    global FedHarpServer
    global create_vit_model
    global create_client_model
    global set_seed
    global get_device
    global save_checkpoint
    global create_directories
    global print_model_info

    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise SystemExit("Missing dependency 'numpy'. Install it before running vision experiments.") from e

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as e:
        raise SystemExit("Missing dependency 'torch'. Install it before running vision experiments.") from e

    from client import FedHarpClient
    from dataset import CIFAR10Dataset, CIFAR100Dataset, ChestXray8Dataset
    from models import create_client_model, create_vit_model
    from server import FedHarpServer
    from factor_projection import project_factor_dicts
    from utils import (
        create_directories,
        get_device,
        print_model_info,
        save_checkpoint,
        set_seed,
    )

    set_seed(args.seed)
    
    # 获取计算设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    # 创建目录
    if args.save_checkpoints:
        create_directories(args.checkpoint_dir)

    safe_method = str(args.method).replace("/", "_").replace(" ", "_")
    safe_dataset = str(args.dataset).replace("/", "_").replace(" ", "_")
    safe_model = str(args.model_name).replace("/", "_").replace(" ", "_")
    safe_alpha = str(args.alpha).replace("/", "_").replace(" ", "_")
    safe_partition = str(args.data_partition).replace("/", "_").replace(" ", "_")
    safe_heterogeneity_type = str(args.heterogeneity_type).replace("/", "_").replace(" ", "_")
    safe_b_train_every = str(args.b_train_every).replace("/", "_").replace(" ", "_")
    log_filename = f"Traininglog-{safe_method}-{safe_partition}-alpha{safe_alpha}-{safe_dataset}-{safe_model}-{safe_b_train_every}--{safe_heterogeneity_type}.txt"
    log_path = os.path.join(args.checkpoint_dir if args.save_checkpoints else ".", log_filename)

    # 将本次训练的配置写入日志文件（覆盖旧内容）
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Training configuration (arguments):\n")
            for k, v in sorted(vars(args).items()):
                f.write(f"{k} = {v}\n")
            f.write("\nRound logs:\n")
    except Exception as e:
        print(f"[警告] 无法写入日志文件 {log_path}: {e}")
    
    print("\n" + "="*80)
    print("FedHARP 实验流程")
    print("="*80)
    print("[配置] 本次实验参数")
    print(f"  方法: {args.method}")
    print(f"  数据集: {args.dataset.upper()} | 划分: {args.data_partition} | alpha={args.alpha}")
    print(f"  客户端: {args.num_clients} 个 | 通信轮数: {args.num_rounds}")
    print(f"  资源异构: {args.heterogeneity_type} | 基础层预算比例: {args.allocation_ratio}")
    print(f"  LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  设备: {device}")
    print("="*80 + "\n")
    
    # 加载数据集
    print(f"[数据] 正在加载 {args.dataset.upper()} 数据集...")
    try:
        splits = [float(x) for x in args.client_splits.split(",")]
    except:
        splits = [0.9, 0.1]
    
    if args.dataset == "cifar100":
        dataset = CIFAR100Dataset(
            data_dir=args.data_dir,
            alpha=args.alpha,
            num_clients=args.num_clients,
            seed=args.seed,
            splits=splits,
            data_partition=args.data_partition
        )
        num_classes = 100
    elif args.dataset == "chestxray8":
        dataset = ChestXray8Dataset(
            data_dir=args.data_dir,
            alpha=args.alpha,
            num_clients=args.num_clients,
            seed=args.seed,
            splits=splits,
            data_partition=args.data_partition
        )
        num_classes = int(getattr(dataset, "num_classes", 15))
    else:
        dataset = CIFAR10Dataset(
            data_dir=args.data_dir,
            alpha=args.alpha,
            num_clients=args.num_clients,
            seed=args.seed,
            splits=splits,
            data_partition=args.data_partition
        )
        num_classes = 10
    
    # 获取全局测试集
    global_test_loader = dataset.get_test_dataloader(batch_size=args.test_batch_size)
    
    # 创建全局模型
    print("\n[模型] 正在创建带有 LoRA 的 Vision Transformer...")
    global_model, global_lora_layers = create_vit_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        pretrained_source=args.pretrained_source,
        pretrained_checkpoint_path=args.pretrained_path,
        modelscope_model_id=args.ms_model_id,
        modelscope_revision=args.ms_revision,
        modelscope_cache_dir=args.ms_cache_dir,
        modelscope_checkpoint_file=args.ms_checkpoint_file,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    global_model.to(device)
    print_model_info(global_model, global_lora_layers)
    
    # --- 关键修改：生成异构的客户端资源能力 ---
    print(f"\n[资源] 正在生成客户端资源分布 ({args.heterogeneity_type})...")
    client_capacities = generate_client_resources(
        args.num_clients, 
        args.heterogeneity_type, 
        args.allocation_ratio
    )
    # 打印资源分布概览
    avg_cap = np.mean(client_capacities)
    print(f"  层训练比例: 平均={avg_cap:.2f}, 最小={min(client_capacities):.2f}, 最大={max(client_capacities):.2f}")
    if args.num_clients <= 20:
        print(f"  详细分布: {['{:.2f}'.format(c) for c in client_capacities]}")

    # 初始化服务器
    print("\n[服务器] 正在初始化 Fed-HARP 服务器...")
    # 注意：这里我们假设 server.py 已经更新了 __init__ 来接收 client_capacities
    # 如果 server.py 尚未更新，它可能会忽略这个参数，但我们在 main 中显式生成它以体现逻辑。
    # 为了兼容旧版 server 代码，我们保留 allocation_ratio 参数，但期望 server 使用 client_capacities
    server = FedHarpServer(
        model=global_model,
        lora_layers=global_lora_layers,
        num_clients=args.num_clients,
        method=args.method,
        allocation_ratio=args.allocation_ratio, # 作为默认值
        aggregation_lr=args.aggregation_lr,
        struct_to_data_rounds=args.struct_to_data_rounds,
        seed=args.seed,
        allocation_temperature=args.allocation_temperature,
        coverage_prior_weight=args.coverage_prior_weight,
        sample_weight_power=args.sample_weight_power,
        fedharp_matrix_mode=args.fedharp_matrix_mode,
    )
    # 手动将生成的异构能力注入到 server (模拟 Server 知晓客户端能力)
    if hasattr(server, 'client_layer_counts'):
        # 将比例转换为具体的层数
        num_total_layers = len(global_lora_layers)
        for cid, cap in enumerate(client_capacities):
            server.client_layer_counts[cid] = max(1, int(num_total_layers * cap))
        print("  已注入客户端层预算。")
    
    # 初始化客户端
    clients = []
    client_sample_counts = {}
    
    print(f"\n[客户端] 正在初始化 {args.num_clients} 个客户端...")
    for client_id in range(args.num_clients):
        # 获取本地训练集
        train_loader = dataset.get_client_dataloader(
            client_id,
            batch_size=args.batch_size,
            shuffle=True,
            split="train",
            fraction=args.data_fraction
        )
        # 获取本地测试集
        client_test_loader = dataset.get_client_dataloader(
            client_id,
            batch_size=args.test_batch_size,
            shuffle=False,
            split="test",
            fraction=1.0
        )
        
        client_sample_counts[client_id] = len(train_loader.dataset)
        
        # 创建客户端模型副本
        client_model, client_lora_layers = create_client_model(
            global_model,
            global_lora_layers
        )
        client_model.to(device)

        client = FedHarpClient(
            client_id=client_id,
            model=client_model,
            lora_layers=client_lora_layers,
            train_loader=train_loader,
            test_loader=client_test_loader, 
            device=device,
            lr=args.client_lr,
            method=args.method,
            warmup_steps=args.warmup_steps,
            warmup_lr=args.warmup_lr
        )
        clients.append(client)
        if client_id < 5:
            print(f"  客户端 {client_id}: train={len(train_loader.dataset)}")
    
    if args.num_clients > 5:
        print(f"  其余 {args.num_clients - 5} 个客户端已初始化。")
    
    # 确定每轮参与的客户端
    clients_per_round = args.clients_per_round or args.num_clients
    
    # 开始联邦学习主循环
    print("\n" + "="*80)
    print("[训练] 开始联邦训练")
    print("="*80)
    
    best_accuracy = 0.0
    training_history = []

    fedharp_matrix_mode = str(getattr(args, "fedharp_matrix_mode", "a")).lower().replace("-", "_")
    fedharp_upload_B = fedharp_matrix_mode in {"b", "ab", "adaptive"}
    b_train_every = int(args.b_train_every)
    client_personal_A_states = {cid: clients[cid].get_all_A_matrices() for cid in range(args.num_clients)}
    client_personal_B_states = {cid: clients[cid].get_all_B_matrices() for cid in range(args.num_clients)}

    # C0 journal / replay / compute-matched control 初始化（默认全关）
    journal_dir = getattr(args, "journal_diagnostics", None)
    journal_trace_path = os.path.join(journal_dir, "allocation_trace.jsonl") if journal_dir else None
    replay_cfg, replay_rounds = (None, None)
    if getattr(args, "replay_allocation_trace_path", None):
        replay_cfg, replay_rounds = _journal_load_trace(args.replay_allocation_trace_path)
        missing = [r for r in range(1, args.num_rounds + 1) if r not in replay_rounds]
        if missing:
            raise SystemExit(
                f"[journal] trace 缺少轮次 {missing[:5]}...，无法重放 {args.num_rounds} 轮"
            )
    compute_matched_control = bool(getattr(args, "journal_compute_matched_control", False))
    if compute_matched_control and replay_rounds is None:
        raise SystemExit("[journal] compute-matched control 必须同时提供 --replay_allocation_trace_path")
    if compute_matched_control and fedharp_matrix_mode != "a":
        raise SystemExit("[journal] compute-matched control 仅支持 fedharp_matrix_mode=a（C0 协议）")
    stale_target_total = 0
    correction_total = 0
    projection_total = 0
    projection_target_total = 0
    control_triggered_total = 0
    control_shortfall_total = 0
    if journal_dir:
        os.makedirs(journal_dir, exist_ok=True)
        _journal_write_line(journal_trace_path, _journal_config_line(args))
        print(f"[journal] 诊断输出目录: {journal_dir}")
    if replay_rounds is not None:
        print(
            f"[journal] 重放 allocation trace: {args.replay_allocation_trace_path} "
            f"(trace seed={replay_cfg.get('seed')}, rounds={len(replay_rounds)})"
        )
    if compute_matched_control:
        print("[journal] compute-matched control 模式已启用（B-only 步数与 stale 目标数对齐）")

    for round_num in range(1, args.num_rounds + 1):
        round_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"[轮次] {round_num}/{args.num_rounds}")
        print(f"{'='*80}")
        
        # 步骤 1: 服务器生成分配映射
        # 重放模式：保留 start_round() 原调用以对齐全局 RNG 消耗，
        # 随后用 trace 覆盖分配结果；staleness 先快照、重放后按 trace 重算
        # （staleness 是分配历史的纯函数，可精确重现 full run 的触发序列）。
        _staleness_before = (
            {cid: dict(m) for cid, m in server.staleness.items()}
            if replay_rounds is not None else None
        )
        allocation_map = server.start_round()
        if replay_rounds is not None:
            _trace_round = replay_rounds[round_num]
            server.staleness = {cid: dict(m) for cid, m in _staleness_before.items()}
            allocation_map = {int(k): set(v) for k, v in _trace_round["allocation"].items()}
        server.print_allocation_stats(allocation_map)
        
        # 步骤 2: 选择客户端
        selected_clients = np.random.choice(
            args.num_clients,
            size=min(clients_per_round, args.num_clients),
            replace=False
        )
        # 重放：覆盖为 trace 记录的参与序列（RNG 消耗保持一致）
        if replay_rounds is not None:
            selected_clients = np.array(replay_rounds[round_num]["selected"], dtype=np.int64)
        round_stale_targets = {}
        round_corrections = {}
        
        print(f"[步骤 2] 选择客户端: 本轮活跃 {len(selected_clients)} / {args.num_clients}")
        
        # 步骤 3: 客户端接收全局 A 并执行本地训练
        global_A = server.get_global_A_matrices()
        upload_b_method = fedharp_upload_B
        global_B = server.get_global_B_matrices() if upload_b_method else None
        downlink_A_mb = _A_matrices_nbytes(global_A) / (1024.0 * 1024.0)
        downlink_B_mb = (_A_matrices_nbytes(global_B) / (1024.0 * 1024.0)) if global_B is not None else 0.0
        client_updates = {}
        client_updates_B = {}
        client_metrics_ab = []
        
        for i, client_id in enumerate(selected_clients):
            client_idx = int(client_id)
            client = clients[client_idx]
            allocated_layers = allocation_map[client_idx]
            
            client_global_A = global_A
            client_global_B = global_B
            # Save the client's retained factor state before synchronization.
            # Projection is applied only to selected blocks whose inactivity
            # counter has reached q; these are the blocks being reactivated
            # in the current round.
            # Evaluation restores global A into the live client module. Use
            # the explicit per-client snapshots instead, otherwise A_old is
            # accidentally replaced by the current global A and the measured
            # stale-factor mismatch collapses to zero.
            projection_old_A = {
                name: tensor.clone()
                for name, tensor in client_personal_A_states.get(
                    client_idx, client.get_all_A_matrices()
                ).items()
            }
            projection_old_B = {
                name: tensor.clone()
                for name, tensor in client_personal_B_states.get(
                    client_idx, client.get_all_B_matrices()
                ).items()
            }
            fedharp_b_sharing = fedharp_matrix_mode == "b"
            if fedharp_b_sharing:
                client_global_A = client_personal_A_states[client_idx]
            if fedharp_b_sharing:
                if client_global_B is None:
                    raise ValueError("FedHARP-B requires the global B matrices")
                client.set_A_matrices(client_global_A)
                client.set_B_matrices(client_global_B)
            else:
                client.receive_global_matrices(client_global_A, client_global_B)

            client_downlink_A_mb = (
                0.0
                if fedharp_b_sharing
                else _A_matrices_nbytes(client_global_A) / (1024.0 * 1024.0)
            )
            client_downlink_B_mb = (
                _A_matrices_nbytes(client_global_B) / (1024.0 * 1024.0)
                if client_global_B is not None
                else 0.0
            )

            if (
                getattr(args, "projection_mode", "none") == "projection_only"
                and fedharp_matrix_mode == "a"
                and b_train_every > 0
            ):
                stale_map = server.staleness.get(client_idx, {})
                projection_layers = {
                    name for name, tau in stale_map.items()
                    if int(tau) >= int(b_train_every) and name in allocated_layers
                }
                if projection_layers:
                    projected_B, diagnostics = project_factor_dicts(
                        projection_old_A,
                        projection_old_B,
                        client.get_all_A_matrices(),
                        client.get_all_B_matrices(),
                        projection_layers,
                        ridge=float(args.projection_ridge),
                    )
                    client.set_B_matrices(projected_B)
                    projection_total += 1
                    projection_target_total += len(diagnostics)
                    for name, diag in diagnostics.items():
                        try:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(
                                    f"round={round_num}, type=projection_event, client={client_idx}, "
                                    f"layer={name}, tau={int(stale_map.get(name, -1))}, "
                                    f"ridge={float(args.projection_ridge):.8g}, "
                                    f"residual_before={float(diag['residual_before']):.8g}, "
                                    f"residual_after={float(diag['residual_after']):.8g}, "
                                    f"relative_reduction={float(diag['relative_reduction']):.8g}\n"
                                )
                        except Exception as e:
                            print(f"[警告] 无法写入 projection 事件: {e}")
                    if journal_dir is not None:
                        round_corrections[client_idx] = sorted(projection_layers)

            if b_train_every > 0:
                stale_map = server.staleness.get(client_idx, {})
                stale_layers = {
                    name for name, tau in stale_map.items()
                    if int(tau) >= int(b_train_every)
                }
                if getattr(args, "projection_mode", "none") == "projection_only":
                    # Projection replaces the old B-only LFC stage for
                    # reactivated blocks. Keep the dormant-block path out of
                    # this arm so the extra optimization step is zero.
                    stale_layers = set()
                if journal_dir is not None or compute_matched_control:
                    stale_target_total += len(stale_layers)
                    round_stale_targets[client_idx] = sorted(stale_layers)
                if len(stale_layers) > 0:
                    if compute_matched_control:
                        # C0 compute-matched control：
                        # 在数量相同的非-stale 块上执行 B-only 训练（算力对齐、
                        # 无 stale 修正语义），剥离"额外算力"混淆（R2-W5）。
                        control_layers, shortfall = _pick_control_layers(
                            stale_map, stale_layers, allocated_layers,
                            server.layer_names, b_train_every,
                        )
                        control_triggered_total += len(control_layers)
                        control_shortfall_total += shortfall
                        round_corrections[client_idx] = sorted(control_layers)
                        if len(control_layers) > 0:
                            control_sens = client.train_B_only(
                                control_layers,
                                num_epochs=args.b_num_epochs,
                                max_steps=(int(args.b_num_steps) if int(args.b_num_steps) > 0 else None),
                            )
                            server.update_client_sensitivities(client_idx, control_sens)
                    elif fedharp_matrix_mode == "b":
                        # Symmetric LPC realigns stale local A against the
                        # newly downloaded global B. DSA remains based on the
                        # B-gradient measured during standard local training.
                        client.train_A_only(
                            stale_layers,
                            num_epochs=args.b_num_epochs,
                            max_steps=(int(args.b_num_steps) if int(args.b_num_steps) > 0 else None),
                        )
                    else:
                        new_b_sensitivity = client.train_B_only(
                            stale_layers,
                            num_epochs=args.b_num_epochs,
                            max_steps=(int(args.b_num_steps) if int(args.b_num_steps) > 0 else None),
                        )
                        server.update_client_sensitivities(client_idx, new_b_sensitivity)
                        if journal_dir is not None:
                            correction_total += len(stale_layers)
                            round_corrections[client_idx] = sorted(stale_layers)
            
            # 本地训练
            if i < 3:
                print(f"  [步骤 3] 客户端 {client_idx} 本地训练: 分配层数={len(allocated_layers)}")

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            delta_A, delta_B, b_sensitivity = client.local_train(
                allocated_layers=allocated_layers,
                num_epochs=args.num_epochs,
                method=args.method,
                matrix_update_mode=fedharp_matrix_mode,
                adaptive_matrix_threshold=args.adaptive_matrix_threshold,
            )
            train_time_sec = time.time() - t0
            if torch.cuda.is_available() and device.type == "cuda":
                peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            else:
                peak_mem_gb = 0.0

            uplink_A_mb = _delta_A_nbytes(delta_A) / (1024.0 * 1024.0)
            if upload_b_method and delta_B is None:
                raise ValueError(f"{args.method} 需要上传 B，但 delta_B 为空")
            uplink_B_mb = (_delta_B_nbytes(delta_B) / (1024.0 * 1024.0)) if (upload_b_method and delta_B is not None) else 0.0
            uplink_mb = uplink_A_mb + uplink_B_mb
            comm_mb = client_downlink_A_mb + client_downlink_B_mb + uplink_mb
            train_samples = len(client.train_loader.dataset)
            trainable_numel = _ab_trainable_param_numel(client.lora_layers, allocated_layers)
            compute_tflops = (3.0 * float(trainable_numel) * float(train_samples) * float(args.num_epochs)) / 1e12
            first_client_layer = next(iter(client.lora_layers.values()), None)
            active_lora_rank = (
                int(first_client_layer.lora_A.shape[0])
                if first_client_layer is not None
                else 0
            )

            client_metrics_ab.append({
                "round": round_num,
                "client": client_idx,
                "phase": "ab_train",
                "lora_rank": int(active_lora_rank),
                "compute_tflops": float(compute_tflops),
                "peak_mem_gb": float(peak_mem_gb),
                "comm_mb": float(comm_mb),
                "train_time_sec": float(train_time_sec),
                "downlink_mb": float(client_downlink_A_mb + client_downlink_B_mb),
                "uplink_mb": float(uplink_mb)
            })
            
            client_updates[client_idx] = delta_A

            if upload_b_method:
                client_updates_B[client_idx] = delta_B

            client_personal_A_states[client_idx] = client.get_all_A_matrices()
            client_personal_B_states[client_idx] = client.get_all_B_matrices()
            
            # 上传 B 矩阵敏感度
            server.update_client_sensitivities(client_idx, b_sensitivity)

            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        "round={round}, type=client_metrics, phase={phase}, client={client}, "
                        "lora_rank={lora_rank}, "
                        "compute_tflops={compute_tflops:.6f}, peak_mem_gb={peak_mem_gb:.6f}, "
                        "comm_mb={comm_mb:.6f}, train_time_sec={train_time_sec:.6f}, "
                        "downlink_mb={downlink_mb:.6f}, uplink_mb={uplink_mb:.6f}\n".format(**client_metrics_ab[-1])
                    )
            except Exception as e:
                print(f"[警告] 无法写入日志文件 {log_path}: {e}")

        # per-round trace 写出（--journal_diagnostics 启用）
        if journal_dir is not None:
            _journal_write_line(journal_trace_path, {
                "type": "round",
                "round": round_num,
                "selected": [int(c) for c in selected_clients],
                "allocation": {int(cid): sorted(layers) for cid, layers in allocation_map.items()},
                "stale_targets": {int(k): sorted(v) for k, v in round_stale_targets.items()},
                "corrections": {int(k): sorted(v) for k, v in round_corrections.items()},
            })

        if len(client_metrics_ab) > 0:
            avg_compute = float(np.mean([m["compute_tflops"] for m in client_metrics_ab]))
            avg_mem = float(np.mean([m["peak_mem_gb"] for m in client_metrics_ab]))
            avg_comm = float(np.mean([m["comm_mb"] for m in client_metrics_ab]))
            avg_time = float(np.mean([m["train_time_sec"] for m in client_metrics_ab]))
            ranks = [int(m["lora_rank"]) for m in client_metrics_ab]
            print(
                f"[统计] 本轮客户端均值: 计算={avg_compute:.4f} TFLOPs, "
                f"显存峰值={avg_mem:.4f} GB, 通信={avg_comm:.2f} MB/round, "
                f"训练时间={avg_time:.2f} s/round, "
                f"LoRA秩={np.mean(ranks):.2f}[{min(ranks)},{max(ranks)}]"
            )
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=round_metrics, phase=ab_train, "
                        f"avg_lora_rank={np.mean(ranks):.4f}, min_lora_rank={min(ranks)}, max_lora_rank={max(ranks)}, "
                        f"avg_compute_tflops={avg_compute:.6f}, avg_peak_mem_gb={avg_mem:.6f}, "
                        f"avg_comm_mb={avg_comm:.6f}, avg_train_time_sec={avg_time:.6f}\n"
                    )
            except Exception as e:
                print(f"[警告] 无法写入日志文件 {log_path}: {e}")
        
        # 步骤 4: 服务器聚合更新
        print(f"\n[步骤 4] 服务器聚合: 接收 {len(client_updates)} 个客户端的 LoRA 更新")
        server.aggregate_matrices(
            client_updates_A=client_updates,
            client_updates_B=(client_updates_B if upload_b_method else None),
            client_sample_counts=client_sample_counts,
            allocation_map=allocation_map
        )
        server.update_staleness(allocation_map)

        # 步骤 5: 更新全局模型用于评估
        updated_global_A = server.get_global_A_matrices()
        for name, layer in global_lora_layers.items():
            if name in updated_global_A:
                layer.set_A(updated_global_A[name])

        if upload_b_method:
            updated_global_B = server.get_global_B_matrices()
            for name, layer in global_lora_layers.items():
                if name in updated_global_B:
                    layer.set_B(updated_global_B[name])
        
        # 步骤 5.5: 记录通信后的全局模型准确率 (已禁用，仅关注个性化)
        # print(f"\n正在评估通信后的全局模型准确率...")
        # global_model.eval()
        # comm_accuracy_results = evaluate_global_model(
        #     model=global_model,
        #     test_loader=global_test_loader,
        #     device=device
        # )
        # comm_accuracy = comm_accuracy_results['accuracy']
        # comm_loss = comm_accuracy_results['loss']
        # print(f"  [Round {round_num}] 通信后全局模型准确率: {comm_accuracy:.2f}%")
        # print(f"  [Round {round_num}] 通信后全局模型 Loss: {comm_loss:.4f}")

        # 追加写入通信后准确率到日志文件
        # try:
        #     with open(log_path, "a", encoding="utf-8") as f:
        #         f.write(
        #             f"round={round_num}, type=post_communication, "
        #             f"accuracy={comm_accuracy:.4f}, loss={comm_loss:.6f}\n"
        #         )
        # except Exception as e:
        #     print(f"[警告] 无法写入日志文件 {log_path}: {e}")
        
        # 更新通信后的最佳准确率
        # if comm_accuracy > best_comm_accuracy:
        #     best_comm_accuracy = comm_accuracy
        #     print(f"  ✓ 通信后准确率创新高!")
        
        # 记录到训练历史中
        # training_history.append({
        #     'round': round_num,
        #     'accuracy': comm_accuracy,
        #     'loss': comm_loss,
        #     'type': 'post_communication'  # 标记为通信后准确率
        # })

        # 步骤 6: 评估 (仅关注个性化)
        if round_num % args.eval_every == 0 or round_num == args.num_rounds:
            print(f"\n[步骤 6] 个性化评估: 在各客户端本地 Test 集上测试")

            global_eval_A = server.get_global_A_matrices()
            global_eval_B = server.get_global_B_matrices() if upload_b_method else None

            if fedharp_matrix_mode == "a":
                eval_adapter_policy = "global_A+local_B"
            elif fedharp_matrix_mode == "b":
                eval_adapter_policy = "local_A+global_B"
            else:
                eval_adapter_policy = "global_A+global_B"

            print(
                f"  [评估口径] adapter={eval_adapter_policy}, "
                "classifier=client_local"
            )
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=eval_policy, "
                        f"adapter={eval_adapter_policy}, classifier=client_local\n"
                    )
            except Exception as e:
                print(f"[警告] 无法写入评估口径: {e}")
            
            total_weighted_correct = 0.0
            total_samples = 0
            total_weighted_loss = 0.0
            
            for cid in range(args.num_clients):
                client = clients[cid]
                personal_A = client_personal_A_states.get(cid, client.get_all_A_matrices())
                personal_B = client_personal_B_states.get(cid, client.get_all_B_matrices())

                if fedharp_matrix_mode == "a":
                    eval_A = global_eval_A
                    eval_B = personal_B
                elif fedharp_matrix_mode == "b":
                    eval_A = personal_A
                    eval_B = global_eval_B
                else:
                    eval_A = global_eval_A
                    eval_B = global_eval_B

                client.set_A_matrices(eval_A)
                if eval_B is not None:
                    client.set_B_matrices(eval_B)
                
                if client.test_loader is None or len(client.test_loader.dataset) == 0:
                    continue
                    
                results = client.evaluate() 
                acc = results['accuracy']
                loss = results['loss']
                num_samples = len(client.test_loader.dataset)
                
                total_weighted_correct += (acc / 100.0) * num_samples
                total_weighted_loss += loss * num_samples
                total_samples += num_samples
                
                if cid < 3:
                    print(f"  客户端 {cid}: accuracy={acc:.2f}%, loss={loss:.4f}")

            if total_samples > 0:
                global_acc = 100.0 * total_weighted_correct / total_samples
                global_loss = total_weighted_loss / total_samples
            else:
                global_acc = 0.0
                global_loss = 0.0

            print(f"\n  [结果] 第 {round_num} 轮个性化加权准确率: {global_acc:.2f}%")
            print(f"  [结果] 第 {round_num} 轮个性化加权 loss: {global_loss:.4f}")
            
            training_history.append({
                'round': round_num,
                'accuracy': global_acc,
                'loss': global_loss,
                'type': 'personalized'  # 标记为个性化评估
            })

            # 追加写入个性化评估准确率到日志文件
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=personalized, "
                        f"accuracy={global_acc:.4f}, loss={global_loss:.6f}\n"
                    )
            except Exception as e:
                print(f"[警告] 无法写入日志文件 {log_path}: {e}")
            
            if global_acc > best_accuracy:
                best_accuracy = global_acc
                print("  [最佳] 刷新历史最佳个性化准确率。")
        
        # 保存检查点
        if args.save_checkpoints and round_num % 10 == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"checkpoint_round_{round_num}.pt"
            )
            save_checkpoint(
                model_state=global_model.state_dict(),
                lora_layers_state={name: layer.get_A() for name, layer in global_lora_layers.items()},
                round_num=round_num,
                filepath=checkpoint_path
            )
        
        round_time = time.time() - round_start_time
        print(f"\n[耗时] 第 {round_num} 轮: {round_time:.2f} 秒")
    
    # journal 收尾：control_shortfall 审计（C0 通过条件 3/4 的机器可查证据）
    if journal_dir is not None or compute_matched_control or replay_rounds is not None:
        _journal_summary = {
            "type": "summary",
            "mode": "control" if compute_matched_control else "full",
            "num_rounds": int(args.num_rounds),
            "stale_target_total": int(stale_target_total),
            "correction_total": int(correction_total),
            "control_triggered_total": int(control_triggered_total) if compute_matched_control else None,
            "control_shortfall": int(control_shortfall_total) if compute_matched_control else 0,
        }
        if journal_dir:
            _journal_write_line(os.path.join(journal_dir, "summary.jsonl"), _journal_summary)
        print("[journal] " + json.dumps(_journal_summary, ensure_ascii=True))
        if compute_matched_control:
            if control_shortfall_total > 0:
                print(
                    f"[journal][WARN] control shortfall = {control_shortfall_total}，"
                    f"未满足 compute-matched 条件，本 run 结果不可用于配对结论"
                )
            elif control_triggered_total != stale_target_total:
                print(
                    f"[journal][WARN] control_triggered({control_triggered_total}) != "
                    f"stale_target({stale_target_total})，需人工审计"
                )
            else:
                print(
                    f"[journal][OK] compute-matched 条件满足："
                    f"triggered == target == {control_triggered_total}，shortfall = 0"
                )

    # 最终总结
    print("\n" + "="*80)
    print("训练完成")
    print("="*80)
    print(f"[总结] 历史最佳个性化准确率: {best_accuracy:.2f}%")
    if getattr(args, "projection_mode", "none") == "projection_only":
        print(f"[总结] projection events={projection_total}, target blocks={projection_target_total}")
    # print(f"历史最佳通信后准确率: {best_comm_accuracy:.2f}%")
    
    # 分别显示通信后和个性化的最终准确率
    # comm_history = [e for e in training_history if e.get('type') == 'post_communication']
    personalized_history = [e for e in training_history if e.get('type') == 'personalized']
    
    # if comm_history:
    #     print(f"最终轮次通信后准确率: {comm_history[-1]['accuracy']:.2f}%")
    if personalized_history:
        print(f"[总结] 最终轮次个性化准确率: {personalized_history[-1]['accuracy']:.2f}%")

    if (
        hasattr(server, "uses_partial_layer_allocation")
        and server.uses_partial_layer_allocation()
    ):
        server.print_underallocated_coverage_stats()
    
    if len(training_history) > 0:
        print("\n[总结] 训练历史记录片段 (每 10 轮):")
        # print("  通信后准确率:")
        # for i, entry in enumerate(comm_history):
        #     if i % 10 == 0 or i == len(comm_history) - 1:
        #         print(f"    轮次 {entry['round']:3d}: 准确率 = {entry['accuracy']:.2f}%, Loss = {entry['loss']:.4f}")
        if personalized_history:
            print("  个性化准确率:")
            for i, entry in enumerate(personalized_history):
                if i % 10 == 0 or i == len(personalized_history) - 1:
                    print(f"    轮次 {entry['round']:3d}: 准确率 = {entry['accuracy']:.2f}%, Loss = {entry['loss']:.4f}")



_EMBEDDED_TASK_MODULES = {}
def _get_embedded_task_module(task_name: str):
    """Load a task entry module from <task>_entry_override.py next to main.py."""
    t = str(task_name).strip().lower()
    if t in _EMBEDDED_TASK_MODULES:
        return _EMBEDDED_TASK_MODULES[t]
    _entry_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"{t}_entry_override.py"
    )
    if not os.path.exists(_entry_path):
        raise SystemExit(f"缺少任务入口文件: {os.path.basename(_entry_path)}")
    with open(_entry_path, "r", encoding="utf-8") as _f:
        src = _f.read()
    m = types.ModuleType(f"embedded_{t}")
    m.__dict__["__name__"] = f"embedded_{t}"
    m.__dict__["__file__"] = _entry_path
    sys.modules[m.__dict__["__name__"]] = m
    exec(compile(src, _entry_path, "exec"), m.__dict__)
    _EMBEDDED_TASK_MODULES[t] = m
    return m

def _run_embedded_task(task_name: str, argv=None):
    m = _get_embedded_task_module(task_name)
    fn = getattr(m, "main", None)
    if fn is None:
        raise SystemExit(f"任务 {task_name} 缺少 main(argv) 入口")
    return fn(argv)



def main(argv=None):
    raw_argv = sys.argv[1:] if argv is None else list(argv)

    def _has_flag(args: List[str], flag: str) -> bool:
        prefix = flag + "="
        return any(a == flag or a.startswith(prefix) for a in args)

    def _get_flag_value(args: List[str], flag: str) -> str | None:
        prefix = flag + "="
        for i, a in enumerate(args):
            if a == flag:
                if i + 1 < len(args):
                    return args[i + 1]
                return None
            if a.startswith(prefix):
                return a[len(prefix) :]
        return None

    entry = argparse.ArgumentParser(description="统一 Fed-HARP 实验入口", add_help=True)
    entry.add_argument(
        "--task",
        type=str,
        default="vision",
        choices=["vision", "roberta"],
    )
    known, rest = entry.parse_known_args(args=raw_argv)
    rest = _strip_double_dash(rest)

    task_explicit = _has_flag(raw_argv, "--task")
    dataset_raw = _get_flag_value(raw_argv, "--dataset")
    model_path_raw = _get_flag_value(raw_argv, "--model_path")
    task_raw = _get_flag_value(raw_argv, "--task")

    dataset = (dataset_raw or "").strip().lower()
    task = (task_raw or str(known.task)).strip().lower()

    is_nlp = bool(model_path_raw) or dataset in {"sst2", "qnli"} or _has_flag(raw_argv, "--lora_targets")
    is_vision = dataset in {"cifar10", "cifar100", "chestxray8"} or _has_flag(raw_argv, "--model_name")

    if (not task_explicit) and is_nlp and not is_vision:
        task = "roberta"

    if task == "vision":
        return vision_main(rest)
    if task == "roberta":
        return _run_embedded_task("roberta", rest)
    raise SystemExit(f"未知 task: {task}")


if __name__ == "__main__":
    main()
