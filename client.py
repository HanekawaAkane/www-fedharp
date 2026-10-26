"""
Fed-HARP 客户端实现。
处理本地训练、B 矩阵的持久化保存以及 B 矩阵敏感度计算逻辑。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Set, Optional, Tuple
import copy
from models import LoRALinear, get_lora_layer_names


class FedHarpClient:
    """
    Fed-HARP 联邦学习客户端。
    
    核心功能:
    - 本地持久化 B 矩阵 (永不上传，实现 Native-privacy)
    - 联合训练分配的层 (Heterogeneous Allocation)
    - 计算 B 矩阵敏感度 (L2 Norm) 用于指导服务器分配策略
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        lora_layers: Dict[str, LoRALinear],
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),   
        lr: float = 0.001,
        warmup_steps: int = 10,
        warmup_lr: float = 0.0001,
        method: str = "fedharp"
    ):
        """
        初始化客户端。
        
        参数:
            client_id: 客户端唯一 ID
            model: 本地模型实例
            lora_layers: LoRA 层字典
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            device: 运行设备
            lr: 正常训练的学习率
            warmup_steps: B 矩阵对齐的预热步数 (预留参数)
            warmup_lr: 预热阶段的学习率 (预留参数)
        """
        self.client_id = client_id
        self.model = model
        self.lora_layers = lora_layers
        self.train_loader = train_loader
        # 可选的本地 test loader；若提供，则 evaluate() 无参时默认使用本地 test
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.method = "fedharp"
        
        # 兼容保留参数
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        
        # 记录上一轮中被冻结的层 (用于潜在的陈旧度控制逻辑)
        self.previously_frozen_layers: Set[str] = set()
        
        # 将模型移动到设备
        self.model.to(device)
        
        # 初始化优化器 (将在每次训练时根据可训练参数重建)
        self.optimizer = None
    
    def get_all_A_matrices(self) -> Dict[str, torch.Tensor]:
        """获取所有 A 矩阵 (用于上传给服务器聚合)"""
        return {name: layer.get_A() for name, layer in self.lora_layers.items()}
    
    def get_all_B_matrices(self) -> Dict[str, torch.Tensor]:
        """获取所有 B 矩阵"""
        return {name: layer.get_B() for name, layer in self.lora_layers.items()}
    
    def set_A_matrices(self, A_matrices: Dict[str, torch.Tensor]):
        """设置来自服务器的 A 矩阵"""
        for name, A in A_matrices.items():
            if name in self.lora_layers:
                self.lora_layers[name].set_A(A)

    def set_B_matrices(self, B_matrices: Dict[str, torch.Tensor]):
        """设置来自服务器的 B 矩阵"""
        for name, B in B_matrices.items():
            if name in self.lora_layers:
                self.lora_layers[name].set_B(B)

    def receive_global_matrices(
        self,
        A_matrices: Dict[str, torch.Tensor],
        B_matrices: Optional[Dict[str, torch.Tensor]] = None,
        method: Optional[str] = None
    ):
        self.set_A_matrices(A_matrices)
        if B_matrices is not None:
            self.set_B_matrices(B_matrices)

    def calculate_B_sensitivity(self) -> Dict[str, float]:
        sensitivities = {}
        for name in self.lora_layers.keys():
            sensitivities[name] = 0.0
        return sensitivities

    def _setup_layer_trainability(self, allocated_layers: Set[str]):
        """
        根据分配结果设置层的可训练性。
        当前策略：
        - 被分配的层: A 与 B 都参与训练
        - 未分配的层: A 与 B 都冻结
        """
        for name, layer in self.lora_layers.items():
            if name in allocated_layers:
                # 被分配的层: 同时训练 A 和 B
                layer.unfreeze_all()
            else:
                # 未分配的层: A 和 B 都冻结
                layer.freeze_all()

    def _setup_layer_trainability_B_only(self, train_B_layers: Set[str]):
        for param in self.model.parameters():
            param.requires_grad = False

        for name, layer in self.lora_layers.items():
            if name in train_B_layers:
                layer.freeze_A()
                layer.unfreeze_B()
            else:
                layer.freeze_all()

    def _setup_layer_trainability_A_only(self, train_A_layers: Set[str]):
        for param in self.model.parameters():
            param.requires_grad = False

        for name, layer in self.lora_layers.items():
            if name in train_A_layers:
                layer.unfreeze_A()
                layer.freeze_B()
            else:
                layer.freeze_all()
    
    def _create_optimizer(self, warmup_mode: bool = False):
        """创建优化器，仅包含当前标记为 requires_grad 的参数"""
        trainable_params = []
        seen = set()
        for layer in self.lora_layers.values():
            for p in layer.get_trainable_params():
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                trainable_params.append(p)

        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            trainable_params.append(p)
        
        if len(trainable_params) == 0:
            return None
        
        # 简化：统一使用 client LR
        lr = self.lr
        return optim.Adam(trainable_params, lr=lr)

    def _train_epoch_collect_B_grad_norms(
        self, warmup_mode: bool = False, include_A: bool = False,
        max_steps: Optional[int] = None
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}

        criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer(warmup_mode=warmup_mode)

        if self.optimizer is None:
            return 0.0, {name: 0.0 for name in self.lora_layers.keys()}

        steps_seen = 0
        for _, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()

            for name, layer in self.lora_layers.items():
                grad_sq = 0.0
                grad_b = layer.lora_B.grad
                if grad_b is not None:
                    n_b = torch.norm(grad_b.detach(), p=2).item()
                    grad_sq += float(n_b) * float(n_b)
                if include_A:
                    grad_a = layer.lora_A.grad
                    if grad_a is not None:
                        n_a = torch.norm(grad_a.detach(), p=2).item()
                        grad_sq += float(n_a) * float(n_a)
                if grad_sq <= 0.0:
                    continue
                n = float(grad_sq) ** 0.5
                grad_norm_sq_sum[name] += n * n

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            steps_seen += 1
            if max_steps is not None and steps_seen >= max(1, int(max_steps)):
                break

        if num_batches == 0:
            return 0.0, {name: 0.0 for name in self.lora_layers.keys()}

        grad_sensitivity = {name: (grad_norm_sq_sum[name] / num_batches) ** 0.5 for name in grad_norm_sq_sum.keys()}
        return total_loss / num_batches, grad_sensitivity

    def _train_epoch(self, warmup_mode: bool = False) -> float:
        """
        训练一个 Epoch。
        
        返回:
            平均 Loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer(warmup_mode=warmup_mode)
        
        if self.optimizer is None:
            return 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def local_train(
        self,
        allocated_layers: Set[str],
        num_epochs: int = 10,
        method: Optional[str] = None,
        matrix_update_mode: str = "a",
        adaptive_matrix_threshold: float = 1.25,
    ):
        """
        执行本地训练流程。
        
        参数:
            allocated_layers: 本轮分配给该客户端的层
            num_epochs: 训练轮数
        
        返回:
            delta_A: A 矩阵的更新量字典
            b_sensitivity: B 矩阵的敏感度字典 (用于下一轮分配)
        """
        method = method or self.method
        matrix_mode = str(matrix_update_mode or "a").lower().replace("-", "_")
        if matrix_mode not in {"a", "b", "ab", "adaptive"}:
            matrix_mode = "a"
        upload_B = matrix_mode in {"b", "ab", "adaptive"}

        # 记录训练前的 A/B 矩阵，用于计算 delta
        A_before = self.get_all_A_matrices()
        B_before = self.get_all_B_matrices() if upload_B else None

        # 设置所有分配层的可训练性
        self._setup_layer_trainability(allocated_layers)

        grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}
        grad_norm_steps = 0

        # 正常训练阶段 (联合更新 A 和 B)
        for epoch in range(num_epochs):
            train_loss, grad_sensitivity_epoch = self._train_epoch_collect_B_grad_norms(
                warmup_mode=False, include_A=False
            )
            # 打印每个 Epoch 信息
            print(f"  客户端 {self.client_id}: 训练 Epoch {epoch+1}/{num_epochs}, Loss = {train_loss:.4f}")

            for name, v in grad_sensitivity_epoch.items():
                grad_norm_sq_sum[name] += v * v
            grad_norm_steps += 1

        A_after = self.get_all_A_matrices()
        B_after = self.get_all_B_matrices() if upload_B else None

        updates_A: Dict[str, torch.Tensor] = {}
        updates_B: Optional[Dict[str, torch.Tensor]] = None

        if matrix_mode in {"b", "ab", "adaptive"}:
            updates_B = {}
            for name in allocated_layers:
                delta_a = A_after[name] - A_before[name] if name in A_before and name in A_after else None
                delta_b = (
                    B_after[name] - B_before[name]
                    if B_before is not None and B_after is not None and name in B_before and name in B_after
                    else None
                )
                if matrix_mode == "b":
                    if delta_b is not None:
                        updates_B[name] = delta_b
                    continue
                if matrix_mode == "ab":
                    if delta_a is not None:
                        updates_A[name] = delta_a
                    if delta_b is not None:
                        updates_B[name] = delta_b
                    continue
                if delta_a is None and delta_b is None:
                    continue
                norm_a = float(torch.norm(delta_a).item()) if delta_a is not None else 0.0
                norm_b = float(torch.norm(delta_b).item()) if delta_b is not None else 0.0
                threshold = max(1.0, float(adaptive_matrix_threshold))
                if norm_b > threshold * max(norm_a, 1e-12):
                    if delta_b is not None:
                        updates_B[name] = delta_b
                elif norm_a > threshold * max(norm_b, 1e-12):
                    if delta_a is not None:
                        updates_A[name] = delta_a
                else:
                    if delta_a is not None:
                        updates_A[name] = delta_a
                    if delta_b is not None:
                        updates_B[name] = delta_b
        else:
            for name in allocated_layers:
                if name in A_before and name in A_after:
                    updates_A[name] = A_after[name] - A_before[name]

            if upload_B and B_before is not None and B_after is not None:
                updates_B = {}
                for name in allocated_layers:
                    if name in B_before and name in B_after:
                        updates_B[name] = B_after[name] - B_before[name]
        
        # 更新 previously_frozen_layers (为了支持潜在的未来扩展)
        all_layer_names = set(self.lora_layers.keys())
        self.previously_frozen_layers = all_layer_names - allocated_layers

        if grad_norm_steps > 0:
            b_grad_sensitivity = {name: (grad_norm_sq_sum[name] / grad_norm_steps) ** 0.5 for name in grad_norm_sq_sum.keys()}
        else:
            b_grad_sensitivity = {name: 0.0 for name in self.lora_layers.keys()}

        return updates_A, updates_B, b_grad_sensitivity

    def train_B_only(
        self,
        train_B_layers: Set[str],
        num_epochs: int = 1,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        requires_grad_state = [(param, bool(param.requires_grad)) for param in self.model.parameters()]
        result = {name: 0.0 for name in self.lora_layers.keys()}
        try:
            self._setup_layer_trainability_B_only(train_B_layers)
            grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}
            grad_norm_steps = 0
            for epoch in range(num_epochs):
                train_loss, grad_sensitivity_epoch = self._train_epoch_collect_B_grad_norms(
                    warmup_mode=False,
                    max_steps=max_steps,
                )
                print(
                    f"  Client {self.client_id}: B-only correction Epoch {epoch+1}/{num_epochs}, "
                    f"steps={max_steps if max_steps is not None else 'all'}, Loss = {train_loss:.4f}"
                )
                for name, value in grad_sensitivity_epoch.items():
                    grad_norm_sq_sum[name] += value * value
                grad_norm_steps += 1
                if max_steps is not None:
                    break
            if grad_norm_steps > 0:
                result = {
                    name: (grad_norm_sq_sum[name] / grad_norm_steps) ** 0.5
                    for name in grad_norm_sq_sum.keys()
                }
        finally:
            for param, requires_grad in requires_grad_state:
                param.requires_grad = requires_grad
            self.optimizer = None
        return result

    def train_A_only(
        self,
        train_A_layers: Set[str],
        num_epochs: int = 1,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        requires_grad_state = [(param, bool(param.requires_grad)) for param in self.model.parameters()]
        result = {name: 0.0 for name in self.lora_layers.keys()}
        try:
            self._setup_layer_trainability_A_only(train_A_layers)
            grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}
            grad_norm_steps = 0
            for epoch in range(num_epochs):
                train_loss, grad_sensitivity_epoch = self._train_epoch_collect_B_grad_norms(
                    warmup_mode=False,
                    include_A=True,
                    max_steps=max_steps,
                )
                print(
                    f"  Client {self.client_id}: A-only correction Epoch {epoch+1}/{num_epochs}, "
                    f"Loss = {train_loss:.4f}"
                )
                for name, value in grad_sensitivity_epoch.items():
                    grad_norm_sq_sum[name] += value * value
                grad_norm_steps += 1
                if max_steps is not None:
                    break
            if grad_norm_steps > 0:
                result = {
                    name: (grad_norm_sq_sum[name] / grad_norm_steps) ** 0.5
                    for name in grad_norm_sq_sum.keys()
                }
        finally:
            for param, requires_grad in requires_grad_state:
                param.requires_grad = requires_grad
            self.optimizer = None
        return result

    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        在测试集上评估模型性能。
        如果未显式传入 test_loader，则优先使用客户端保存的本地 'test' 数据集。
        """
        if test_loader is None:
            if self.test_loader is None:
                raise ValueError("未提供测试集 DataLoader，且客户端未配置本地 test_loader")
            test_loader = self.test_loader
            
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
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
