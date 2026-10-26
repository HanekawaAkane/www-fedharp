"""
Fed-HARP 服务器实现。
处理异构层分配、陈旧度跟踪以及陈旧度加权的聚合逻辑。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Optional
import numpy as np
from models import LoRALinear, get_lora_layer_names


class FedHarpServer:
    """
    Fed-HARP 联邦学习服务器。
    
    核心功能:
    - 生成分配映射 (Allocation Map): 
        * 融合全局结构先验与客户端 B 矩阵敏感度
        * 按融合分数进行无放回概率抽样
    - 跟踪陈旧度 (Staleness): 记录每个客户端每个层有多少轮未更新
    - 执行陈旧度衰减聚合 (Staleness-Weighted Aggregation): 对陈旧更新进行降权
    """
    
    def __init__(
        self,
        model: nn.Module,
        lora_layers: Dict[str, LoRALinear],
        num_clients: int,
        allocation_ratio: float = 0.5,
        aggregation_lr: float = 1.0,  # 注意：如果是加权平均，这里的 LR 通常设为 1.0，或者作为全局学习率
        struct_to_data_rounds: int = 20,
        seed: int = 42,
        method: str = "fedharp",
        sensitivity_momentum: float = 0.8,
        allocation_temperature: float = 1.0,
        coverage_prior_weight: float = 0.15,
        sample_weight_power: float = 0.5,
        fedharp_matrix_mode: str = "a",
    ):
        """
        初始化服务器。
        
        参数:
            model: 全局模型实例
            lora_layers: LoRA 层字典
            num_clients: 客户端总数
            allocation_ratio: 每个客户端更新的层比例 (例如 0.5 表示 50%)
            aggregation_lr: 服务器聚合学习率 (eta)。如果使用标准 FedAvg 逻辑，这通常是 1.0。
            seed: 随机分配种子
        """
        self.model = model
        self.lora_layers = lora_layers
        self.num_clients = num_clients
        self.allocation_ratio = allocation_ratio
        self.aggregation_lr = aggregation_lr
        self.struct_to_data_rounds = max(1, int(struct_to_data_rounds))
        self.seed = seed
        self.method = "fedharp"
        
        # 获取排序后的层名称列表
        self.layer_names = get_lora_layer_names(lora_layers)
        if len(lora_layers) > 0 and len(self.layer_names) == 0:
            raise ValueError(
                "No attention LoRA layers are available for FedHARP allocation. "
                "Please set lora_targets to attention projections such as "
                "qkv/query/value/q_proj/v_proj; fc/mlp/classifier adapters are "
                "not allocated by default."
            )
        self.num_layers = len(self.layer_names)
        
        # 陈旧度跟踪表: staleness[client_id][layer_name] = 自上次更新以来的轮数
        self.staleness: Dict[int, Dict[str, int]] = {
            client_id: {layer_name: 0 for layer_name in self.layer_names}
            for client_id in range(num_clients)
        }
        
        # 存储客户端上传的敏感度 (B 矩阵范数)
        # client_sensitivities[client_id][layer_name] = float value
        self.client_sensitivities: Dict[int, Dict[str, float]] = {}
        
        # 当前轮次
        self.current_round = 0

        # 记录每层被选中的总次数，用于覆盖校正先验
        self.layer_selection_counts: Dict[str, int] = {
            layer_name: 0 for layer_name in self.layer_names
        }

        self.client_layer_counts: Dict[int, int] = {}
        self.client_allocation_history: Dict[int, List[Set[str]]] = {
            client_id: [] for client_id in range(num_clients)
        }
        self.client_first_allocation: Dict[int, Set[str]] = {}
        self.client_unique_allocated_layers: Dict[int, Set[str]] = {
            client_id: set() for client_id in range(num_clients)
        }
        self.client_first_new_layer_round: Dict[int, Dict[str, int]] = {
            client_id: {} for client_id in range(num_clients)
        }

        try:
            m = float(sensitivity_momentum)
        except Exception:
            m = 0.8
        if m < 0.0:
            m = 0.0
        if m >= 1.0:
            m = 0.99
        self.sensitivity_momentum = m

        try:
            temp = float(allocation_temperature)
        except Exception:
            temp = 1.0
        self.allocation_temperature = max(1e-6, temp)

        try:
            prior_w = float(coverage_prior_weight)
        except Exception:
            prior_w = 0.15
        self.coverage_prior_weight = max(0.0, min(1.0, prior_w))

        try:
            sample_power = float(sample_weight_power)
        except Exception:
            sample_power = 0.5
        self.sample_weight_power = max(0.0, min(1.0, sample_power))
        matrix_mode = str(fedharp_matrix_mode or "a").lower().replace("-", "_")
        if matrix_mode not in {"a", "b", "ab", "adaptive"}:
            matrix_mode = "a"
        self.fedharp_matrix_mode = matrix_mode
        
        # 设置随机种子
        np.random.seed(seed)
    
    def update_client_sensitivities(self, client_id: int, sensitivities: Dict[str, float]):
        """
        接收并更新客户端的 B 矩阵敏感度信息。
        这些信息将用于下一轮的分配。
        """
        prev = self.client_sensitivities.get(client_id, {})
        if prev is None or len(prev) == 0:
            self.client_sensitivities[client_id] = dict(sensitivities)
            return

        beta = float(getattr(self, "sensitivity_momentum", 0.2))
        eps = 1e-12
        merged: Dict[str, float] = {}

        keys = set(prev.keys()) | set(sensitivities.keys()) | set(self.layer_names)
        for k in keys:
            old_v = float(prev.get(k, 0.0) or 0.0)
            new_v = float(sensitivities.get(k, 0.0) or 0.0)
            if new_v <= eps:
                merged[k] = old_v
            elif old_v <= eps:
                merged[k] = new_v
            else:
                merged[k] = beta * old_v + (1.0 - beta) * new_v

        self.client_sensitivities[client_id] = merged

    def _get_layer_depth(self, layer_name: str) -> int:
        if "blocks." in layer_name:
            parts = layer_name.split("blocks.", 1)[1]
            idx_str = parts.split(".", 1)[0]
            if idx_str.isdigit():
                return int(idx_str)
        return 0

    def _get_structure_weight(self) -> float:
        t = max(0, self.current_round - 1)
        w = 1.0 - (t / float(self.struct_to_data_rounds))
        if w < 0.0:
            return 0.0
        if w > 1.0:
            return 1.0
        return w
    
    def _z_score_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Z-Score 归一化：将原始分数转换为标准正态分布，放大微小差异
        """
        if scores.size == 0:
            return scores
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return np.zeros_like(scores)
        return (scores - mean) / std

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """
        Softmax 将分数转换为概率分布（Sum = 1）
        """
        if scores.size == 0:
            return scores
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)  # 防止数值溢出
        sum_exp = np.sum(exp_scores)
        if sum_exp == 0:
            return np.ones_like(scores) / scores.size
        return exp_scores / sum_exp
    
    def _softmax(self, scores: np.ndarray, temperature: Optional[float] = None) -> np.ndarray:
        if scores.size == 0:
            return scores
        tau = self.allocation_temperature if temperature is None else max(1e-6, float(temperature))
        scaled_scores = scores / tau
        max_score = np.max(scaled_scores)
        exp_scores = np.exp(scaled_scores - max_score)
        sum_exp = np.sum(exp_scores)
        if sum_exp == 0:
            return np.ones_like(scores) / scores.size
        return exp_scores / sum_exp

    def _group_rms_score(self, scores: List[float], default: float = 1e-12) -> float:
        vals = [max(float(v or 0.0), 0.0) for v in scores if float(v or 0.0) > 0.0]
        if len(vals) == 0:
            return float(default)
        arr = np.array(vals, dtype=np.float64)
        return float(np.sqrt(np.mean(np.square(arr))))

    def _normalize_positive(self, values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        if values.size == 0:
            return values
        vals = np.asarray(values, dtype=np.float64)
        vals = np.maximum(vals, float(eps))
        total = float(np.sum(vals))
        if not np.isfinite(total) or total <= 0.0:
            return np.ones_like(vals, dtype=np.float64) / float(max(1, vals.size))
        return vals / total

    def _effective_sample_weight(self, client_id: int, client_sample_counts: Dict[int, int], use_sample_weights: bool) -> float:
        if not use_sample_weights:
            return 1.0
        n_k = max(1.0, float(client_sample_counts.get(client_id, 1.0)))
        return float(n_k ** float(getattr(self, "sample_weight_power", 0.5)))

    def _record_allocation_coverage(self, allocation_map: Dict[int, Set[str]]):
        all_layers = set(self.layer_names)
        for client_id in range(self.num_clients):
            allocated = set(allocation_map.get(client_id, set()))
            self.client_allocation_history.setdefault(client_id, []).append(set(allocated))
            if client_id not in self.client_first_allocation:
                self.client_first_allocation[client_id] = set(allocated)
            first_alloc = self.client_first_allocation.get(client_id, set())
            first_missing = all_layers.difference(first_alloc)
            unique_alloc = self.client_unique_allocated_layers.setdefault(client_id, set())
            first_new_round = self.client_first_new_layer_round.setdefault(client_id, {})
            for layer_name in allocated:
                if layer_name in first_missing and layer_name not in first_new_round:
                    first_new_round[layer_name] = int(self.current_round)
            unique_alloc.update(allocated)

    def get_underallocated_coverage_stats(self) -> List[Dict[str, object]]:
        all_layers = set(self.layer_names)
        total_layers = len(all_layers)
        stats: List[Dict[str, object]] = []
        for client_id in range(self.num_clients):
            budget = int(self.client_layer_counts.get(client_id, max(1, int(self.num_layers * self.allocation_ratio))))
            budget = max(0, min(budget, total_layers))
            if total_layers <= 0 or budget >= total_layers:
                continue

            first_alloc = set(self.client_first_allocation.get(client_id, set()))
            unique_alloc = set(self.client_unique_allocated_layers.get(client_id, set()))
            first_missing = all_layers.difference(first_alloc)
            first_new_round = dict(self.client_first_new_layer_round.get(client_id, {}))
            covered_initial_missing = first_missing.intersection(unique_alloc)
            remaining_initial_missing = first_missing.difference(unique_alloc)
            first_new_round_values = [int(v) for v in first_new_round.values()]
            full_cover_round = None
            if len(first_missing) > 0 and len(remaining_initial_missing) == 0:
                full_cover_round = max(first_new_round_values) if len(first_new_round_values) > 0 else 1

            stats.append(
                {
                    "client": int(client_id),
                    "budget_layers": int(budget),
                    "total_layers": int(total_layers),
                    "initial_layers": int(len(first_alloc)),
                    "initial_missing_layers": int(len(first_missing)),
                    "unique_layers": int(len(unique_alloc)),
                    "new_layers_after_round1": int(len(unique_alloc.difference(first_alloc))),
                    "initial_missing_covered": int(len(covered_initial_missing)),
                    "initial_missing_remaining": int(len(remaining_initial_missing)),
                    "first_new_layer_round": min(first_new_round_values) if len(first_new_round_values) > 0 else None,
                    "full_initial_missing_cover_round": full_cover_round,
                    "observed_rounds": int(len(self.client_allocation_history.get(client_id, []))),
                }
            )
        return stats

    def uses_partial_layer_allocation(self) -> bool:
        """Return whether the active method selects per-client layer subsets."""
        return True

    def print_underallocated_coverage_stats(self, max_clients: int = 12):
        stats = self.get_underallocated_coverage_stats()
        print("\n[总结] 非满配客户端层覆盖统计")
        print("-" * 80)
        print(
            f"概率抽样参数: rounds={self.current_round}, "
            f"coverage_prior_weight={self.coverage_prior_weight:.2f}, "
            f"struct_to_data_rounds={self.struct_to_data_rounds}, "
            f"allocation_temperature={self.allocation_temperature:.2f}"
        )
        if len(stats) == 0:
            print("没有非满配客户端，所有客户端都具备全层训练预算。")
            print("-" * 80 + "\n")
            return

        completed = [s for s in stats if s["full_initial_missing_cover_round"] is not None]
        remaining_clients = [s for s in stats if int(s["initial_missing_remaining"]) > 0]
        avg_budget = float(np.mean([int(s["budget_layers"]) for s in stats]))
        avg_unique = float(np.mean([int(s["unique_layers"]) for s in stats]))
        avg_covered = float(np.mean([int(s["initial_missing_covered"]) for s in stats]))
        avg_missing = float(np.mean([int(s["initial_missing_layers"]) for s in stats]))
        print(
            f"非满配客户端: {len(stats)} 个 | 平均预算层数={avg_budget:.2f} | "
            f"平均累计覆盖层数={avg_unique:.2f}"
        )
        print(
            f"第一轮未选层: 平均={avg_missing:.2f} | "
            f"后续已补到={avg_covered:.2f} | "
            f"已全部补齐客户端={len(completed)}/{len(stats)}"
        )
        if len(completed) > 0:
            cover_rounds = [int(s["full_initial_missing_cover_round"]) for s in completed]
            print(
                f"补齐第一轮未选层所需轮数: 平均={float(np.mean(cover_rounds)):.2f}, "
                f"最大={int(np.max(cover_rounds))}"
            )
        if len(remaining_clients) > 0:
            total_remaining = int(sum(int(s["initial_missing_remaining"]) for s in remaining_clients))
            print(
                f"当前仍未补齐: {len(remaining_clients)} 个客户端，共剩余 {total_remaining} 个第一轮未选层。"
            )
            print(
                "若希望概率抽样更快覆盖最初没选到的层，可适当增大 "
                "--allocation_temperature 或 --coverage_prior_weight，并确保 --struct_to_data_rounds 和 --num_rounds "
                "足够长；当 structure_weight 衰减为 0 后，coverage prior 就不再主导选择。"
            )

        shown = 0
        for s in stats:
            if shown >= max_clients:
                break
            print(
                "  client={client}: budget={budget_layers}/{total_layers}, "
                "round1={initial_layers}, unique={unique_layers}, "
                "covered_missing={initial_missing_covered}/{initial_missing_layers}, "
                "remaining={initial_missing_remaining}, first_new_round={first_new_layer_round}, "
                "full_cover_round={full_initial_missing_cover_round}".format(**s)
            )
            shown += 1
        if len(stats) > shown:
            print(f"  ... 其余 {len(stats) - shown} 个非满配客户端略。")
        print("-" * 80 + "\n")

    def generate_allocation_map(self) -> Dict[int, Set[str]]:
        """
        生成当前轮次的分配映射 (Heterogeneous Allocation)。
        
        策略:
        1. 全局先验 (早期占主导): U-shaped 结构先验 + 历史覆盖校正。
        2. 客户端反馈 (随轮次增大): 客户端侧 B 梯度范数形成的敏感度分布。
        3. 随着轮数增加，敏感度反馈比例上升；初始更依赖全局先验。
        4. 使用归一化分布和 Product-of-Experts 得到融合概率，并无放回抽样层组。
        """
        method = str(self.method).lower()

        suffixes = (
            ".qkv",
            ".query",
            ".key",
            ".value",
            ".q_proj",
            ".k_proj",
            ".v_proj",
            ".o_proj",
            ".out_proj",
            ".c_attn",
            ".c_proj",
        )
        group_to_layers: Dict[str, List[str]] = {}
        for name in self.layer_names:
            group_id = name
            for suf in suffixes:
                if name.endswith(suf):
                    group_id = name[: -len(suf)]
                    break
            group_to_layers.setdefault(group_id, []).append(name)
        group_ids = sorted(group_to_layers.keys(), key=lambda g: (self._get_layer_depth(str(g)), str(g)))

        allocation_map = {}
        structure_weight = self._get_structure_weight()
        data_weight = 1.0 - structure_weight

        # 1) 计算全局结构分数（所有客户端共享）
        struct_scores = []
        struct_scores = []
        coverage_scores = []
        max_depth = max(self._get_layer_depth(gid) for gid in group_ids) if len(group_ids) > 0 else 0
        for gid in group_ids:
            member_counts = [self.layer_selection_counts.get(n, 0) for n in group_to_layers.get(gid, [])]
            count = float(np.mean(member_counts)) if len(member_counts) > 0 else 0.0
            depth = self._get_layer_depth(gid)
            theta = 0.9
            # 让靠近输入（depth 小）和靠近输出（depth 大）的层获得更高权重
            # 使用倒 V 形：proximity_weight = 1 - |2*normalized_depth - 1| ^ gamma
            gamma = 0.5  # 控制“倒 V”陡峭程度，越小越陡
            if max_depth <= 0:
                pos = 0.0
            else:
                pos = 2.0 * float(depth) / float(max_depth) - 1.0
            struct_scores.append((1.0 - theta) + theta * (pos ** 2))
            coverage_scores.append((count + 1e-12) ** (-0.5))
        struct_scores = np.array(struct_scores, dtype=np.float64)
        coverage_scores = np.array(coverage_scores, dtype=np.float64)
        struct_probs = self._normalize_positive(struct_scores)
        coverage_probs = self._normalize_positive(coverage_scores)
        # Z-Score 归一化后 Softmax 成概率
        struct_probs = self._softmax(self._z_score_normalize(struct_scores))

        coverage_weight = float(getattr(self, "coverage_prior_weight", 0.0))
        if coverage_weight <= 0.0:
            global_probs = struct_probs
        else:
            log_global = (
                (1.0 - coverage_weight) * np.log(struct_probs + 1e-20)
                + coverage_weight * np.log(coverage_probs + 1e-20)
            )
            global_probs = np.exp(log_global - np.max(log_global))
            global_probs = self._normalize_positive(global_probs)

        for client_id in range(self.num_clients):
            client_layer_budget = self.client_layer_counts.get(
                client_id,
                max(1, int(self.num_layers * self.allocation_ratio))
            )
            if len(group_ids) == 0:
                allocation_map[client_id] = set()
                continue

            denom = float(len(self.layer_names)) if len(self.layer_names) > 0 else 1.0
            ratio = float(client_layer_budget) / denom
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0
            client_num_allocated = max(1, int(len(group_ids) * ratio))
            client_num_allocated = min(client_num_allocated, len(group_ids))

            client_sense = self.client_sensitivities.get(client_id, {})

            # 2) 计算该客户端的数据分数
            data_scores = []
            data_scores = []
            for gid in group_ids:
                member_scores = []
                for n in group_to_layers.get(gid, []):
                    score = float(client_sense.get(n, 0.0))
                    if score <= 0.0:
                        score = 1e-12
                    member_scores.append(score)
                data_scores.append(self._group_rms_score(member_scores))
            data_scores = np.array(data_scores, dtype=np.float64)
            data_probs = self._softmax(np.log(np.maximum(data_scores, 1e-12)))
            # Z-Score 归一化后 Softmax 成概率
            data_probs = self._softmax(self._z_score_normalize(data_scores))

            # 3) 融合全局先验与客户端数据分数
            log_final = (
                structure_weight * np.log(global_probs + 1e-20)
                + data_weight * np.log(data_probs + 1e-20)
            )
            final_probs = np.exp(log_final - np.max(log_final))
            final_probs = self._normalize_positive(final_probs)

            # 4) 按融合分数无放回抽样
            selected_group_ids = np.random.choice(
                group_ids,
                size=client_num_allocated,
                replace=False,
                p=final_probs,
            ).tolist()
            allocated_layers: Set[str] = set()
            for gid in selected_group_ids:
                allocated_layers.update(group_to_layers.get(str(gid), []))
            allocation_map[client_id] = allocated_layers

            # 5) 更新全局选中计数
            for name in allocated_layers:
                self.layer_selection_counts[name] += 1

        return allocation_map
               
    
    def update_staleness(self, allocation_map: Dict[int, Set[str]]):
        """
        更新陈旧度计数器。
        如果某层本轮被分配，陈旧度重置为 0；否则陈旧度 +1。
        """
        for client_id in range(self.num_clients):
            allocated_layers = allocation_map.get(client_id, set())
            
            for layer_name in self.layer_names:
                if layer_name in allocated_layers:
                    # 客户端本轮更新了该层，重置陈旧度
                    self.staleness[client_id][layer_name] = 0
                else:
                    # 客户端本轮未更新该层，陈旧度增加
                    self.staleness[client_id][layer_name] += 1
    
    def get_staleness_dampening(self, client_id: int, layer_name: str) -> float:
        """
        计算陈旧度阻尼因子: α = 1 / sqrt(1 + τ)
        """
        tau = self.staleness[client_id][layer_name]
        return 1.0 / np.sqrt(1.0 + tau)

    def aggregate_matrices(
        self,
        client_updates_A: Dict[int, Dict[str, torch.Tensor]],
        client_updates_B: Optional[Dict[int, Dict[str, torch.Tensor]]],
        client_sample_counts: Dict[int, int],
        allocation_map: Dict[int, Set[str]]
    ):
        """
        聚合客户端提交的 A/B 矩阵更新。
        
        修正后的聚合逻辑 (Layer-wise Weighted Average):
        对于每一层 j:
            Delta_j = (Sum_{k in Active} n_k * alpha_k * Delta_{k,j}) / (Sum_{k in Active} n_k)
        
        参数:
            client_updates_A: 客户端 ID 到 delta_A 的映射
            client_updates_B: 客户端 ID 到 delta_B 的映射（可选）
            client_sample_counts: 客户端 ID 到其样本数量的映射 (用于加权)
            allocation_map: 当前轮次的分配映射
        """
        method = str(self.method).lower()
        matrix_mode = str(getattr(self, "fedharp_matrix_mode", "a")).lower()
        aggregate_B = matrix_mode in {"b", "ab", "adaptive"}
        use_staleness = True
        use_sample_weights = True

        weighted_updates_sum_A = {
            layer_name: torch.zeros_like(self.lora_layers[layer_name].get_A())
            for layer_name in self.layer_names
        }
        total_weights_A = {layer_name: 0.0 for layer_name in self.layer_names}

        weighted_updates_sum_B = None
        total_weights_B = None
        if aggregate_B:
            if client_updates_B is None:
                raise ValueError(f"{self.method} 需要 client_updates_B")
            weighted_updates_sum_B = {
                layer_name: torch.zeros_like(self.lora_layers[layer_name].get_B())
                for layer_name in self.layer_names
            }
            total_weights_B = {layer_name: 0.0 for layer_name in self.layer_names}

        for client_id, delta_A in client_updates_A.items():
            allocated_layers = allocation_map.get(client_id, set())
            n_k = self._effective_sample_weight(client_id, client_sample_counts, use_sample_weights)

            for layer_name, delta in delta_A.items():
                if layer_name not in allocated_layers:
                    continue
                alpha = self.get_staleness_dampening(client_id, layer_name) if use_staleness else 1.0
                weighted_updates_sum_A[layer_name] += (n_k * alpha) * delta
                total_weights_A[layer_name] += (n_k * alpha) if use_staleness else n_k

            if aggregate_B:
                b_map = client_updates_B.get(client_id, {})
                for layer_name, delta in b_map.items():
                    if layer_name not in allocated_layers:
                        continue
                    alpha = self.get_staleness_dampening(client_id, layer_name) if use_staleness else 1.0
                    weighted_updates_sum_B[layer_name] += (n_k * alpha) * delta
                    total_weights_B[layer_name] += (n_k * alpha) if use_staleness else n_k

        for layer_name, update_sum in weighted_updates_sum_A.items():
            denominator = total_weights_A[layer_name]
            if denominator <= 0:
                continue
            avg_update = update_sum / denominator
            current_A = self.lora_layers[layer_name].get_A()
            new_A = current_A + self.aggregation_lr * avg_update
            self.lora_layers[layer_name].set_A(new_A)

        if aggregate_B and weighted_updates_sum_B is not None and total_weights_B is not None:
            for layer_name, update_sum in weighted_updates_sum_B.items():
                denominator = total_weights_B[layer_name]
                if denominator <= 0:
                    continue
                avg_update = update_sum / denominator
                current_B = self.lora_layers[layer_name].get_B()
                new_B = current_B + self.aggregation_lr * avg_update
                self.lora_layers[layer_name].set_B(new_B)

    def aggregate_A_matrices(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_sample_counts: Dict[int, int],
        allocation_map: Dict[int, Set[str]]
    ):
        return self.aggregate_matrices(
            client_updates_A=client_updates,
            client_updates_B=None,
            client_sample_counts=client_sample_counts,
            allocation_map=allocation_map
        )
    
    def get_global_A_matrices(self) -> Dict[str, torch.Tensor]:
        """获取所有全局 A 矩阵"""
        return {name: layer.get_A() for name, layer in self.lora_layers.items()}

    def get_global_B_matrices(self) -> Dict[str, torch.Tensor]:
        """获取所有全局 B 矩阵"""
        return {name: layer.get_B() for name, layer in self.lora_layers.items()}
    
    def set_global_A_matrices(self, A_matrices: Dict[str, torch.Tensor]):
        """设置全局 A 矩阵 (用于初始化)"""
        for name, A in A_matrices.items():
            if name in self.lora_layers:
                self.lora_layers[name].set_A(A)

    def set_global_B_matrices(self, B_matrices: Dict[str, torch.Tensor]):
        """设置全局 B 矩阵 (用于初始化)"""
        for name, B in B_matrices.items():
            if name in self.lora_layers:
                self.lora_layers[name].set_B(B)
    
    def get_staleness_stats(self) -> Dict[str, float]:
        """获取陈旧度统计信息"""
        all_staleness = []
        for client_staleness in self.staleness.values():
            all_staleness.extend(client_staleness.values())
        
        if len(all_staleness) == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0}
        
        all_staleness = np.array(all_staleness)
        return {
            'mean': float(np.mean(all_staleness)),
            'std': float(np.std(all_staleness)),
            'max': int(np.max(all_staleness))
        }
    
    def print_allocation_stats(self, allocation_map: Dict[int, Set[str]]):
        """打印本轮分配的统计信息"""
        selection_mode = "融合分数概率抽样（无放回）"

        print(f"\n[步骤 1] 第 {self.current_round} 轮层分配")
        print("-" * 60)
        print(f"方法: {self.method} | 层选择策略: {selection_mode}")

        # 统计每层被多少客户端选中
        layer_client_count = {layer_name: 0 for layer_name in self.layer_names}
        for client_id, allocated_layers in allocation_map.items():
            for layer_name in allocated_layers:
                layer_client_count[layer_name] += 1
        
        # 打印陈旧度统计
        staleness_stats = self.get_staleness_stats()
        print(f"本轮层覆盖: 最小={min(layer_client_count.values())}, 最大={max(layer_client_count.values())}")
        print(f"陈旧度: 均值={staleness_stats['mean']:.2f}, 最大={staleness_stats['max']}")
        
        # 打印历史覆盖/选中次数统计
        print("历史选中次数 Top-5:")
        sorted_counts = sorted(self.layer_selection_counts.items(), key=lambda x: x[1], reverse=True)
        for name, count in sorted_counts[:5]:
            coverage_score = 1.0 / (1.0 + float(count))
            print(f"  {name}: {count} 次 (覆盖校正分数: {coverage_score:.4f})")

        structure_weight = self._get_structure_weight()
        data_weight = 1.0 - structure_weight
        if self.client_sensitivities:
            print(
                f"融合权重: 全局先验={structure_weight:.2f}, 敏感度反馈={data_weight:.2f} "
                f"(已收集 {len(self.client_sensitivities)} 个客户端反馈)"
            )
        else:
            print(f"融合权重: 全局先验={structure_weight:.2f}, 敏感度反馈={data_weight:.2f}")
            
        print("-" * 60 + "\n")
    
    def start_round(self) -> Dict[int, Set[str]]:
        """开始新的一轮"""
        self.current_round += 1
        allocation_map = self.generate_allocation_map()
        self._record_allocation_coverage(allocation_map)
        # Advance staleness only after current-round aggregation; LPC and the
        # staleness weight must see the pre-round inactivity counter.
        return allocation_map
