"""
FedHARP 的自定义 LoRA 实现，适配 Vision Transformer (ViT)。
核心在于实现了 A（共享）和 B（本地/私有）矩阵的分离管理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
from typing import Optional, List, Dict
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("警告: 未找到 timm 库，将使用 torchvision 的 ViT 实现")

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def _unfreeze_classifier_params(model: nn.Module):
    classifier = None
    if hasattr(model, "get_classifier"):
        try:
            classifier = model.get_classifier()
        except Exception:
            classifier = None
    if classifier is None:
        for attr in ("head", "classifier", "fc"):
            if hasattr(model, attr):
                classifier = getattr(model, attr)
                break
    if classifier is None and hasattr(model, "heads"):
        heads = getattr(model, "heads")
        if hasattr(heads, "head"):
            classifier = heads.head
    if isinstance(classifier, nn.Module):
        for p in classifier.parameters():
            p.requires_grad = True


def _select_checkpoint_file(snapshot_dir: str, checkpoint_file: Optional[str]) -> str:
    if checkpoint_file:
        ckpt = os.path.join(snapshot_dir, checkpoint_file)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"未找到 checkpoint 文件: {ckpt}")
        return ckpt

    candidates: List[str] = []
    for root, _, files in os.walk(snapshot_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith((".safetensors", ".pth", ".pt", ".bin")):
                candidates.append(os.path.join(root, fn))

    if len(candidates) == 0:
        raise FileNotFoundError(f"未在 ModelScope snapshot 目录中找到可用 checkpoint: {snapshot_dir}")

    candidates.sort()
    safetensors = [p for p in candidates if p.lower().endswith(".safetensors")]
    if len(safetensors) == 1:
        return safetensors[0]
    if len(candidates) == 1:
        return candidates[0]

    msg = "ModelScope snapshot 目录存在多个 checkpoint 候选文件，请用 --ms_checkpoint_file 指定其一:\n"
    msg += "\n".join([os.path.relpath(p, snapshot_dir) for p in candidates[:50]])
    if len(candidates) > 50:
        msg += f"\n... (共 {len(candidates)} 个候选)"
    raise ValueError(msg)


def _load_state_dict_from_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    low = checkpoint_path.lower()
    if low.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise ImportError(f"加载 .safetensors 需要安装 safetensors: {e}") from e
        state = load_file(checkpoint_path, device="cpu")
        return dict(state)

    try:
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "params"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"无法从 checkpoint 解析 state_dict: {checkpoint_path}")
    return obj


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if len(state_dict) == 0:
        return state_dict
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        out[k] = v
    return out


def _looks_like_transformers_vit_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    arch = cfg.get("architectures")
    if not isinstance(arch, list) or len(arch) == 0:
        return False
    if not any(isinstance(a, str) and a.startswith("ViT") for a in arch):
        return False
    for fn in ("pytorch_model.bin", "model.safetensors"):
        if os.path.isfile(os.path.join(path, fn)):
            return True
    return False


class _TransformersImageClassifierWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        logits = getattr(out, "logits", None)
        if logits is None:
            raise ValueError("Transformers ViT 输出不包含 logits")
        return logits

    def get_classifier(self):
        return getattr(self.backbone, "classifier", None)


class LoRALinear(nn.Module):
    """
    自定义 LoRA 线性层，公式为 W = W_0 + B @ A。
    
    FedHARP 的核心特性:
    - 矩阵 A: 全局共享，由服务器聚合 (初始化为高斯分布)
    - 矩阵 B: 原生隐私 (Native-privacy)，永远保留在本地，不上传 (初始化为零)
    - 基础权重 W_0: 冻结，不更新
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        output_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.init_rank = int(rank)
        self.scaling_rank = int(rank)
        self.alpha = alpha
        
        # 基础权重 (冻结，不可训练)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        # LoRA 矩阵
        # A: 共享矩阵 (服务器聚合) - 使用高斯初始化
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        
        # B: 个性化矩阵 (本地保留) - 使用全零初始化
        # 这种初始化保证了训练开始时 LoRA 分支输出为 0，不影响原模型
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if output_mask is None:
            self.output_mask = None
        else:
            self.register_buffer("output_mask", output_mask.clone())
        
        # 跟踪哪些矩阵当前是可训练的 (用于部分层冻结策略)
        self.A_trainable = True
        self.B_trainable = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: y = x @ W_0^T + x @ A^T @ B^T * scaling
        """
        # 基础权重路径
        base_output = F.linear(x, self.base_weight, getattr(self, "bias", None))
        
        # LoRA 路径
        # 先乘 A，再乘 B
        current_rank = int(self.lora_A.shape[0])
        scaling_rank = int(getattr(self, "scaling_rank", self.init_rank))
        scaling = (self.alpha / float(scaling_rank)) if current_rank > 0 and scaling_rank > 0 else 0.0
        lora_output = F.linear(
            F.linear(x, self.lora_A),
            self.lora_B
        ) * scaling

        if self.output_mask is not None:
            mask = self.output_mask
            view_shape = [1] * (lora_output.ndim - 1) + [-1]
            lora_output = lora_output * mask.view(*view_shape)

        return base_output + self.dropout(lora_output)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key_a = prefix + "lora_A"
        key_b = prefix + "lora_B"
        if key_a in state_dict:
            A = state_dict[key_a]
            if tuple(A.shape) != tuple(self.lora_A.shape):
                self.lora_A = nn.Parameter(A.clone(), requires_grad=bool(self.lora_A.requires_grad))
        if key_b in state_dict:
            B = state_dict[key_b]
            if tuple(B.shape) != tuple(self.lora_B.shape):
                self.lora_B = nn.Parameter(B.clone(), requires_grad=bool(self.lora_B.requires_grad))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    
    def get_A(self) -> torch.Tensor:
        """获取共享矩阵 A (用于上传给服务器)"""
        return self.lora_A.data.clone()
    
    def set_A(self, A: torch.Tensor):
        """设置共享矩阵 A (从服务器接收)"""
        A = A.to(device=self.lora_A.device, dtype=self.lora_A.dtype).clone()
        if tuple(A.shape) != tuple(self.lora_A.shape):
            requires_grad = bool(self.lora_A.requires_grad)
            self.lora_A = nn.Parameter(A, requires_grad=requires_grad)
        else:
            self.lora_A.data = A
        # Rank-changing configurations must evaluate the
        # received adapter with the scaling associated with its actual rank.
        current_rank = int(A.shape[0])
        self.rank = current_rank
        self.scaling_rank = current_rank
    
    def get_B(self) -> torch.Tensor:
        """获取个性化矩阵 B (仅用于本地持久化)"""
        return self.lora_B.data.clone()
    
    def set_B(self, B: torch.Tensor):
        """设置个性化矩阵 B"""
        B = B.to(device=self.lora_B.device, dtype=self.lora_B.dtype).clone()
        if tuple(B.shape) != tuple(self.lora_B.shape):
            requires_grad = bool(self.lora_B.requires_grad)
            self.lora_B = nn.Parameter(B, requires_grad=requires_grad)
        else:
            self.lora_B.data = B

    def reset_lora(self, rank: Optional[int] = None):
        target_rank = int(rank if rank is not None else self.init_rank)
        device = self.base_weight.device
        dtype = self.base_weight.dtype

        A = torch.randn(target_rank, self.in_features, device=device, dtype=dtype) * 0.02
        B = torch.zeros(self.out_features, target_rank, device=device, dtype=dtype)

        A_requires_grad = bool(self.lora_A.requires_grad)
        B_requires_grad = bool(self.lora_B.requires_grad)
        self.lora_A = nn.Parameter(A, requires_grad=A_requires_grad)
        self.lora_B = nn.Parameter(B, requires_grad=B_requires_grad)
        self.scaling_rank = int(target_rank)

    def merge_lora_weights(self):
        current_rank = int(self.lora_A.shape[0])
        if current_rank <= 0:
            return
        scaling_rank = int(getattr(self, "scaling_rank", self.init_rank))
        scaling = self.alpha / float(scaling_rank) if scaling_rank > 0 else 0.0
        delta_w = (self.lora_B @ self.lora_A) * scaling
        self.base_weight.data = self.base_weight.data + delta_w.to(
            device=self.base_weight.device, dtype=self.base_weight.dtype
        )

    def merge_lora_into_base_and_reset(self, rank: Optional[int] = None):
        self.merge_lora_weights()
        self.reset_lora(rank=rank)
    
    def freeze_A(self):
        """冻结矩阵 A (用于预热阶段)"""
        self.lora_A.requires_grad = False
        self.A_trainable = False
    
    def unfreeze_A(self):
        """解冻矩阵 A"""
        self.lora_A.requires_grad = True
        self.A_trainable = True
    
    def freeze_B(self):
        """冻结矩阵 B"""
        self.lora_B.requires_grad = False
        self.B_trainable = False
    
    def unfreeze_B(self):
        """解冻矩阵 B (用于预热阶段和正常训练)"""
        self.lora_B.requires_grad = True
        self.B_trainable = True
    
    def freeze_all(self):
        """冻结 A 和 B (用于未被分配的层)"""
        self.freeze_A()
        self.freeze_B()
    
    def unfreeze_all(self):
        """解冻 A 和 B (用于正常联合训练)"""
        self.unfreeze_A()
        self.unfreeze_B()
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """根据当前状态获取可训练参数列表"""
        params = []
        if self.A_trainable:
            params.append(self.lora_A)
        if self.B_trainable:
            params.append(self.lora_B)
        return params


class PeftLoRALinearAdapter:
    def __init__(self, peft_linear_module: nn.Module, adapter_name: str = "default"):
        self._m = peft_linear_module
        self._adapter = adapter_name
        self.init_rank = int(self.lora_A.shape[0])
        self.rank = int(self.init_rank)
        self.scaling_rank = int(self.init_rank)
        self.alpha = float(self._get_lora_alpha())

    def _get_lora_alpha(self) -> float:
        alpha_map = getattr(self._m, "lora_alpha", None)
        if alpha_map is not None:
            try:
                return float(alpha_map[self._adapter])
            except Exception:
                pass
        scaling_map = getattr(self._m, "scaling", None)
        if scaling_map is not None:
            try:
                return float(scaling_map[self._adapter]) * float(max(1, self.lora_A.shape[0]))
            except Exception:
                pass
        return float(max(1, self.lora_A.shape[0]))

    def _update_rank_metadata(self, rank: int):
        rank = max(1, int(rank))
        self.rank = rank
        self.scaling_rank = rank
        rank_map = getattr(self._m, "r", None)
        if rank_map is not None:
            try:
                rank_map[self._adapter] = rank
            except Exception:
                pass
        alpha_map = getattr(self._m, "lora_alpha", None)
        if alpha_map is not None:
            try:
                alpha_map[self._adapter] = float(self.alpha)
            except Exception:
                pass
        scaling_map = getattr(self._m, "scaling", None)
        if scaling_map is not None:
            try:
                scaling_map[self._adapter] = float(self.alpha) / float(rank)
            except Exception:
                pass

    def _replace_projection(self, attr: str, weight: torch.Tensor):
        modules = getattr(self._m, attr)
        old = modules[self._adapter]
        requires_grad = bool(old.weight.requires_grad)
        weight = weight.to(device=old.weight.device, dtype=old.weight.dtype).clone()
        new_layer = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=False).to(
            device=weight.device,
            dtype=weight.dtype,
        )
        new_layer.weight.data.copy_(weight)
        new_layer.weight.requires_grad = requires_grad
        modules[self._adapter] = new_layer

    @property
    def lora_A(self) -> torch.nn.Parameter:
        return self._m.lora_A[self._adapter].weight

    @property
    def lora_B(self) -> torch.nn.Parameter:
        return self._m.lora_B[self._adapter].weight

    def get_A(self) -> torch.Tensor:
        return self.lora_A.data.clone()

    def set_A(self, A: torch.Tensor):
        A = A.to(device=self.lora_A.device, dtype=self.lora_A.dtype)
        if tuple(A.shape) != tuple(self.lora_A.shape):
            self._replace_projection("lora_A", A)
        else:
            self.lora_A.data.copy_(A)
        self._update_rank_metadata(int(A.shape[0]))

    def get_B(self) -> torch.Tensor:
        return self.lora_B.data.clone()

    def set_B(self, B: torch.Tensor):
        B = B.to(device=self.lora_B.device, dtype=self.lora_B.dtype)
        if tuple(B.shape) != tuple(self.lora_B.shape):
            self._replace_projection("lora_B", B)
        else:
            self.lora_B.data.copy_(B)
        self._update_rank_metadata(int(B.shape[1]))

    def reset_lora(self, rank: Optional[int] = None):
        target_rank = max(1, int(rank if rank is not None else self.init_rank))
        old_A = self.lora_A
        old_B = self.lora_B
        A = torch.empty(
            target_rank,
            int(old_A.shape[1]),
            device=old_A.device,
            dtype=old_A.dtype,
        )
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        B = torch.zeros(
            int(old_B.shape[0]),
            target_rank,
            device=old_B.device,
            dtype=old_B.dtype,
        )
        self.set_A(A)
        self.set_B(B)
        self._update_rank_metadata(target_rank)

    def merge_lora_weights(self):
        base_layer = None
        if hasattr(self._m, "get_base_layer"):
            try:
                base_layer = self._m.get_base_layer()
            except Exception:
                base_layer = None
        if base_layer is None:
            base_layer = getattr(self._m, "base_layer", None)
        if base_layer is None or not hasattr(base_layer, "weight"):
            raise ValueError("PEFT LoRA module does not expose a mergeable base weight")

        rank = max(1, int(self.lora_A.shape[0]))
        scaling_map = getattr(self._m, "scaling", None)
        try:
            scaling = float(scaling_map[self._adapter])
        except Exception:
            scaling = float(self.alpha) / float(rank)
        delta = torch.matmul(self.lora_B.data, self.lora_A.data) * scaling
        if bool(getattr(self._m, "fan_in_fan_out", False)):
            delta = delta.transpose(0, 1)
        base_layer.weight.data.add_(
            delta.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        )

    def freeze_A(self):
        self.lora_A.requires_grad = False

    def unfreeze_A(self):
        self.lora_A.requires_grad = True

    def freeze_B(self):
        self.lora_B.requires_grad = False

    def unfreeze_B(self):
        self.lora_B.requires_grad = True

    def freeze_all(self):
        self.freeze_A()
        self.freeze_B()

    def unfreeze_all(self):
        self.unfreeze_A()
        self.unfreeze_B()

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        if self.lora_A.requires_grad:
            params.append(self.lora_A)
        if self.lora_B.requires_grad:
            params.append(self.lora_B)
        return params


def _extract_peft_lora_layers(model: nn.Module, adapter_name: str = "default") -> Dict[str, PeftLoRALinearAdapter]:
    lora_layers: Dict[str, PeftLoRALinearAdapter] = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        try:
            _ = module.lora_A[adapter_name].weight
            _ = module.lora_B[adapter_name].weight
        except Exception:
            continue
        lora_layers[name] = PeftLoRALinearAdapter(module, adapter_name=adapter_name)
    return lora_layers


def _patch_peft_qkv_qv_only(peft_linear_module: nn.Module):
    if getattr(peft_linear_module, "_qv_only_patched", False):
        return
    base_layer = getattr(peft_linear_module, "base_layer", None)
    if base_layer is None or not hasattr(base_layer, "out_features"):
        return
    out_features = int(base_layer.out_features)
    if out_features % 3 != 0:
        return

    d = out_features // 3
    mask = torch.ones(out_features, dtype=torch.float32)
    mask[d:2 * d] = 0.0
    peft_linear_module.register_buffer("qv_output_mask", mask)
    orig_forward = peft_linear_module.forward

    def forward(x: torch.Tensor, *args, **kwargs):
        full_out = orig_forward(x, *args, **kwargs)
        base_out = base_layer(x)
        lora_out = full_out - base_out
        m = peft_linear_module.qv_output_mask.to(dtype=lora_out.dtype, device=lora_out.device)
        view_shape = [1] * (lora_out.ndim - 1) + [-1]
        return base_out + (lora_out * m.view(*view_shape))

    peft_linear_module.forward = forward
    peft_linear_module._qv_only_patched = True


def _discover_vit_lora_targets(model: nn.Module, use_transformers_dir: bool = False) -> List[str]:
    """Pick attention projection names that actually exist in the loaded ViT."""
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_names.append(str(name).lower())

    suffixes = {n.rsplit(".", 1)[-1] for n in linear_names}
    if "qkv" in suffixes:
        return ["qv"]
    if "q_proj" in suffixes or "v_proj" in suffixes:
        targets = []
        if "q_proj" in suffixes:
            targets.append("q_proj")
        if "v_proj" in suffixes:
            targets.append("v_proj")
        return targets
    if "query" in suffixes or "value" in suffixes:
        targets = []
        if "query" in suffixes:
            targets.append("query")
        if "value" in suffixes:
            targets.append("value")
        return targets
    if any(n.endswith(".out_proj") and ("attn" in n or "attention" in n) for n in linear_names):
        return ["out_proj"]
    if any(n.endswith(".c_attn") for n in linear_names):
        return ["c_attn"]

    return ["qv"]


def inject_lora_to_vit(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0
) -> Dict[str, LoRALinear]:
    """
    向 Vision Transformer 注入 LoRA 层。
    
    参数:
        model: ViT 模型实例
        target_modules: 要替换的目标模块名称列表。若为 None，使用默认 ViT 模块。
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        dropout: LoRA dropout 率
    
    返回:
        包含层名称到 LoRALinear 模块映射的字典
    """
    if target_modules is None:
        target_modules = ['qv']
    
    lora_layers = {}
    targets = [t.lower() for t in target_modules]
    
    def replace_module(parent, name, module):
        """递归替换目标模块为 LoRA 层"""
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            child_name_lower = child_name.lower()
            
            # 检查是否为目标模块
            qv_only = ("qv" in targets) and ("qkv" in child_name_lower)
            standard_match = any(t in child_name_lower for t in targets if t != "qv")

            if isinstance(child_module, nn.Linear) and (qv_only or standard_match):
                output_mask = None
                if qv_only:
                    out_features = int(child_module.out_features)
                    if out_features % 3 == 0:
                        d = out_features // 3
                        mask = torch.ones(out_features, dtype=torch.float32)
                        mask[d:2 * d] = 0.0
                        output_mask = mask

                lora_layer = LoRALinear(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    output_mask=output_mask
                )
                lora_layer.base_weight.data = child_module.weight.data.clone()
                if child_module.bias is not None:
                    lora_layer.register_buffer('bias', child_module.bias.data.clone())
                
                setattr(module, child_name, lora_layer)
                lora_layers[full_name] = lora_layer
            
            # 递归处理子模块
            replace_module(module, full_name, child_module)
    
    replace_module(None, "", model)
    
    return lora_layers


def create_vit_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 10,
    pretrained: bool = False,
    pretrained_source: str = "modelscope",
    pretrained_checkpoint_path: Optional[str] = None,
    modelscope_model_id: Optional[str] = None,
    modelscope_revision: Optional[str] = None,
    modelscope_cache_dir: Optional[str] = None,
    modelscope_checkpoint_file: Optional[str] = None,
    use_peft: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0
) -> tuple:
    """
    创建带有 LoRA 注入的 ViT 模型。
    
    参数:
        model_name: 模型名称 (timm) 或架构名
        num_classes: 输出类别数
        pretrained: 是否使用预训练权重
        lora_rank: LoRA 秩
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    
    返回:
        (模型实例, LoRA 层字典) 的元组
    """
    use_transformers_dir = (
        pretrained
        and pretrained_checkpoint_path is not None
        and _looks_like_transformers_vit_dir(pretrained_checkpoint_path)
    )

    if use_transformers_dir:
        try:
            from transformers import ViTConfig, ViTForImageClassification
        except Exception as e:
            raise ImportError(f"加载本地 vit 目录需要 transformers: {e}") from e

        cfg = ViTConfig.from_pretrained(pretrained_checkpoint_path)
        cfg.num_labels = int(num_classes)
        cfg.id2label = {i: f"LABEL_{i}" for i in range(int(num_classes))}
        cfg.label2id = {label: i for i, label in cfg.id2label.items()}

        # Use Transformers' version-aware key mapping. Manual strict=False
        # loading silently skipped the encoder after the ViT key migration.
        backbone, loading_info = ViTForImageClassification.from_pretrained(
            pretrained_checkpoint_path,
            config=cfg,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )

        def loading_key(item) -> str:
            if isinstance(item, (tuple, list)) and len(item) > 0:
                return str(item[0])
            return str(item)

        missing_keys = list(loading_info.get("missing_keys", []) or [])
        unexpected_keys = list(loading_info.get("unexpected_keys", []) or [])
        mismatched_keys = list(loading_info.get("mismatched_keys", []) or [])
        error_msgs = list(loading_info.get("error_msgs", []) or [])
        non_classifier_missing = [
            key for key in missing_keys if not loading_key(key).startswith("classifier.")
        ]
        non_classifier_unexpected = [
            key for key in unexpected_keys if not loading_key(key).startswith("classifier.")
        ]
        non_classifier_mismatched = [
            key for key in mismatched_keys if not loading_key(key).startswith("classifier.")
        ]
        if (
            non_classifier_missing
            or non_classifier_unexpected
            or non_classifier_mismatched
            or error_msgs
        ):
            raise RuntimeError(
                "ViT pretrained backbone 未完整加载: "
                f"missing={non_classifier_missing[:10]}, "
                f"unexpected={non_classifier_unexpected[:10]}, "
                f"mismatched={non_classifier_mismatched[:10]}, "
                f"errors={error_msgs[:5]}"
            )

        print(
            "  ViT pretrained backbone 加载成功; "
            f"classifier_reinitialized={len(mismatched_keys)}"
        )
        model = _TransformersImageClassifierWrapper(backbone)
    else:
        if TIMM_AVAILABLE:
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=num_classes
                )
            except:
                # 回退到 torchvision
                from torchvision.models import vit_b_16
                model = vit_b_16(weights=None, num_classes=num_classes)
        else:
            from torchvision.models import vit_b_16
            model = vit_b_16(weights=None, num_classes=num_classes)

    if pretrained and not use_transformers_dir:
        ckpt_path = None
        if pretrained_checkpoint_path:
            if os.path.isfile(pretrained_checkpoint_path):
                ckpt_path = pretrained_checkpoint_path
            elif os.path.isdir(pretrained_checkpoint_path):
                ckpt_path = _select_checkpoint_file(pretrained_checkpoint_path, None)
        elif pretrained_source == "local":
            ckpt_path = pretrained_checkpoint_path
        elif pretrained_source == "modelscope":
            if not modelscope_model_id:
                raise ValueError("pretrained_source=modelscope 时必须提供 --ms_model_id")
            try:
                from modelscope.hub.snapshot_download import snapshot_download
            except Exception as e:
                raise ImportError(f"使用 ModelScope 需要安装 modelscope: {e}") from e
            try:
                snapshot_dir = snapshot_download(
                    modelscope_model_id,
                    cache_dir=modelscope_cache_dir,
                    revision=modelscope_revision
                )
            except TypeError:
                snapshot_dir = snapshot_download(modelscope_model_id)
            ckpt_path = _select_checkpoint_file(snapshot_dir, modelscope_checkpoint_file)
        else:
            raise ValueError(f"不支持的 pretrained_source: {pretrained_source}")

        if not ckpt_path:
            raise ValueError("启用 --pretrained 时必须提供权重来源与路径")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"未找到 pretrained checkpoint: {ckpt_path}")

        state_dict = _load_state_dict_from_checkpoint(ckpt_path)
        state_dict = _normalize_state_dict_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
    
    # 冻结基础模型的所有参数
    for param in model.parameters():
        param.requires_grad = False

    _unfreeze_classifier_params(model)
    
    discovered_targets = _discover_vit_lora_targets(model, use_transformers_dir=use_transformers_dir)

    if use_peft and PEFT_AVAILABLE and not use_transformers_dir and discovered_targets == ["qv"]:
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["qkv"],
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        for n, module in model.named_modules():
            if n.endswith("qkv") and hasattr(module, "base_layer") and hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                _patch_peft_qkv_qv_only(module)
        lora_layers = _extract_peft_lora_layers(model, adapter_name="default")
        _unfreeze_classifier_params(model)
    else:
        lora_layers = inject_lora_to_vit(
            model,
            target_modules=discovered_targets,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
    
    if len(lora_layers) == 0:
        linear_examples = [
            name for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ][:20]
        raise ValueError(
            "未注入任何 LoRA 层。"
            f"discovered_targets={discovered_targets}; "
            f"linear_examples={linear_examples}. "
            "请确认当前 ViT 是否包含 qkv/query/value/q_proj/v_proj/out_proj 等 attention projection。"
        )
    
    return model, lora_layers


def is_allocatable_lora_layer_name(name: str) -> bool:
    """Return whether a LoRA module should participate in FedHARP allocation.

    FedHARP's main protocol allocates Transformer attention LoRA modules.  Some
    PEFT target settings may also inject adapters into MLP/classifier modules
    such as fc1/fc2; those should not silently enter the DSA candidate pool.
    """
    n = str(name).lower()
    parts = n.replace("/", ".").split(".")

    blocked_parts = {
        "fc",
        "fc1",
        "fc2",
        "mlp",
        "intermediate",
        "output.dense",
        "classifier",
        "score",
        "head",
        "lm_head",
    }
    if any(p in blocked_parts for p in parts) or ".mlp." in n:
        return False

    allowed_suffixes = (
        "qkv",
        "query",
        "key",
        "value",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
    )
    if any(n.endswith(suf) for suf in allowed_suffixes):
        return True

    # GPT-style fused attention projection names are only safe when they sit
    # under an attention module; MLP c_proj should remain excluded.
    if n.endswith("c_attn"):
        return True
    if n.endswith("c_proj") and (".attn." in n or ".attention." in n):
        return True

    return False


def get_lora_layer_names(lora_layers: Dict[str, nn.Module]) -> List[str]:
    """获取排序后的 LoRA 层名称列表，仅包含 FedHARP 可分配的 attention 层。"""
    return sorted(name for name in lora_layers.keys() if is_allocatable_lora_layer_name(name))


def create_client_model(
    template_model: nn.Module,
    template_lora_layers: Dict[str, nn.Module]
) -> tuple:
    """
    为客户端创建模型副本，使其拥有独立的 LoRA 层实例。
    这对模拟 FedHARP 至关重要，因为每个客户端的 B 矩阵都是独立的。
    
    参数:
        template_model: 模板模型
        template_lora_layers: 模板 LoRA 层
    
    返回:
        (客户端模型副本, 客户端 LoRA 层字典)
    """
    import copy
    
    # 深拷贝模型结构
    client_model = copy.deepcopy(template_model)
    
    if PEFT_AVAILABLE and hasattr(template_model, "peft_config"):
        client_lora_layers = _extract_peft_lora_layers(client_model, adapter_name="default")
        for name, layer in client_lora_layers.items():
            template_layer = template_lora_layers.get(name)
            if template_layer is None:
                continue
            layer.set_A(template_layer.get_A())
            layer.set_B(template_layer.get_B())
        return client_model, client_lora_layers
    
    client_lora_layers = {}

    # 替换拷贝模型中的 LoRA 层为新实例
    def replace_lora_layers(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, LoRALinear):
                # 这是一个 LoRA 层，我们需要替换它
                template_layer = template_lora_layers.get(full_name)
                if template_layer is not None:
                    dropout_p = 0.0
                    if hasattr(template_layer, "dropout") and isinstance(template_layer.dropout, nn.Dropout):
                        dropout_p = float(template_layer.dropout.p)
                    output_mask = getattr(template_layer, "output_mask", None)
                    # 创建新的 LoRA 层实例
                    new_layer = LoRALinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        rank=child.rank,
                        alpha=child.alpha,
                        dropout=dropout_p,
                        output_mask=output_mask
                    )
                    # 从模板复制权重
                    # 基础权重 (W0) 共享 (其实不需要复制数据，指向同一个 tensor 即可省内存，这里为了简单直接复制)
                    new_layer.base_weight.data = template_layer.base_weight.data.clone()
                    if hasattr(template_layer, "bias"):
                        new_layer.register_buffer("bias", template_layer.bias.data.clone())
                    # A 矩阵初始复制全局的
                    new_layer.lora_A.data = template_layer.lora_A.data.clone()
                    # B 矩阵初始复制全局的 (全0)
                    new_layer.lora_B.data = template_layer.lora_B.data.clone()
                    
                    # 替换模型中的层
                    setattr(module, name, new_layer)
                    client_lora_layers[full_name] = new_layer
            
            # 递归处理子模块
            replace_lora_layers(child, full_name)
    
    replace_lora_layers(client_model)
    
    return client_model, client_lora_layers
