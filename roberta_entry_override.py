"""
RoBERTa + GLUE 的 Fed-HARP 联邦训练入口。
"""

import argparse
import csv
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
import numpy as np
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependency 'torch'. Run in fa_gpu or install requirements.") from e

from client import FedHarpClient
from server import FedHarpServer
from early_stopping import RoundEarlyStopping
from factor_projection import project_factor_dicts
from projection_validation import run_fork_probe
from utils import create_directories, get_device, set_seed


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="FedHARP on GLUE with RoBERTa + LoRA")

    p.add_argument(
        "--method",
        type=str,
        default="fedharp",
        choices=[
            "fedharp",
        ],
    )
    p.add_argument("--model_path", type=str, default="./Roberta")
    p.add_argument("--hf_endpoint", type=str, default=None)
    p.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "qnli"])
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--sst2_path", type=str, default=None)
    p.add_argument("--datasets_cache_dir", type=str, default=None)

    p.add_argument("--data_partition", type=str, default="dirichlet", choices=["dirichlet", "uniform"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--client_splits", type=str, default="0.9,0.1")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--data_fraction", type=float, default=1.0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)

    p.add_argument("--num_rounds", type=int, default=1)
    p.add_argument("--clients_per_round", type=int, default=None)
    p.add_argument("--heterogeneity_type", type=str, default="6-3-1", choices=["uniform", "6-3-1", "1-1-1"])
    p.add_argument("--allocation_ratio", type=float, default=0.5)
    p.add_argument("--aggregation_lr", type=float, default=1.0)
    p.add_argument("--struct_to_data_rounds", type=int, default=20)

    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str, default="query,value")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--b_train_every", type=int, default=25)
    p.add_argument("--b_num_epochs", type=int, default=1)
    p.add_argument("--b_num_steps", type=int, default=3,
                   help="LFC B-only 阶段每次最多执行的 minibatch 数 (0 表示不限制)")
    p.add_argument(
        "--projection_mode", type=str, default="none",
        choices=["none", "projection", "projection_1step"],
        help="stale-B transport mode",
    )
    p.add_argument(
        "--projection_ridge", type=float, default=1e-6,
        help="Tikhonov ridge for the rank-sized projection solve",
    )
    p.add_argument("--client_lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--eval_on", type=str, default="val", choices=["val", "client_holdout"])
    p.add_argument("--eval_scope", type=str, default="personalized", choices=["global_test", "personalized"])
    p.add_argument(
        "--early_stop_patience", type=int, default=0,
        help="连续多少次评估无提升后停止；0 表示禁用（保持固定轮数协议）",
    )
    p.add_argument(
        "--early_stop_min_delta", type=float, default=0.05,
        help="早停判定的最小变化量；accuracy 使用百分点，loss 使用 loss 单位",
    )
    p.add_argument(
        "--early_stop_min_rounds", type=int, default=8,
        help="早停预热轮数；预热期内不累计 patience",
    )
    p.add_argument(
        "--early_stop_metric", type=str, default="accuracy",
        choices=["accuracy", "loss"],
        help="早停监控指标：personalized accuracy 最大化或 loss 最小化",
    )
    p.add_argument("--fedharp_matrix_mode", type=str, default="a", choices=["a", "b", "ab", "adaptive", "auto"])
    p.add_argument("--adaptive_matrix_threshold", type=float, default=1.25)

    p.add_argument("--allocation_temperature", type=float, default=1.0)
    p.add_argument("--coverage_prior_weight", type=float, default=0.15)
    p.add_argument("--sample_weight_power", type=float, default=0.5)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_checkpoints", action="store_true")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    # E3 shock probe 接口（默认关闭；不传参时行为与基线实现完全一致）
    p.add_argument("--shock_probe", action="store_true",
                   help="启用重激活/shock 事件探针：固定 probe batch 上测 L_pre/L_sync/L_after")
    p.add_argument("--shock_probe_dir", type=str, default="./shock_probe_out",
                   help="探针输出目录（shock_probe_events.jsonl / allocation_trace.jsonl / summary）")
    p.add_argument("--shock_probe_samples", type=int, default=32,
                   help="每客户端固定 probe 样本数（取训练集头部，跨轮跨 run 一致）")
    p.add_argument("--allocation_trace_replay_path", type=str, default=None,
                   help="Replay selected clients and per-client block allocations from a JSONL trace")
    p.add_argument("--allocation_trace_dump_path", type=str, default=None,
                   help="Append the actual selected clients and DSA allocation after every round is formed")
    p.add_argument("--allocation_trace_replay_online", action="store_true",
                   help="Wait for each requested round to appear in a concurrently written trace")
    p.add_argument("--allocation_trace_wait_seconds", type=int, default=10800)
    p.add_argument("--expected_allocation_modules", type=int, default=0,
                   help="If positive, assert this exact per-client LoRA module budget each round")
    p.add_argument("--projection_fork_probe", action="store_true")
    p.add_argument("--projection_fork_limit", type=int, default=30)
    p.add_argument("--projection_fork_steps", type=int, default=5)
    p.add_argument("--projection_fork_eval_samples", type=int, default=128)
    p.add_argument("--projection_fork_dir", type=str, default="./projection_fork_out")
    p.add_argument("--projection_matched_bookkeeping", action="store_true",
                   help="Off control mirrors zero-sensitivity feedback at the same stale allocated targets")
    p.add_argument("--client_eval_jsonl", type=str, default=None)
    p.add_argument("--crossover_dump", type=str, default=None,
                   help="[S3c 2026-10-11] 若设置：把每个事件的全层 A/B（同步前 + 同步后）"
                        "dump 到该 .npz 路径，供另一臂做交叉反事实")
    p.add_argument("--crossover_load", type=str, default=None,
                   help="[S3c 2026-10-11] 若设置：加载另一臂的 A/B 快照 npz，"
                        "对每个事件做 2x2 交叉测量 {本臂A更新, 另一臂A更新} x {本臂B, 另一臂B}")

    return p.parse_args(args=argv)


def _read_sst2_tsv(path: str) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s = row.get("sentence")
            y = row.get("label")
            if s is None or y is None:
                continue
            try:
                labels.append(int(y))
            except Exception:
                continue
            texts.append(str(s))
    return texts, labels


def _read_sst2_parquet(path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_parquet(path)
    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Parquet file {path} must contain 'sentence' and 'label' columns. Found: {df.columns}")
    
    texts = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


TextInput = Union[str, Tuple[str, str]]

def _read_qnli_tsv(path: str) -> Tuple[List[Tuple[str, str]], List[int]]:
    texts: List[Tuple[str, str]] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            q = row.get("question")
            s = row.get("sentence")
            y = row.get("label")
            if q is None or s is None or y is None:
                continue
            y_str = str(y).strip()
            if y_str.lower() == "entailment":
                yy = 0
            elif y_str.lower() == "not_entailment":
                yy = 1
            else:
                try:
                    yy = int(y_str)
                except Exception:
                    continue
            texts.append((str(q), str(s)))
            labels.append(int(yy))
    return texts, labels


def _read_qnli_parquet(path: str) -> Tuple[List[Tuple[str, str]], List[int]]:
    df = pd.read_parquet(path)
    if "question" not in df.columns or "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Parquet file {path} must contain 'question', 'sentence' and 'label' columns. Found: {df.columns}"
        )
    texts = list(zip(df["question"].astype(str).tolist(), df["sentence"].astype(str).tolist()))
    labels = df["label"].astype(int).tolist()
    return texts, labels


def _read_qnli_jsonl(path: str) -> Tuple[List[Tuple[str, str]], List[int]]:
    texts: List[Tuple[str, str]] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue

            if "text1" in obj and "text2" in obj:
                q = obj.get("text1")
                sent = obj.get("text2")
            else:
                q = obj.get("question")
                sent = obj.get("sentence")

            y = obj.get("label")
            if q is None or sent is None or y is None:
                continue

            yy: Optional[int]
            if isinstance(y, str):
                y_str = y.strip().lower().replace(" ", "_")
                if y_str == "entailment":
                    yy = 0
                elif y_str in {"not_entailment", "not_entail"}:
                    yy = 1
                else:
                    try:
                        yy = int(y_str)
                    except Exception:
                        yy = None
            else:
                try:
                    yy = int(y)
                except Exception:
                    yy = None

            if yy is None or int(yy) < 0:
                continue

            texts.append((str(q), str(sent)))
            labels.append(int(yy))
    return texts, labels


def _candidate_dataset_dir(*paths: Optional[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None


def _load_dataset(args) -> Tuple[List[TextInput], List[int], List[TextInput], List[int]]:
    dataset = str(args.dataset).lower()
    candidate_dir = _candidate_dataset_dir(
        args.data_path,
        args.sst2_path if dataset == "sst2" else None,
        "./data/sst-2" if dataset == "sst2" else None,
        "./data/sst2" if dataset == "sst2" else None,
        "./data/SST-2" if dataset == "sst2" else None,
        "./data/qnli" if dataset == "qnli" else None,
        "./data/QNLI" if dataset == "qnli" else None,
        "./QNLI" if dataset == "qnli" else None,
        "./qnli" if dataset == "qnli" else None,
    )

    if candidate_dir is not None:
        train_parquet = os.path.join(candidate_dir, "train-00000-of-00001.parquet")
        val_parquet = os.path.join(candidate_dir, "validation-00000-of-00001.parquet")
        if os.path.isfile(train_parquet) and os.path.isfile(val_parquet):
            if dataset == "sst2":
                print(f"Loading SST-2 from Parquet: {candidate_dir}")
                train_texts, train_labels = _read_sst2_parquet(train_parquet)
                val_texts, val_labels = _read_sst2_parquet(val_parquet)
                return train_texts, train_labels, val_texts, val_labels
            if dataset == "qnli":
                print(f"Loading QNLI from Parquet: {candidate_dir}")
                train_texts, train_labels = _read_qnli_parquet(train_parquet)
                val_texts, val_labels = _read_qnli_parquet(val_parquet)
                return train_texts, train_labels, val_texts, val_labels

        train_tsv = os.path.join(candidate_dir, "train.tsv")
        dev_tsv = os.path.join(candidate_dir, "dev.tsv")
        if not os.path.isfile(dev_tsv):
            dev_tsv = os.path.join(candidate_dir, "validation.tsv")
        if os.path.isfile(train_tsv) and os.path.isfile(dev_tsv):
            if dataset == "sst2":
                print(f"Loading SST-2 from TSV: {candidate_dir}")
                train_texts, train_labels = _read_sst2_tsv(train_tsv)
                val_texts, val_labels = _read_sst2_tsv(dev_tsv)
                return train_texts, train_labels, val_texts, val_labels
            if dataset == "qnli":
                print(f"Loading QNLI from TSV: {candidate_dir}")
                train_texts, train_labels = _read_qnli_tsv(train_tsv)
                val_texts, val_labels = _read_qnli_tsv(dev_tsv)
                return train_texts, train_labels, val_texts, val_labels

        train_jsonl = os.path.join(candidate_dir, "train.jsonl")
        dev_jsonl = os.path.join(candidate_dir, "dev.jsonl")
        if not os.path.isfile(dev_jsonl):
            dev_jsonl = os.path.join(candidate_dir, "validation.jsonl")
        if dataset == "qnli" and os.path.isfile(train_jsonl) and os.path.isfile(dev_jsonl):
            print(f"Loading QNLI from JSONL: {candidate_dir}")
            train_texts, train_labels = _read_qnli_jsonl(train_jsonl)
            val_texts, val_labels = _read_qnli_jsonl(dev_jsonl)
            return train_texts, train_labels, val_texts, val_labels

        if args.data_path is not None or (args.sst2_path is not None and dataset == "sst2"):
            raise SystemExit(f"Invalid dataset path: {candidate_dir}")

    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit("Missing dependency 'datasets'. Install with: pip install datasets") from e

    try:
        raw = load_dataset("glue", dataset, cache_dir=args.datasets_cache_dir)
    except Exception as e:
        raise SystemExit(
            f"Cannot load {dataset} from the Hub. Provide --data_path, or set --hf_endpoint / proxy."
        ) from e

    train_split = raw["train"]
    val_split = raw["validation"]
    if dataset == "sst2":
        train_texts = list(train_split["sentence"])
        train_labels = [int(x) for x in train_split["label"]]
        val_texts = list(val_split["sentence"])
        val_labels = [int(x) for x in val_split["label"]]
        return train_texts, train_labels, val_texts, val_labels
    if dataset == "qnli":
        train_texts = list(zip(train_split["question"], train_split["sentence"]))
        train_labels = [int(x) for x in train_split["label"]]
        val_texts = list(zip(val_split["question"], val_split["sentence"]))
        val_labels = [int(x) for x in val_split["label"]]
        return train_texts, train_labels, val_texts, val_labels
    raise SystemExit(f"Unsupported dataset: {dataset}")


def _safe_token(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_")


def _build_log_path(args) -> str:
    safe_method = _safe_token(args.method)
    safe_alpha = _safe_token(args.alpha)
    safe_dataset = _safe_token(args.dataset)
    safe_model = _safe_token(os.path.basename(os.path.abspath(args.model_path)) or "roberta")
    safe_heterogeneity = _safe_token(args.heterogeneity_type)
    safe_partition = _safe_token(getattr(args, "data_partition", "dirichlet"))
    log_filename = f"Traininglog-{safe_method}-{safe_partition}-alpha{safe_alpha}-{safe_dataset}-{safe_model}-{safe_heterogeneity}.txt"
    return os.path.join(args.checkpoint_dir if args.save_checkpoints else ".", log_filename)


def generate_client_resources(num_clients: int, heterogeneity_type: str, base_ratio: float) -> List[float]:
    if heterogeneity_type == "uniform":
        return [base_ratio] * num_clients
    if heterogeneity_type == "6-3-1":
        n_low = int(0.6 * num_clients)
        n_mid = int(0.3 * num_clients)
        n_high = num_clients - n_low - n_mid
        ratios: List[float] = []
        ratios.extend([base_ratio] * n_low)
        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid)
        ratios.extend([1.0] * n_high)
        np.random.shuffle(ratios)
        return ratios
    if heterogeneity_type == "1-1-1":
        n_low = int(num_clients // 3)
        n_mid = int(num_clients // 3)
        n_high = int(num_clients) - int(n_low) - int(n_mid)
        ratios: List[float] = []
        ratios.extend([base_ratio] * int(n_low))
        ratios.extend([min(1.0, base_ratio * 1.5)] * int(n_mid))
        ratios.extend([1.0] * int(n_high))
        np.random.shuffle(ratios)
        return ratios
    return [base_ratio] * num_clients


def _unfreeze_classifier_params(model: nn.Module):
    for attr in ("classifier", "score"):
        if hasattr(model, attr):
            head = getattr(model, attr)
            if isinstance(head, nn.Module):
                for p in head.parameters():
                    p.requires_grad = True


def _freeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _extract_peft_lora_layers(model: nn.Module, adapter_name: str = "default") -> Dict[str, nn.Module]:
    class _Adapter:
        def __init__(self, peft_linear_module: nn.Module, adapter: str):
            self._m = peft_linear_module
            self._adapter = adapter
            self.init_rank = int(self.lora_A.shape[0])

        @property
        def lora_A(self) -> torch.nn.Parameter:
            return self._m.lora_A[self._adapter].weight

        @property
        def lora_B(self) -> torch.nn.Parameter:
            return self._m.lora_B[self._adapter].weight

        def get_A(self) -> torch.Tensor:
            return self.lora_A.data.clone()

        def set_A(self, A: torch.Tensor):
            A = A.to(device=self.lora_A.device, dtype=self.lora_A.dtype).clone()
            if tuple(A.shape) != tuple(self.lora_A.shape):
                old_requires_grad = bool(self.lora_A.requires_grad)
                in_features = int(A.shape[1])
                out_features = int(A.shape[0])
                new_A = nn.Linear(in_features, out_features, bias=False).to(device=A.device, dtype=A.dtype)
                new_A.weight.data.copy_(A)
                new_A.weight.requires_grad = old_requires_grad
                self._m.lora_A[self._adapter] = new_A
                self._update_peft_rank_meta(rank=out_features)
            else:
                self.lora_A.data = A

        def get_B(self) -> torch.Tensor:
            return self.lora_B.data.clone()

        def set_B(self, B: torch.Tensor):
            B = B.to(device=self.lora_B.device, dtype=self.lora_B.dtype).clone()
            if tuple(B.shape) != tuple(self.lora_B.shape):
                old_requires_grad = bool(self.lora_B.requires_grad)
                in_features = int(B.shape[1])
                out_features = int(B.shape[0])
                new_B = nn.Linear(in_features, out_features, bias=False).to(device=B.device, dtype=B.dtype)
                new_B.weight.data.copy_(B)
                new_B.weight.requires_grad = old_requires_grad
                self._m.lora_B[self._adapter] = new_B
                self._update_peft_rank_meta(rank=in_features)
            else:
                self.lora_B.data = B

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

        def _get_base_weight(self) -> torch.nn.Parameter:
            base_layer = getattr(self._m, "base_layer", None)
            if base_layer is not None and hasattr(base_layer, "weight"):
                return base_layer.weight
            if hasattr(self._m, "weight"):
                return getattr(self._m, "weight")
            raise ValueError(f"Unsupported PEFT LoRA module type: {type(self._m)}")

        def _get_scaling(self) -> float:
            scaling = getattr(self._m, "scaling", None)
            if isinstance(scaling, dict) and self._adapter in scaling:
                try:
                    return float(scaling[self._adapter])
                except Exception:
                    pass
            lora_alpha = getattr(self._m, "lora_alpha", None)
            if isinstance(lora_alpha, dict) and self._adapter in lora_alpha:
                try:
                    alpha = float(lora_alpha[self._adapter])
                    r = int(self.lora_A.shape[0])
                    return alpha / float(r) if r > 0 else 0.0
                except Exception:
                    pass
            return 1.0

        def _update_peft_rank_meta(self, rank: int):
            rank = int(rank)
            r_map = getattr(self._m, "r", None)
            if isinstance(r_map, dict):
                r_map[self._adapter] = rank
            scaling_map = getattr(self._m, "scaling", None)
            lora_alpha = getattr(self._m, "lora_alpha", None)
            if isinstance(scaling_map, dict) and isinstance(lora_alpha, dict) and self._adapter in lora_alpha:
                try:
                    scaling_map[self._adapter] = float(lora_alpha[self._adapter]) / float(rank) if rank > 0 else 0.0
                except Exception:
                    pass

        def reset_lora(self, rank: Optional[int] = None):
            target_rank = int(rank if rank is not None else self.init_rank)
            A_old = self.lora_A
            B_old = self.lora_B
            A_requires_grad = bool(A_old.requires_grad)
            B_requires_grad = bool(B_old.requires_grad)
            device = A_old.device
            dtype = A_old.dtype

            in_features = int(A_old.shape[1])
            out_features = int(B_old.shape[0])

            new_A = nn.Linear(in_features, target_rank, bias=False).to(device=device, dtype=dtype)
            new_B = nn.Linear(target_rank, out_features, bias=False).to(device=device, dtype=dtype)
            new_A.weight.data.normal_(mean=0.0, std=0.02)
            new_B.weight.data.zero_()
            new_A.weight.requires_grad = A_requires_grad
            new_B.weight.requires_grad = B_requires_grad

            self._m.lora_A[self._adapter] = new_A
            self._m.lora_B[self._adapter] = new_B
            self._update_peft_rank_meta(rank=target_rank)

        def merge_lora_weights(self):
            r = int(self.lora_A.shape[0])
            if r <= 0:
                return
            scaling = self._get_scaling()
            delta_w = (self.lora_B @ self.lora_A) * float(scaling)
            base_w = self._get_base_weight()
            with torch.no_grad():
                base_w.data = base_w.data + delta_w.to(device=base_w.device, dtype=base_w.dtype)

        def merge_lora_into_base_and_reset(self, rank: Optional[int] = None):
            self.merge_lora_weights()
            self.reset_lora(rank=rank)

    lora_layers: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        try:
            _ = module.lora_A[adapter_name].weight
            _ = module.lora_B[adapter_name].weight
        except Exception:
            continue
        lora_layers[name] = _Adapter(module, adapter=adapter_name)
    return lora_layers


def create_roberta_lora_model(
    *,
    model_path: str,
    num_labels: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    target_modules: Sequence[str],
):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import transformers.utils.import_utils
        import transformers.modeling_utils
        transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
        transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
    except Exception as e:
        raise SystemExit("Missing dependency 'transformers'. Install requirements.") from e
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise SystemExit("Missing dependency 'peft'. Install requirements.") from e

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=int(num_labels), 
            weights_only=False,
            low_cpu_mem_usage=False
        )
    except TypeError:
        # Fallback for older transformers that don't support weights_only
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=int(num_labels))

    lora_config = LoraConfig(
        r=int(lora_rank),
        lora_alpha=float(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    lora_layers = _extract_peft_lora_layers(model, adapter_name="default")

    _freeze_all_params(model)
    _unfreeze_classifier_params(model)
    for layer in lora_layers.values():
        layer.unfreeze_all()

    return model, tokenizer, lora_layers


def _uniform_partition(
    labels: np.ndarray,
    num_clients: int,
    seed: int,
):
    rng = np.random.RandomState(int(seed))
    client_indices: List[List[int]] = [[] for _ in range(int(num_clients))]
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        for cid, chunk in enumerate(np.array_split(cls_idx, int(num_clients))):
            client_indices[cid].extend(int(x) for x in chunk.tolist())
    for cid in range(int(num_clients)):
        rng.shuffle(client_indices[cid])
    return client_indices


def _dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
):
    rng = np.random.RandomState(int(seed))
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    labels = labels[indices]

    num_classes = int(labels.max()) + 1
    client_indices: List[List[int]] = [[] for _ in range(int(num_clients))]
    for cls in range(num_classes):
        cls_idx = indices[labels == cls]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([float(alpha)] * int(num_clients))
        counts = (proportions * len(cls_idx)).astype(int)
        counts[-1] = len(cls_idx) - int(counts[:-1].sum())
        start = 0
        for cid in range(int(num_clients)):
            end = start + int(counts[cid])
            client_indices[cid].extend(cls_idx[start:end].tolist())
            start = end
    for cid in range(int(num_clients)):
        rng.shuffle(client_indices[cid])
    return client_indices


@dataclass
class EncodedText:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def _encode_texts(tokenizer, texts: Sequence[TextInput], labels: Sequence[int], max_length: int) -> EncodedText:
    texts_list: List[TextInput] = list(texts)
    if len(texts_list) > 0 and isinstance(texts_list[0], tuple) and len(texts_list[0]) == 2:
        a: List[str] = []
        b: List[str] = []
        for t in texts_list:
            if not isinstance(t, tuple) or len(t) != 2:
                raise ValueError("Mixed text inputs: expected (text_a, text_b) pairs")
            a.append(str(t[0]))
            b.append(str(t[1]))
        enc = tokenizer(
            a,
            b,
            padding="max_length",
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
    else:
        enc = tokenizer(
            [str(t) for t in texts_list],
            padding="max_length",
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
    y = torch.tensor(list(labels), dtype=torch.long)
    return EncodedText(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=y)


class SST2ClientDataset(Dataset):
    def __init__(self, encoded: EncodedText, indices: Sequence[int]):
        self.provenance_indices = tuple(int(index) for index in indices)
        self.provenance_encoded_id = id(encoded)
        idx = torch.tensor(list(indices), dtype=torch.long)
        self.input_ids = encoded.input_ids.index_select(0, idx)
        self.attention_mask = encoded.attention_mask.index_select(0, idx)
        self.labels = encoded.labels.index_select(0, idx)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int):
        return {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i]}, self.labels[i]


class TextFedHarpClient(FedHarpClient):
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
        for _, (inputs, target) in enumerate(self.train_loader):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            target = target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(**inputs)
            logits = getattr(out, "logits", out)
            loss = criterion(logits, target)
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

            total_loss += float(loss.item())
            num_batches += 1
            steps_seen += 1
            if max_steps is not None and steps_seen >= max(1, int(max_steps)):
                break

        if num_batches == 0:
            return 0.0, {name: 0.0 for name in self.lora_layers.keys()}

        grad_sensitivity = {name: (grad_norm_sq_sum[name] / num_batches) ** 0.5 for name in grad_norm_sq_sum.keys()}
        return total_loss / num_batches, grad_sensitivity

    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
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
            for inputs, target in test_loader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                target = target.to(self.device)
                out = self.model(**inputs)
                logits = getattr(out, "logits", out)
                test_loss += float(criterion(logits, target).item())
                pred = logits.argmax(dim=1)
                correct += int(pred.eq(target).sum().item())
                total += int(target.size(0))

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}


class RobertaFedHarpServer(FedHarpServer):
    _layer_re = re.compile(r"(?:encoder\\.)?layer\\.(\\d+)")

    def _get_layer_depth(self, layer_name: str) -> int:
        m = self._layer_re.search(layer_name)
        if not m:
            return 0
        try:
            return int(m.group(1))
        except Exception:
            return 0


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _dict_tensor_nbytes(d: Dict[str, torch.Tensor]) -> int:
    total = 0
    for _, v in d.items():
        total += _tensor_nbytes(v)
    return total


def _A_matrices_nbytes(A_matrices: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(A_matrices)


def _delta_A_nbytes(delta_A: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(delta_A)

def _delta_B_nbytes(delta_B: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(delta_B)


def _ab_trainable_param_numel(client_lora_layers: Dict[str, nn.Module], allocated_layers: Set[str]) -> int:
    n = 0
    for name in allocated_layers:
        layer = client_lora_layers.get(name)
        if layer is None:
            continue
        n += int(layer.lora_A.numel())
        n += int(layer.lora_B.numel())
    return n


def _clone_model_and_layers(template_model: nn.Module, template_lora_layers: Dict[str, nn.Module]):
    import copy

    m = copy.deepcopy(template_model)
    layers = _extract_peft_lora_layers(m, adapter_name="default")
    for name, layer in layers.items():
        src = template_lora_layers.get(name)
        if src is None:
            continue
        layer.set_A(src.get_A())
        layer.set_B(src.get_B())
        layer.unfreeze_all()
    _freeze_all_params(m)
    _unfreeze_classifier_params(m)
    return m, layers


class _ShockProbeState:
    """重激活事件探针的跨轮状态。"""

    def __init__(self):
        self.last_active_round = {}   # cid -> {layer_name: round}
        self.events = []             # 已完成事件（每 (round, client) 一行）
        self.round_allocations = []   # per-round 分配记录（审计用）


def _build_fixed_probe_batch(client, num_samples):
    """从客户端训练集头部取固定 probe 样本。

    直接索引 dataset（绕过 DataLoader shuffle），保证 probe batch 跨轮、
    跨 run 完全一致；三时点（pre/sync/after）配对比较才有意义。
    """
    ds = client.train_loader.dataset
    n = min(int(num_samples), len(ds))
    inputs_list, target_list = [], []
    for i in range(n):
        item_input, item_target = ds[i]
        inputs_list.append(item_input)
        target_list.append(item_target)
    inputs = {k: torch.stack([x[k] for x in inputs_list]) for k in inputs_list[0].keys()}
    target = torch.stack(target_list)
    inputs = {k: v.to(client.device) for k, v in inputs.items()}
    target = target.to(client.device)
    return inputs, target


def _probe_forward_loss(client, probe_inputs, probe_target):
    """固定 probe batch 上的前向 CE loss（无梯度、不动训练状态）。"""
    client.model.eval()
    with torch.no_grad():
        out = client.model(**probe_inputs)
        logits = getattr(out, "logits", out)
        return float(nn.CrossEntropyLoss()(logits, probe_target).item())


# =====================================================================
# [S3c 2026-10-11] 2x2 交叉反事实（cross-over counterfactual）
# ---------------------------------------------------------------------
# 目的：把「A 更新本身」与「B 放大器」分开。对每个事件，在同一条训练轨迹上
# 构造 4 个反事实单元：{本臂 A 更新, 另一臂 A 更新} x {本臂 B, 另一臂 B}，
# 每单元 = L(A_after, B) - L(A_before, B)。
# 测量均为 eval + no_grad 前向，测完按 A_after/B_after 精确还原，
# 故不改变训练轨迹（可与 S3/S3b 逐位比对验证）。
# =====================================================================
CROSSOVER_DUMP = None
CROSSOVER_LOAD = None
_CROSSOVER_BUF = {}


def _crossover_key(rd, cid, layer, field):
    return "r%d_c%d__%s__%s" % (int(rd), int(cid), layer, field)


def _crossover_record(evt):
    """把事件的全层 A/B（同步前 + 同步后）收入 dump 缓冲（CPU float32）。"""
    rd, cid = evt["round"], evt["client"]
    for field, d in (("Abefore", evt["_A_before_all"]),
                     ("Bbefore", evt["_B_before_all"]),
                     ("Aafter", evt["_A_after_all"]),
                     ("Bafter", evt["_B_after_all"])):
        for name, t in d.items():
            _CROSSOVER_BUF[_crossover_key(rd, cid, name, field)] = (
                t.detach().float().cpu().numpy())


def _crossover_load_file(path):
    """加载另一臂快照 -> {(round, client, layer): {field: torch.Tensor}}。"""
    import re as _re
    out = {}
    with np.load(path) as z:
        for k in z.files:
            m = _re.match(r"^r(\d+)_c(\d+)__(.+)__(Abefore|Bbefore|Aafter|Bafter)$", k)
            if m is None:
                continue
            rd, cid, layer, field = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            out.setdefault((rd, cid, layer), {})[field] = torch.from_numpy(z[k])
    n_ev = len({(rd, cid) for (rd, cid, _l) in out})
    print("[crossover] loaded %s: %d layer-snapshots, %d events"
          % (path, len(out), n_ev))
    return out


def _shock_probe_crossover(evt, client):
    """2x2 交叉测量：四单元共享同一条轨迹背景与同一 probe batch。"""
    if CROSSOVER_LOAD is None:
        return
    key = (evt["round"], evt["client"])
    other = {n: d for (rd, cid, n), d in CROSSOVER_LOAD.items() if (rd, cid) == key}
    if not other:
        return
    oth_b = {n: d.get("Abefore") for n, d in other.items()}
    oth_a = {n: d.get("Aafter") for n, d in other.items()}
    oth_B = {n: d.get("Bbefore") for n, d in other.items()}
    if any(v is None for v in list(oth_b.values()) + list(oth_a.values()) + list(oth_B.values())):
        return
    probe_inputs, probe_target = evt["_probe_batch"]
    Ab, Aa = evt["_A_before_all"], evt["_A_after_all"]
    Bb, Ba = evt["_B_before_all"], evt["_B_after_all"]
    for tag, A_b, A_a, B in (("own_own", Ab, Aa, Bb),
                             ("own_oth", Ab, Aa, oth_B),
                             ("oth_own", oth_b, oth_a, Bb),
                             ("oth_oth", oth_b, oth_a, oth_B)):
        client.set_A_matrices(A_b)
        client.set_B_matrices(B)
        l_pre = _probe_forward_loss(client, probe_inputs, probe_target)
        client.set_A_matrices(A_a)
        l_post = _probe_forward_loss(client, probe_inputs, probe_target)
        evt["xL_pre_" + tag] = l_pre
        evt["xL_post_" + tag] = l_post
        evt["xdL_" + tag] = l_post - l_pre
    evt["crossover_ok"] = True
    client.set_A_matrices(Aa)
    client.set_B_matrices(Ba)


def _shock_probe_begin(client, cid, round_num, allocated_layers, stale_map,
                       b_train_every, projection_mode, state, probe_batch):
    """A 同步前调用：判定事件块并测 L_pre、快照 A/B。

    事件块 = 重激活块（本轮被分配且距上次活跃 > 1 轮） ∪ stale 触发块
    （tau 计数 >= b_train_every）。无事件返回 None（零开销）。
    """
    last_active = state.last_active_round.setdefault(int(cid), {})
    event_blocks = {}
    for name in sorted(allocated_layers):
        gap = None
        if name in last_active:
            gap = int(round_num) - int(last_active[name])
        tau_counter = int(stale_map.get(name, 0))
        is_reactivation = gap is not None and gap > 1
        is_stale_trigger = b_train_every > 0 and tau_counter >= int(b_train_every)
        if is_reactivation or is_stale_trigger:
            event_blocks[name] = {"gap": gap, "tau_counter": tau_counter}
    if b_train_every > 0:
        for name, tau in stale_map.items():
            if int(tau) >= int(b_train_every) and name not in event_blocks:
                event_blocks[name] = {"gap": None, "tau_counter": int(tau)}
    if not event_blocks:
        return None
    probe_inputs, probe_target = probe_batch
    evt = {
        "round": int(round_num),
        "client": int(cid),
        "projection_mode": str(projection_mode),
        "blocks": event_blocks,
        "probe_samples": int(probe_target.shape[0]),
        "L_pre": _probe_forward_loss(client, probe_inputs, probe_target),
        "_probe_batch": (probe_inputs, probe_target),
        "_A_before": {},
        "_B_before": {},
    }
    for name in event_blocks:
        layer = client.lora_layers[name]
        evt["_A_before"][name] = layer.lora_A.detach().clone()
        evt["_B_before"][name] = layer.lora_B.detach().clone()
    # [S3c 2026-10-11] 全层快照：交叉反事实要求把模型「整体」恢复到同步前状态，
    # 只恢复事件块会让非事件块停留在同步后，导致交叉基线与 L_pre 不可比。
    evt["_A_before_all"] = {n: ly.lora_A.detach().clone()
                            for n, ly in client.lora_layers.items()}
    evt["_B_before_all"] = {n: ly.lora_B.detach().clone()
                            for n, ly in client.lora_layers.items()}
    return evt


def _shock_probe_post_a_sync(evt, client):
    """[S3b 2026-10-11] 仅 A 同步后、B（若有）应用前调用：测 L_A_sync。

    把「A 同步」与「B 聚合应用」两个源分开测量，使 ab 臂的
    「仅 A-sync 冲击」(delta_L_A_only = L_A_sync - L_pre) 与 native 臂的
    delta_L 同口径可比（S3 遗留方法学待办）。native 臂（无 B 下发）中
    L_A_sync 与 L_sync 为同一时点的重复测量，两值应恒等，可作一致性自检。
    """
    probe_inputs, probe_target = evt["_probe_batch"]
    evt["L_A_sync"] = _probe_forward_loss(client, probe_inputs, probe_target)


def _shock_probe_post_sync(evt, client):
    """A 同步后、B-only 修正前调用：测 L_sync 与块级 E_fro / s_rel。

    E = B^s (A^t - A^s)：本地保留的 B 作用在「本轮同步 A − 客户端此前持有的 A」
    上的有效漂移；s_rel = ||E||_F / (||B A^s||_F + eps)。
    """
    probe_inputs, probe_target = evt["_probe_batch"]
    evt["L_sync"] = _probe_forward_loss(client, probe_inputs, probe_target)
    for name, blk in evt["blocks"].items():
        layer = client.lora_layers[name]
        A_new = layer.lora_A.detach()
        A_old = evt["_A_before"][name]
        B = evt["_B_before"][name]
        with torch.no_grad():
            dA = A_new - A_old
            E = B @ dA
            blk["A_drift_fro"] = float(torch.norm(dA, p="fro").item())
            blk["E_fro"] = float(torch.norm(E, p="fro").item())
            blk["B_fro"] = float(torch.norm(B, p="fro").item())
            denom = float(torch.norm(B @ A_old, p="fro").item()) + 1e-12
            blk["s_rel"] = blk["E_fro"] / denom
    # [S3c 2026-10-11] 同步后全层快照：交叉测量的「同步后」端点
    evt["_A_after_all"] = {n: ly.lora_A.detach().clone()
                           for n, ly in client.lora_layers.items()}
    evt["_B_after_all"] = {n: ly.lora_B.detach().clone()
                           for n, ly in client.lora_layers.items()}
    if CROSSOVER_DUMP is not None:
        _crossover_record(evt)


def _shock_probe_post_lpc(evt, client, b_only_layers, new_b_sensitivity,
                          corrector_kind="legacy_task_loss"):
    """B-only 修正后调用：测 L_after、drift 与块级 B 梯度范数。"""
    probe_inputs, probe_target = evt["_probe_batch"]
    evt["L_after"] = _probe_forward_loss(client, probe_inputs, probe_target)
    evt["corrector_kind"] = str(corrector_kind)
    evt["projection_applied"] = str(corrector_kind) == "projection" and bool(b_only_layers)
    evt["corrected_blocks"] = sorted(b_only_layers)
    for name in sorted(b_only_layers):
        layer = client.lora_layers[name]
        B_before = evt["_B_before"].get(name)
        A_new = layer.lora_A.detach()
        B_after = layer.lora_B.detach()
        blk = evt["blocks"].get(name)
        if blk is None:
            blk = evt["blocks"][name] = {"gap": None, "tau_counter": None}
        if B_before is not None:
            with torch.no_grad():
                dW = (B_after @ A_new) - (B_before @ A_new)
            blk["drift_fro"] = float(torch.norm(dW, p="fro").item())
        else:
            blk["drift_fro"] = None
        blk["b_grad_norm"] = float(new_b_sensitivity.get(name, 0.0))


def _shock_probe_finalize(evt, state):
    """事件收尾：补齐衍生量、剥离内部张量字段后落盘。"""
    if "L_sync" in evt:
        evt["delta_L"] = evt["L_sync"] - evt["L_pre"]
    else:
        evt["delta_L"] = None
    # [S3b 2026-10-11] 仅 A-sync 冲击：与 native 臂 delta_L 同口径的可比量
    if evt.get("L_A_sync") is not None:
        evt["delta_L_A_only"] = evt["L_A_sync"] - evt["L_pre"]
    else:
        evt["delta_L_A_only"] = None
    if evt.get("L_after") is not None:
        evt["lpc_benefit"] = evt["L_sync"] - evt["L_after"]
    else:
        evt["lpc_benefit"] = None
    # Projection evidence is based on paired absolute loss jumps around the
    # same sync event: |L_sync-L_pre| versus |L_after_projection-L_pre|.
    # This avoids calling a signed decrease mitigation when a negative jump
    # simply becomes more negative. Legacy task-loss correction is kept out.
    if evt.get("corrector_kind") == "projection" and evt.get("projection_applied"):
        evt["projection_jump_before"] = evt["delta_L"]
        evt["projection_jump_after"] = evt["L_after"] - evt["L_pre"]
        evt["projection_abs_jump_reduction"] = (
            abs(evt["projection_jump_before"]) - abs(evt["projection_jump_after"])
        )
        evt["projection_abs_jump_reduced"] = (
            abs(evt["projection_jump_after"]) < abs(evt["projection_jump_before"])
        )
    else:
        evt["projection_jump_before"] = None
        evt["projection_jump_after"] = None
        evt["projection_abs_jump_reduction"] = None
        evt["projection_abs_jump_reduced"] = None
    evt["num_event_blocks"] = len(evt["blocks"])
    evt["blocks"] = {k: v for k, v in sorted(evt["blocks"].items())}
    evt.pop("_probe_batch", None)
    evt.pop("_A_before", None)
    evt.pop("_B_before", None)
    evt.pop("_A_before_all", None)
    evt.pop("_B_before_all", None)
    evt.pop("_A_after_all", None)
    evt.pop("_B_after_all", None)
    state.events.append(evt)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    set_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        dev = str(args.device)
        if (":" not in dev) or dev.endswith(":0") or dev == "cuda":
            print(f"检测到 {torch.cuda.device_count()} 张 GPU，可用 --device cuda:1 使用第二张 GPU")

    if args.save_checkpoints:
        create_directories(args.checkpoint_dir)

    log_path = _build_log_path(args)
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Training configuration (arguments):\n")
            for k, v in sorted(vars(args).items()):
                f.write(f"{k} = {v}\n")
            f.write("\nRound logs:\n")
    except Exception as e:
        print(f"[警告] 无法写入日志文件 {log_path}: {e}")

    train_texts_all, train_labels_all, val_texts_all, val_labels_all = _load_dataset(args)

    rng = np.random.RandomState(int(args.seed))
    train_perm = rng.permutation(len(train_texts_all))
    if args.max_train_samples is not None:
        train_perm = train_perm[: int(args.max_train_samples)]
    train_texts = [train_texts_all[int(i)] for i in train_perm.tolist()]
    train_labels = np.asarray([int(train_labels_all[int(i)]) for i in train_perm.tolist()], dtype=np.int64)

    val_perm = rng.permutation(len(val_texts_all))
    if args.max_eval_samples is not None:
        val_perm = val_perm[: int(args.max_eval_samples)]
    val_texts = [val_texts_all[int(i)] for i in val_perm.tolist()]
    val_labels = np.asarray([int(val_labels_all[int(i)]) for i in val_perm.tolist()], dtype=np.int64)

    lora_targets = [s.strip() for s in str(args.lora_targets).split(",") if s.strip()]
    global_model, tokenizer, global_lora_layers = create_roberta_lora_model(
        model_path=args.model_path,
        num_labels=2,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
    )
    global_model.to(device)
    for layer in global_lora_layers.values():
        layer.unfreeze_all()

    encoded_train = _encode_texts(tokenizer, train_texts, train_labels.tolist(), max_length=args.max_length)
    encoded_val = _encode_texts(tokenizer, val_texts, val_labels.tolist(), max_length=args.max_length)

    data_partition = str(getattr(args, "data_partition", "dirichlet")).lower().strip()
    if data_partition == "uniform":
        client_indices = _uniform_partition(
            train_labels,
            num_clients=args.num_clients,
            seed=args.seed,
        )
        print("Data partition: uniform (stratified IID; alpha ignored)")
    else:
        client_indices = _dirichlet_partition(
            train_labels,
            num_clients=args.num_clients,
            alpha=args.alpha,
            seed=args.seed,
        )
        print(f"Data partition: dirichlet alpha={args.alpha}")
    if int(len(train_labels)) < int(args.num_clients):
        raise SystemExit(f"num_clients={args.num_clients} 大于训练样本数={len(train_labels)}，无法分配到每个客户端")

    empty_clients = [cid for cid in range(int(args.num_clients)) if len(client_indices[cid]) == 0]
    if empty_clients:
        donors = sorted(range(int(args.num_clients)), key=lambda c: len(client_indices[c]), reverse=True)
        for cid in empty_clients:
            for d in donors:
                if len(client_indices[d]) > 1:
                    client_indices[cid].append(client_indices[d].pop())
                    break

        empty_clients = [cid for cid in range(int(args.num_clients)) if len(client_indices[cid]) == 0]
        for cid in empty_clients:
            donor = max(range(int(args.num_clients)), key=lambda c: len(client_indices[c]))
            if len(client_indices[donor]) == 0:
                break
            client_indices[cid].append(client_indices[donor].pop())

    try:
        split_ratios = [float(x) for x in str(args.client_splits).split(",")]
    except Exception:
        split_ratios = [0.9, 0.1]
    train_ratio = float(split_ratios[0]) / max(1e-12, float(sum(split_ratios)))

    client_train_indices: List[List[int]] = []
    client_test_indices: List[List[int]] = []
    for cid in range(args.num_clients):
        idxs = client_indices[cid]
        if 0.0 < args.data_fraction < 1.0:
            k = max(1, int(len(idxs) * float(args.data_fraction)))
            idxs = idxs[:k]
        split_point = int(len(idxs) * train_ratio)
        if split_point <= 0:
            split_point = 1
        if split_point > len(idxs):
            split_point = len(idxs)
        client_train_indices.append(idxs[:split_point])
        client_test_indices.append(idxs[split_point:])

    client_capacities = generate_client_resources(args.num_clients, args.heterogeneity_type, args.allocation_ratio)
    num_total_layers = len(global_lora_layers)

    method = str(args.method).lower()
    server = RobertaFedHarpServer(
        model=global_model,
        lora_layers=global_lora_layers,
        num_clients=args.num_clients,
        allocation_ratio=args.allocation_ratio,
        aggregation_lr=args.aggregation_lr,
        struct_to_data_rounds=args.struct_to_data_rounds,
        seed=args.seed,
        method=args.method,
        fedharp_matrix_mode=("adaptive" if str(getattr(args, "fedharp_matrix_mode", "a")).lower() == "auto" else getattr(args, "fedharp_matrix_mode", "a")),
        allocation_temperature=float(args.allocation_temperature),
        coverage_prior_weight=float(args.coverage_prior_weight),
        sample_weight_power=float(args.sample_weight_power),
    )
    if hasattr(server, "client_layer_counts"):
        for cid, cap in enumerate(client_capacities):
            server.client_layer_counts[cid] = max(1, int(num_total_layers * float(cap)))

    clients: List[TextFedHarpClient] = []
    client_sample_counts: Dict[int, int] = {}
    eval_on = str(getattr(args, "eval_on", "val")).lower().strip()
    if eval_on == "client_holdout":
        print("Eval split: client_holdout (来自训练集内部切分，不等同于 GLUE validation)")
    else:
        print("Eval split: val (GLUE validation)")
    for cid in range(args.num_clients):
        train_ds = SST2ClientDataset(encoded_train, client_train_indices[cid])
        if eval_on == "client_holdout":
            test_ds = SST2ClientDataset(encoded_train, client_test_indices[cid])
            if len(test_ds) == 0:
                test_ds = SST2ClientDataset(encoded_val, list(range(len(encoded_val.labels))))
        else:
            val_chunks = np.array_split(np.arange(len(encoded_val.labels)), int(args.num_clients))
            val_idx = [int(x) for x in val_chunks[cid].tolist()]
            if len(val_idx) == 0:
                val_idx = list(range(len(encoded_val.labels)))
            test_ds = SST2ClientDataset(encoded_val, val_idx)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

        client_model, client_lora_layers = _clone_model_and_layers(global_model, global_lora_layers)
        client_model.to(device)
        client_sample_counts[cid] = len(train_ds)

        clients.append(
            TextFedHarpClient(
                client_id=cid,
                model=client_model,
                lora_layers=client_lora_layers,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                lr=args.client_lr,
                method=args.method,
                warmup_steps=0,
                warmup_lr=args.client_lr,
            )
        )

    clients_per_round = args.clients_per_round or args.num_clients
    best_accuracy = 0.0
    early_stop_metric = str(getattr(args, "early_stop_metric", "accuracy")).lower().strip()
    early_stopper = RoundEarlyStopping(
        patience=int(getattr(args, "early_stop_patience", 0)),
        min_delta=float(getattr(args, "early_stop_min_delta", 0.05)),
        min_rounds=int(getattr(args, "early_stop_min_rounds", 8)),
        mode="min" if early_stop_metric == "loss" else "max",
    )
    print(
        "Early stopping: "
        f"enabled={early_stopper.enabled}, metric={early_stop_metric}, "
        f"patience={early_stopper.patience}, min_delta={early_stopper.min_delta}, "
        f"min_rounds={early_stopper.min_rounds}"
    )
    layer_names = list(server.layer_names)
    if hasattr(server, "client_layer_counts"):
        client_max_layers = {cid: int(server.client_layer_counts.get(cid, len(layer_names))) for cid in range(args.num_clients)}
    else:
        client_max_layers = {cid: len(layer_names) for cid in range(args.num_clients)}

    fedharp_matrix_mode = str(getattr(args, "fedharp_matrix_mode", "a")).lower().replace("-", "_")
    if fedharp_matrix_mode == "auto":
        fedharp_matrix_mode = "adaptive"
    if fedharp_matrix_mode not in {"a", "b", "ab", "adaptive"}:
        fedharp_matrix_mode = "adaptive"
    fedharp_upload_B = fedharp_matrix_mode in {"b", "ab", "adaptive"}
    b_train_every = int(args.b_train_every)
    client_personal_A_states = {cid: clients[cid].get_all_A_matrices() for cid in range(args.num_clients)}
    client_personal_B_states = {cid: clients[cid].get_all_B_matrices() for cid in range(args.num_clients)}

    # E3 shock probe 初始化（默认关闭）
    shock_probe_on = bool(getattr(args, "shock_probe", False))
    probe_state = _ShockProbeState() if shock_probe_on else None
    projection_fork_count = 0
    allocation_trace_replay = {}
    trace_path = getattr(args, "allocation_trace_replay_path", None)
    online_trace_replay = bool(getattr(args, "allocation_trace_replay_online", False))
    if online_trace_replay and not trace_path:
        raise ValueError("online replay requires allocation_trace_replay_path")
    if getattr(args, "allocation_trace_dump_path", None) and trace_path:
        raise ValueError("trace dump and replay cannot both be enabled")
    if trace_path and not online_trace_replay:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    round_id = int(record["round"])
                    if round_id in allocation_trace_replay:
                        raise ValueError(f"duplicate round in allocation trace: {round_id}")
                    allocation_trace_replay[round_id] = record
        expected_rounds = set(range(1, int(args.num_rounds) + 1))
        if set(allocation_trace_replay) != expected_rounds:
            raise ValueError(
                f"allocation trace rounds mismatch: got={sorted(allocation_trace_replay)} "
                f"expected={sorted(expected_rounds)}"
            )
        print(f"[allocation_replay] loaded {len(allocation_trace_replay)} rounds from {trace_path}")
    if trace_path and online_trace_replay:
        print(f"[allocation_replay] online mode waiting on {trace_path}")
    # [S3c 2026-10-11] 交叉反事实开关：dump 本臂张量 / load 另一臂张量
    global CROSSOVER_DUMP, CROSSOVER_LOAD
    CROSSOVER_DUMP = getattr(args, "crossover_dump", None) if shock_probe_on else None
    CROSSOVER_LOAD = None
    if shock_probe_on and getattr(args, "crossover_load", None):
        CROSSOVER_LOAD = _crossover_load_file(args.crossover_load)
    probe_batches = {}
    if shock_probe_on:
        for cid in range(args.num_clients):
            probe_batches[cid] = _build_fixed_probe_batch(
                clients[cid], int(args.shock_probe_samples)
            )
        print(
            f"[shock_probe] enabled: dir={args.shock_probe_dir}, "
            f"fixed probe samples per client={args.shock_probe_samples}; "
            "events record tau/L_pre/L_sync/L_after/E_fro/s_rel/drift/b_grad_norm"
        )

    for round_num in range(1, args.num_rounds + 1):
        round_start_time = time.time()

        online_record = None
        if online_trace_replay:
            wait_started = time.time()
            expected_round = int(round_num)
            while True:
                try:
                    with open(trace_path, "r", encoding="utf-8") as trace_file:
                        records = [json.loads(line) for line in trace_file if line.strip() and line.endswith("\n")]
                except FileNotFoundError:
                    records = []
                if len(records) >= expected_round:
                    online_record = records[expected_round - 1]
                    if int(online_record.get("round", -1)) != expected_round:
                        raise ValueError(f"online allocation trace out of order at round {expected_round}")
                    break
                if time.time() - wait_started > int(getattr(args, "allocation_trace_wait_seconds", 10800)):
                    raise TimeoutError(f"timed out waiting for allocation trace round {expected_round}: {trace_path}")
                time.sleep(2.0)
            selected_clients = np.asarray(online_record["selected"], dtype=np.int64)
            print(f"[allocation_replay] consumed round={expected_round}")
        else:
            selected_clients = np.random.choice(
                args.num_clients, size=min(clients_per_round, args.num_clients), replace=False
            )
        if allocation_trace_replay:
            selected_clients = np.asarray(
                allocation_trace_replay[int(round_num)]["selected"], dtype=np.int64
            )
        needs_b = fedharp_upload_B
        if online_record is not None:
            server.current_round += 1
            allocation_map = {int(cid): set(layers) for cid, layers in online_record["allocation"].items()}
            if set(allocation_map) != set(range(int(args.num_clients))):
                raise ValueError(f"online trace client set mismatch at round {round_num}")
            server._record_allocation_coverage(allocation_map)
            for layers in allocation_map.values():
                for name in layers:
                    server.layer_selection_counts[name] += 1
        else:
            allocation_map = server.start_round()

        if allocation_trace_replay:
            trace_allocation = allocation_trace_replay[int(round_num)]["allocation"]
            allocation_map = {
                int(cid): set(layers) for cid, layers in trace_allocation.items()
            }
            if set(allocation_map) != set(range(int(args.num_clients))):
                raise ValueError(f"allocation trace client set mismatch at round {round_num}")

        expected_modules = int(getattr(args, "expected_allocation_modules", 0))
        if expected_modules > 0:
            if set(allocation_map) != set(range(int(args.num_clients))):
                raise ValueError(f"allocation client set mismatch at round {round_num}")
            for allocated_cid, names in allocation_map.items():
                if len(names) != expected_modules or not set(names).issubset(server.layer_names):
                    raise ValueError(f"wrong active module budget: round={round_num}, client={allocated_cid}, count={len(names)}")

        dump_path = getattr(args, "allocation_trace_dump_path", None)
        if dump_path:
            os.makedirs(os.path.dirname(os.path.abspath(dump_path)), exist_ok=True)
            record = {
                "round": int(round_num),
                "selected": [int(c) for c in selected_clients],
                "allocation": {int(k): sorted(v) for k, v in allocation_map.items()},
            }
            with open(dump_path, "a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                trace_file.flush()
                os.fsync(trace_file.fileno())

        # per-round 分配记录（探针启用时，审计用）
        if shock_probe_on:
            probe_state.round_allocations.append({
                "round": int(round_num),
                "selected": [int(c) for c in selected_clients],
                "allocation": {int(k): sorted(v) for k, v in allocation_map.items()},
            })

        global_A = server.get_global_A_matrices()
        global_B = server.get_global_B_matrices() if needs_b else None
        downlink_A_mb = _A_matrices_nbytes(global_A) / (1024.0 * 1024.0)
        downlink_B_mb = (_A_matrices_nbytes(global_B) / (1024.0 * 1024.0)) if global_B is not None else 0.0

        client_updates: Dict[int, Dict[str, torch.Tensor]] = {}
        client_updates_B: Dict[int, Dict[str, torch.Tensor]] = {}
        client_metrics_ab = []

        for client_id in selected_clients:
            cid = int(client_id)
            client = clients[cid]
            allocated_layers = allocation_map[cid]
            projection_mode = str(getattr(args, "projection_mode", "none")).lower().strip()
            # Evaluation restores the live client module to the current
            # global A. Projection must use the retained per-client snapshot
            # from the preceding local round, otherwise A_old is replaced by
            # A_new and the stale mismatch is artificially zero.
            pre_sync_A = (
                {name: tensor.clone() for name, tensor in client_personal_A_states.get(cid, client.get_all_A_matrices()).items()}
                if projection_mode != "none" else None
            )
            pre_sync_B = (
                {name: tensor.clone() for name, tensor in client_personal_B_states.get(cid, client.get_all_B_matrices()).items()}
                if projection_mode != "none" else None
            )

            client_global_A = global_A
            client_global_B = global_B
            # E3 探针：A 同步前判定事件块、测 L_pre、快照 A/B
            probe_evt = None
            if shock_probe_on:
                probe_evt = _shock_probe_begin(
                    client=client, cid=cid, round_num=round_num,
                    allocated_layers=allocated_layers,
                    stale_map=server.staleness.get(int(cid), {}),
                    b_train_every=b_train_every, projection_mode=projection_mode,
                    state=probe_state, probe_batch=probe_batches[cid],
                )
            # 探针开启且有事件时把同步拆为 (1) 仅应用 A -> 测 L_A_sync -> (2) 应用 B（若有）。
            # 数值等价：receive(A,None) 后再 set_B(B) == receive(A,B)。
            split_sync = shock_probe_on and probe_evt is not None
            if split_sync:
                client.receive_global_matrices(client_global_A, None, method=args.method)
                _shock_probe_post_a_sync(probe_evt, client)
                if client_global_B is not None:
                    client.set_B_matrices(client_global_B)
            else:
                client.receive_global_matrices(client_global_A, client_global_B, method=args.method)
            sync_snapshot_A = {
                name: tensor.clone() for name, tensor in client.get_all_A_matrices().items()
            }
            sync_snapshot_B = {
                name: tensor.clone() for name, tensor in client.get_all_B_matrices().items()
            }
            pre_train_A = {
                name: tensor.clone() for name, tensor in sync_snapshot_A.items()
            }
            client_downlink_A_mb = _A_matrices_nbytes(client_global_A) / (1024.0 * 1024.0)
            client_downlink_B_mb = (
                _A_matrices_nbytes(client_global_B) / (1024.0 * 1024.0)
                if client_global_B is not None else 0.0
            )

            # 投影触发块：本轮重激活（allocated 且 tau >= q）的 stale 块执行闭式 B 输运
            b_only_layers = set()
            if b_train_every > 0 and projection_mode in {"projection", "projection_1step"}:
                stale_map = server.staleness.get(int(cid), {})
                b_only_layers = {
                    name for name, tau in stale_map.items()
                    if int(tau) >= int(b_train_every) and name in allocated_layers
                }
            if probe_evt is not None:
                _shock_probe_post_sync(probe_evt, client)
                # 2x2 交叉反事实：必须在投影之前执行，
                # 函数内部测完会精确还原到同步后状态，故不影响后续投影与训练。
                if CROSSOVER_LOAD is not None:
                    _shock_probe_crossover(probe_evt, client)
            if len(b_only_layers) > 0:
                if pre_sync_A is None or pre_sync_B is None:
                    raise RuntimeError("projection mode requires pre-sync A/B snapshots")
                projected_B, projection_stats = project_factor_dicts(
                    pre_sync_A,
                    pre_sync_B,
                    client.get_all_A_matrices(),
                    client.get_all_B_matrices(),
                    b_only_layers,
                    ridge=float(getattr(args, "projection_ridge", 1e-6)),
                )
                client.set_B_matrices(projected_B)
                if (bool(getattr(args, "projection_fork_probe", False))
                        and projection_fork_count < int(args.projection_fork_limit)):
                    completed = run_fork_probe(
                        client=client, round_num=round_num, cid=cid,
                        allocated_layers=allocated_layers, old_A=pre_sync_A, old_B=pre_sync_B,
                        stale_B=sync_snapshot_B, projected_B=projected_B,
                        target_layers=b_only_layers, output_dir=args.projection_fork_dir,
                        seed=args.seed, steps=int(args.projection_fork_steps),
                        eval_samples=int(args.projection_fork_eval_samples),
                    )
                    projection_fork_count += int(completed)
                residual_before = sum(v["residual_before"] for v in projection_stats.values())
                residual_after = sum(v["residual_after"] for v in projection_stats.values())
                try:
                    with open(log_path, "a", encoding="utf-8") as _projection_log:
                        _projection_log.write(
                            f"round={round_num}, type=projection_event, client={int(cid)}, "
                            f"projected_layers={len(projection_stats)}, "
                            f"residual_before={residual_before:.8f}, "
                            f"residual_after={residual_after:.8f}, "
                            f"ridge={float(getattr(args, 'projection_ridge', 1e-6)):.8g}, "
                            f"refine_steps={1 if projection_mode == 'projection_1step' else 0}\n"
                        )
                except Exception as _projection_exc:
                    print(f"[warning] failed to record projection_event: {_projection_exc}")
                if projection_mode == "projection_1step":
                    new_b_sensitivity = client.train_B_only(
                        b_only_layers,
                        num_epochs=1,
                        max_steps=1,
                    )
                    try:
                        with open(log_path, "a", encoding="utf-8") as _projection_log:
                            _projection_log.write(
                                f"round={round_num}, type=projection_refine_event, client={int(cid)}, "
                                f"refine_steps=1, corrected_layers={len(b_only_layers)}\n"
                            )
                    except Exception:
                        pass
                else:
                    new_b_sensitivity = {name: 0.0 for name in client.lora_layers.keys()}
                server.update_client_sensitivities(int(cid), new_b_sensitivity)
                if probe_evt is not None:
                    _shock_probe_post_lpc(
                        probe_evt, client, b_only_layers, new_b_sensitivity,
                        corrector_kind=projection_mode,
                    )

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            delta_A, delta_B, b_sensitivity = client.local_train(
                allocated_layers=allocated_layers,
                num_epochs=args.num_epochs,
                method=args.method,
                matrix_update_mode=fedharp_matrix_mode,
                adaptive_matrix_threshold=float(getattr(args, "adaptive_matrix_threshold", 1.25)),
            )
            train_time_sec = time.time() - t0
            if torch.cuda.is_available() and device.type == "cuda":
                peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            else:
                peak_mem_gb = 0.0

            uplink_A_mb = _delta_A_nbytes(delta_A) / (1024.0 * 1024.0)
            uplink_B_mb = 0.0
            if needs_b:
                if delta_B is None:
                    raise ValueError(f"{method} 需要上传 B")
                uplink_B_mb = _delta_B_nbytes(delta_B) / (1024.0 * 1024.0)
            comm_mb = client_downlink_A_mb + client_downlink_B_mb + uplink_A_mb + uplink_B_mb

            train_samples = len(client.train_loader.dataset)
            trainable_numel = _ab_trainable_param_numel(client.lora_layers, allocated_layers)
            compute_tflops = (3.0 * float(trainable_numel) * float(train_samples) * float(args.num_epochs)) / 1e12
            first_client_layer = next(iter(client.lora_layers.values()), None)
            active_lora_rank = int(first_client_layer.lora_A.shape[0]) if first_client_layer is not None else 0

            client_metrics_ab.append(
                {
                    "round": round_num,
                    "client": int(cid),
                    "phase": "ab_train",
                    "lora_rank": int(active_lora_rank),
                    "compute_tflops": float(compute_tflops),
                    "peak_mem_gb": float(peak_mem_gb),
                    "comm_mb": float(comm_mb),
                    "train_time_sec": float(train_time_sec),
                    "downlink_mb": float(client_downlink_A_mb + client_downlink_B_mb),
                    "uplink_mb": float(uplink_A_mb + uplink_B_mb),
                }
            )

            client_updates[cid] = delta_A
            if needs_b:
                client_updates_B[cid] = delta_B
            server.update_client_sensitivities(cid, b_sensitivity)
            client_personal_A_states[cid] = client.get_all_A_matrices()
            client_personal_B_states[cid] = client.get_all_B_matrices()

            # 探针收尾：last_active 更新 + 事件落盘
            if shock_probe_on:
                _la = probe_state.last_active_round.setdefault(int(cid), {})
                for _name in allocated_layers:
                    _la[_name] = int(round_num)
                if probe_evt is not None:
                    _shock_probe_finalize(probe_evt, probe_state)

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

        if len(client_metrics_ab) > 0:
            avg_compute = float(np.mean([m["compute_tflops"] for m in client_metrics_ab]))
            avg_mem = float(np.mean([m["peak_mem_gb"] for m in client_metrics_ab]))
            avg_comm = float(np.mean([m["comm_mb"] for m in client_metrics_ab]))
            avg_time = float(np.mean([m["train_time_sec"] for m in client_metrics_ab]))
            ranks = [int(m["lora_rank"]) for m in client_metrics_ab]
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=round_metrics, phase=ab_train, "
                        f"avg_lora_rank={float(np.mean(ranks)):.4f}, min_lora_rank={min(ranks)}, max_lora_rank={max(ranks)}, "
                        f"avg_compute_tflops={avg_compute:.6f}, avg_peak_mem_gb={avg_mem:.6f}, "
                        f"avg_comm_mb={avg_comm:.6f}, avg_train_time_sec={avg_time:.6f}\n"
                    )
            except Exception as e:
                print(f"[警告] 无法写入日志文件 {log_path}: {e}")

        if needs_b:
            server.aggregate_matrices(
                client_updates_A=client_updates,
                client_updates_B=client_updates_B,
                client_sample_counts=client_sample_counts,
                allocation_map=allocation_map,
            )
        else:
            server.aggregate_A_matrices(
                client_updates=client_updates,
                client_sample_counts=client_sample_counts,
                allocation_map=allocation_map,
            )

        server.update_staleness(allocation_map)

        if round_num % args.eval_every == 0 or round_num == args.num_rounds:
            total_weighted_correct = 0.0
            total_weighted_loss = 0.0
            total_samples = 0
            eval_scope = str(getattr(args, "eval_scope", "global_test")).lower().strip()
            global_eval_A = server.get_global_A_matrices()
            global_eval_B = server.get_global_B_matrices() if needs_b else None
            if fedharp_matrix_mode == "a":
                eval_adapter_policy = "global_A+local_B"
            elif fedharp_matrix_mode == "b":
                eval_adapter_policy = "local_A+global_B"
            else:
                eval_adapter_policy = "global_A+global_B"
            print(f"Evaluation policy: adapter={eval_adapter_policy}, classifier=client_local")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=eval_policy, "
                        f"adapter={eval_adapter_policy}, classifier=client_local\n"
                    )
            except Exception as e:
                print(f"[warning] failed to write evaluation policy: {e}")
            for cid in range(args.num_clients):
                client = clients[cid]
                personal_A = client_personal_A_states.get(cid, client.get_all_A_matrices())
                personal_B = client_personal_B_states.get(cid, client.get_all_B_matrices())
                if fedharp_matrix_mode == "a":
                    eval_A_source, eval_B_source = global_eval_A, personal_B
                elif fedharp_matrix_mode == "b":
                    eval_A_source, eval_B_source = personal_A, global_eval_B
                else:
                    eval_A_source, eval_B_source = global_eval_A, global_eval_B
                eval_A, eval_B = eval_A_source, eval_B_source
                # 探针保护：评估按 eval 策略覆写客户端 A/B，会提前
                # 吸收下一轮 receive 的 A 同步冲击（eval_every=1 时 L_pre≡L_sync、E≡0，
                # 已在 20261010 smoke 实测确认）。探针启用时保存客户端自身 A/B、评估后
                # 恢复——不改变评估结果，也不改变训练语义（receive 会设置相同的值）。
                _saved_client_A = client.get_all_A_matrices() if shock_probe_on else None
                _saved_client_B = client.get_all_B_matrices() if shock_probe_on else None
                client.set_A_matrices(eval_A)
                if eval_B is not None:
                    client.set_B_matrices(eval_B)
                if client.test_loader is None or len(client.test_loader.dataset) == 0:
                    if shock_probe_on:
                        client.set_A_matrices(_saved_client_A)
                        client.set_B_matrices(_saved_client_B)
                    continue
                results = client.evaluate()
                acc = float(results["accuracy"])
                loss = float(results["loss"])
                n = len(client.test_loader.dataset)
                if args.client_eval_jsonl:
                    with open(args.client_eval_jsonl, "a", encoding="utf-8") as client_eval_file:
                        client_eval_file.write(json.dumps({
                            "seed": int(args.seed), "round": int(round_num), "client": int(cid),
                            "accuracy": acc, "loss": loss, "samples": n,
                        }) + "\n")
                total_weighted_correct += (acc / 100.0) * n
                total_weighted_loss += loss * n
                total_samples += n
                if shock_probe_on:
                    client.set_A_matrices(_saved_client_A)
                    client.set_B_matrices(_saved_client_B)

            if total_samples > 0:
                global_acc = 100.0 * total_weighted_correct / total_samples
                global_loss = total_weighted_loss / total_samples
            else:
                global_acc = 0.0
                global_loss = 0.0

            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"round={round_num}, type={eval_scope}, accuracy={global_acc:.4f}, loss={global_loss:.6f}\n")
            except Exception as e:
                print(f"[警告] 无法写入日志文件 {log_path}: {e}")

            if global_acc > best_accuracy:
                best_accuracy = global_acc

            monitored_value = global_loss if early_stop_metric == "loss" else global_acc
            early_decision = early_stopper.update(monitored_value, round_num)
            if early_decision.stop:
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"round={round_num}, type=early_stop, metric={early_stop_metric}, "
                            f"value={float(monitored_value):.6f}, "
                            f"best_value={float(early_decision.best_value):.6f}, "
                            f"best_round={int(early_decision.best_round)}, "
                            f"bad_evaluations={int(early_decision.bad_evaluations)}, "
                            f"patience={int(early_stopper.patience)}, "
                            f"min_delta={float(early_stopper.min_delta):.6f}, "
                            f"min_rounds={int(early_stopper.min_rounds)}\n"
                        )
                except Exception as e:
                    print(f"[warning] failed to write early-stop event: {e}")
                print(
                    f"Early stopping at round {round_num}: metric={early_stop_metric}, "
                    f"best={early_decision.best_value:.4f} (round {early_decision.best_round}), "
                    f"bad_evaluations={early_decision.bad_evaluations}"
                )
                break

        _ = time.time() - round_start_time

    # shock probe 收尾：写出事件 JSONL + 分配 trace + summary
    if shock_probe_on:
        os.makedirs(args.shock_probe_dir, exist_ok=True)
        events_path = os.path.join(args.shock_probe_dir, "shock_probe_events.jsonl")
        with open(events_path, "w", encoding="utf-8") as f:
            for evt in probe_state.events:
                f.write(json.dumps(evt, ensure_ascii=True) + "\n")
        if CROSSOVER_DUMP is not None and _CROSSOVER_BUF:
            np.savez_compressed(CROSSOVER_DUMP, **_CROSSOVER_BUF)
            print("[crossover] dumped %d arrays -> %s"
                  % (len(_CROSSOVER_BUF), CROSSOVER_DUMP))
        rounds_path = os.path.join(args.shock_probe_dir, "allocation_trace.jsonl")
        with open(rounds_path, "w", encoding="utf-8") as f:
            for rec in probe_state.round_allocations:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        _deltas = [e["delta_L"] for e in probe_state.events if e.get("delta_L") is not None]
        _benefits = [e["lpc_benefit"] for e in probe_state.events if e.get("lpc_benefit") is not None]
        _projection_events = [
            e for e in probe_state.events
            if e.get("corrector_kind") == "projection" and e.get("projection_applied")
            and e.get("projection_abs_jump_reduction") is not None
        ]
        # [S3b 2026-10-11] 仅 A-sync 冲击统计（native/ab 同口径可比量）
        _deltas_a = [e["delta_L_A_only"] for e in probe_state.events if e.get("delta_L_A_only") is not None]
        summary = {
            "type": "shock_probe_summary",
            "num_events": len(probe_state.events),
            "num_events_with_lpc": sum(1 for e in probe_state.events if e.get("L_after") is not None),
            "num_event_blocks_total": sum(e["num_event_blocks"] for e in probe_state.events),
            "mean_delta_L": float(np.mean(_deltas)) if _deltas else None,
            "mean_delta_L_A_only": float(np.mean(_deltas_a)) if _deltas_a else None,
            "mean_lpc_benefit": float(np.mean(_benefits)) if _benefits else None,
            "num_events_with_projection": len(_projection_events),
            "num_projection_events_abs_jump_reduced": sum(
                1 for e in _projection_events if e["projection_abs_jump_reduced"]
            ),
            "projection_abs_jump_reduction_rate": (
                float(np.mean([e["projection_abs_jump_reduced"] for e in _projection_events]))
                if _projection_events else None
            ),
            "mean_abs_projection_jump_before": (
                float(np.mean([abs(e["projection_jump_before"]) for e in _projection_events]))
                if _projection_events else None
            ),
            "mean_abs_projection_jump_after": (
                float(np.mean([abs(e["projection_jump_after"]) for e in _projection_events]))
                if _projection_events else None
            ),
            "mean_projection_abs_jump_reduction": (
                float(np.mean([e["projection_abs_jump_reduction"] for e in _projection_events]))
                if _projection_events else None
            ),
        }
        with open(os.path.join(args.shock_probe_dir, "summary.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=True) + "\n")
        print(
            f"[shock_probe] events={summary['num_events']} "
            f"(with_corrector={summary['num_events_with_lpc']}, "
            f"with_projection={summary['num_events_with_projection']}) -> {events_path}"
        )

    print(f"Best local-weighted accuracy (personalized): {best_accuracy:.2f}%")
    if (
        hasattr(server, "uses_partial_layer_allocation")
        and server.uses_partial_layer_allocation()
    ):
        server.print_underallocated_coverage_stats()
    print(f"Log saved to: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
