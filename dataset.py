"""
CIFAR-10 dataset with Non-IID partitioning using Dirichlet distribution.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import csv
from PIL import Image


class CIFAR10Dataset:
    """CIFAR-10 dataset handler with Non-IID partitioning"""
    
    def __init__(
        self,
        data_dir: str = "./data",
        alpha: float = 0.5,
        num_clients: int = 10,
        seed: int = 42,
        splits: List[float] = None,
        data_partition: str = "dirichlet"
    ):
        """
        Initialize CIFAR-10 dataset with Non-IID partitioning.
        
        Args:
            data_dir: Directory to store/download CIFAR-10
            alpha: Dirichlet distribution parameter (lower = more Non-IID)
            num_clients: Number of clients
            seed: Random seed for reproducibility
            data_partition: 'dirichlet' for Non-IID or 'uniform' for class-balanced IID
        """
        self.data_dir = data_dir
        self.alpha = alpha
        self.num_clients = num_clients
        self.seed = seed
        self.dataset_name = "cifar10"
        self.num_classes = 10
        self.data_partition = self._normalize_partition_name(data_partition)
        # 本地 train / test 划分比例，例如 [0.9, 0.1]
        if splits is None:
            splits = [0.9, 0.1]
        if len(splits) < 2:
            raise ValueError("splits 必须至少包含两个值，例如 [0.9, 0.1]")
        splits = np.array(splits, dtype=float)
        if splits.sum() <= 0:
            raise ValueError("splits 之和必须大于 0")
        self.splits = (splits / splits.sum()).tolist()
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Download and load CIFAR-10
        self._load_datasets()
        
        # Partition data
        # 每个客户端内部再按 self.splits 划分出本地 train / test
        self.client_datasets = self._partition_data()

    def _normalize_partition_name(self, data_partition: str) -> str:
        p = str(data_partition or "dirichlet").lower().replace("-", "_").strip()
        if p in {"uniform", "iid", "balanced"}:
            return "uniform"
        if p in {"dirichlet", "non_iid", "noniid", "non_iid_dirichlet"}:
            return "dirichlet"
        raise ValueError(f"Unknown data_partition={data_partition}; expected 'dirichlet' or 'uniform'")
    
    def _load_datasets(self):
        """Load CIFAR-10 train and test datasets"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            transforms.Resize(224)  # Resize for ViT
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            transforms.Resize(224)  # Resize for ViT
        ])
        
        self.train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform_test
        )
        
        print(f"Loaded CIFAR-10: {len(self.train_dataset)} train, {len(self.test_dataset)} test samples")
    
    def _get_train_labels(self) -> np.ndarray:
        num_samples = len(self.train_dataset)
        if hasattr(self.train_dataset, "targets"):
            return np.asarray(self.train_dataset.targets)
        elif hasattr(self.train_dataset, "labels"):
            return np.asarray(self.train_dataset.labels)
        return np.array([self.train_dataset[i][1] for i in range(num_samples)])

    def _build_dirichlet_client_indices(self, labels: np.ndarray) -> List[List[int]]:
        client_indices = [[] for _ in range(self.num_clients)]

        for class_idx in range(int(self.num_classes)):
            class_indices = np.where(labels == class_idx)[0]
            np.random.shuffle(class_indices)

            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            proportions = (proportions * len(class_indices)).astype(int)
            proportions[-1] = len(class_indices) - sum(proportions[:-1])

            start_idx = 0
            for client_idx in range(self.num_clients):
                end_idx = start_idx + proportions[client_idx]
                client_indices[client_idx].extend(class_indices[start_idx:end_idx])
                start_idx = end_idx

        return client_indices

    def _build_uniform_client_indices(self, labels: np.ndarray) -> List[List[int]]:
        client_indices = [[] for _ in range(self.num_clients)]

        for class_idx in range(int(self.num_classes)):
            class_indices = np.where(labels == class_idx)[0]
            np.random.shuffle(class_indices)
            chunks = np.array_split(class_indices, self.num_clients)
            for client_idx, chunk in enumerate(chunks):
                client_indices[client_idx].extend([int(x) for x in chunk.tolist()])

        return client_indices

    def _make_client_datasets_from_indices(self, labels: np.ndarray, client_indices: List[List[int]]) -> List[Dict[str, Subset]]:
        for client_idx in range(self.num_clients):
            np.random.shuffle(client_indices[client_idx])

        seen: set[int] = set()
        for client_idx in range(self.num_clients):
            idxs = list(client_indices[client_idx])
            dup = seen.intersection(idxs)
            if len(dup) > 0:
                example = next(iter(dup))
                raise ValueError(f"{self.dataset_name} 分配出现跨客户端重叠: client={client_idx} example_idx={example}")
            seen.update(idxs)
        
        client_datasets: List[Dict[str, Subset]] = []
        train_ratio = self.splits[0]
        for client_idx in range(self.num_clients):
            indices = client_indices[client_idx]
            num_client_samples = len(indices)
            split_point = int(num_client_samples * train_ratio)
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]

            overlap = set(train_indices).intersection(test_indices)
            if len(overlap) > 0:
                example = next(iter(overlap))
                raise ValueError(f"{self.dataset_name} 客户端本地 train/test 出现重叠: client={client_idx} example_idx={example}")
            
            client_datasets.append({
                "train": Subset(self.train_dataset, train_indices),
                "test": Subset(self.train_dataset, test_indices)
            })
        
        # Print statistics（基于整体客户端数据分布）
        self._print_partition_stats(labels, client_indices)
        return client_datasets

    def _partition_data(self) -> List[Dict[str, Subset]]:
        labels = self._get_train_labels()
        if self.data_partition == "uniform":
            client_indices = self._build_uniform_client_indices(labels)
        else:
            client_indices = self._build_dirichlet_client_indices(labels)
        return self._make_client_datasets_from_indices(labels, client_indices)

    def _partition_non_iid(self) -> List[Dict[str, Subset]]:
        return self._partition_data()
    
    def _print_partition_stats(self, labels: np.ndarray, client_indices: List[List[int]]):
        """Print statistics about the data partition"""
        print("\n" + "="*60)
        if self.data_partition == "uniform":
            title = f"Uniform Partition Statistics ({self.dataset_name}, class-balanced IID)"
        else:
            title = f"Non-IID Partition Statistics ({self.dataset_name}, Dirichlet alpha={self.alpha})"
        print(title)
        print("="*60)
        
        for client_idx in range(self.num_clients):
            client_labels = labels[client_indices[client_idx]]
            unique, counts = np.unique(client_labels, return_counts=True)
            label_dist = dict(zip(unique, counts))
            
            total = len(client_labels)
            dist_str = ", ".join([f"Class {k}: {v} ({100*v/total:.1f}%)" 
                                 for k, v in sorted(label_dist.items())])
            
            print(f"Client {client_idx}: {total} samples")
            print(f"  Distribution: {dist_str}")
        
        print("="*60 + "\n")

    def get_client_dataset(self, client_idx: int, split: str = "train") -> Subset:
        """Get dataset for a specific client and split ('train' or 'test')"""
        if client_idx < 0 or client_idx >= self.num_clients:
            raise ValueError(f"Client index must be in [0, {self.num_clients})")
        if split not in self.client_datasets[client_idx]:
            raise ValueError(f"Unknown split '{split}', expected one of {list(self.client_datasets[client_idx].keys())}")
        return self.client_datasets[client_idx][split]

    def get_test_dataset(self) -> Dataset:
        """Get test dataset"""
        return self.test_dataset

    def get_client_dataloader(
        self,
        client_idx: int,
        batch_size: int = 32,
        shuffle: bool = True,
        split: str = "train",
        fraction: float = 1.0
    ) -> DataLoader:
        """Get DataLoader for a specific client and split
        
        Args:
            client_idx: 客户端索引
            batch_size: 批大小
            shuffle: 是否打乱
            split: 使用的本地划分 ('train' 或 'test')
            fraction: 使用的数据比例 (0-1), 例如 0.01 表示使用 1% 的该客户端数据
        """
        base_dataset = self.get_client_dataset(client_idx, split=split)
        
        if 0.0 < fraction < 1.0:
            num_samples = len(base_dataset)
            subset_size = max(1, int(num_samples * fraction))
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            subset_indices = all_indices[:subset_size]
            dataset = Subset(base_dataset, subset_indices)
        else:
            dataset = base_dataset
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )

    def get_test_dataloader(self, batch_size: int = 100) -> DataLoader:
        """Get DataLoader for test dataset"""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )


class CIFAR100Dataset(CIFAR10Dataset):
    """CIFAR-100 dataset handler with Non-IID partitioning (Dirichlet)"""

    def __init__(
        self,
        data_dir: str = "./data",
        alpha: float = 0.5,
        num_clients: int = 10,
        seed: int = 42,
        splits: List[float] = None,
        data_partition: str = "dirichlet"
    ):
        super().__init__(
            data_dir=data_dir,
            alpha=alpha,
            num_clients=num_clients,
            seed=seed,
            splits=splits,
            data_partition=data_partition
        )

    def _load_datasets(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
            transforms.Resize(224)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
            transforms.Resize(224)
        ])

        self.train_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )

        self.test_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform_test
        )

        self.dataset_name = "cifar100"
        self.num_classes = 100

        print(f"Loaded CIFAR-100: {len(self.train_dataset)} train, {len(self.test_dataset)} test samples")
    
    def get_client_dataset(self, client_idx: int, split: str = "train") -> Subset:
        """Get dataset for a specific client and split ('train' or 'test')"""
        if client_idx < 0 or client_idx >= self.num_clients:
            raise ValueError(f"Client index must be in [0, {self.num_clients})")
        if split not in self.client_datasets[client_idx]:
            raise ValueError(f"Unknown split '{split}', expected one of {list(self.client_datasets[client_idx].keys())}")
        return self.client_datasets[client_idx][split]
    
    def get_test_dataset(self) -> Dataset:
        """Get test dataset"""
        return self.test_dataset
    
    def get_client_dataloader(
        self,
        client_idx: int,
        batch_size: int = 32,
        shuffle: bool = True,
        split: str = "train",
        fraction: float = 1.0
    ) -> DataLoader:
        """Get DataLoader for a specific client and split
        
        Args:
            client_idx: 客户端索引
            batch_size: 批大小
            shuffle: 是否打乱
            split: 使用的本地划分 ('train' 或 'test')
            fraction: 使用的数据比例 (0-1), 例如 0.01 表示使用 1% 的该客户端数据
        """
        base_dataset = self.get_client_dataset(client_idx, split=split)
        
        # 如果只使用部分数据，随机采样一个子集
        if 0.0 < fraction < 1.0:
            num_samples = len(base_dataset)
            subset_size = max(1, int(num_samples * fraction))
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            subset_indices = all_indices[:subset_size]
            dataset = Subset(base_dataset, subset_indices)
        else:
            dataset = base_dataset
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )
    
    def get_test_dataloader(self, batch_size: int = 100) -> DataLoader:
        """Get DataLoader for test dataset"""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )


def get_class_distribution(dataset: Dataset) -> Dict[int, int]:
    """Get class distribution in a dataset"""
    if isinstance(dataset, Subset):
        labels = [dataset.dataset[dataset.indices[i]][1] for i in range(len(dataset))]
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


class ChestXray8Dataset:
    def __init__(
        self,
        data_dir: str = "./data",
        alpha: float = 0.5,
        num_clients: int = 10,
        seed: int = 42,
        splits: List[float] = None,
        data_partition: str = "dirichlet",
    ):
        self.data_dir = data_dir
        self.alpha = alpha
        self.num_clients = num_clients
        self.seed = seed
        self.dataset_name = "chestxray8"
        self.num_classes = 15
        self.data_partition = self._normalize_partition_name(data_partition)
        if splits is None:
            splits = [0.9, 0.1]
        if len(splits) < 2:
            raise ValueError("splits 必须至少包含两个值，例如 [0.9, 0.1]")
        splits = np.array(splits, dtype=float)
        if splits.sum() <= 0:
            raise ValueError("splits 之和必须大于 0")
        self.splits = (splits / splits.sum()).tolist()

        np.random.seed(seed)
        torch.manual_seed(seed)

        self._load_datasets()
        self.client_datasets = self._partition_data()

    def _resolve_root(self) -> str:
        d = str(self.data_dir) if self.data_dir is not None else "./data"
        cand = os.path.join(d, "chestxray8")
        if os.path.isdir(cand):
            return cand
        return d

    def _find_csv_path(self, root: str) -> str:
        candidates = [
            os.path.join(root, "Data_Entry_2017.csv"),
            os.path.join(root, "data", "Data_Entry_2017.csv"),
            os.path.join(root, "Data_Entry_2017_v2020.csv"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(
            "未找到 ChestX-ray8 元数据 CSV（Data_Entry_2017.csv）。"
            f"请将数据集放到 {root} 或 {os.path.join(root, 'chestxray8')} 下，并包含 Data_Entry_2017.csv"
        )

    def _list_image_dir_candidates(self, root: str) -> List[str]:
        out: List[str] = []
        direct = [
            os.path.join(root, "images"),
            os.path.join(root, "images", "images"),
            os.path.join(root, "Images"),
            os.path.join(root, "Imgs"),
        ]
        for p in direct:
            if os.path.isdir(p):
                out.append(p)

        try:
            for name in os.listdir(root):
                full = os.path.join(root, name)
                if not os.path.isdir(full):
                    continue
                low = name.lower()
                if low.startswith("images_") or low.startswith("images"):
                    out.append(full)
                    maybe = os.path.join(full, "images")
                    if os.path.isdir(maybe):
                        out.append(maybe)
        except Exception:
            pass

        uniq: List[str] = []
        seen = set()
        for p in out:
            ap = os.path.abspath(p)
            if ap in seen:
                continue
            seen.add(ap)
            uniq.append(ap)
        return uniq

    def _resolve_image_path(self, fname: str) -> str:
        if fname in self._path_cache:
            return self._path_cache[fname]
        for d in self._image_dir_candidates:
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                self._path_cache[fname] = p
                return p
        raise FileNotFoundError(f"未找到图像文件: {fname} (searched_dirs={len(self._image_dir_candidates)})")

    def _parse_labels(self, raw: str) -> str:
        s = str(raw) if raw is not None else ""
        if s.strip() == "":
            return "No Finding"
        parts = [p.strip() for p in s.split("|") if p.strip() != ""]
        if len(parts) == 0:
            return "No Finding"
        if "No Finding" in parts:
            return "No Finding"
        return parts[0]

    def _load_rows(self, csv_path: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not isinstance(r, dict):
                    continue
                rows.append({k: ("" if v is None else str(v)) for k, v in r.items()})
        if len(rows) == 0:
            raise ValueError(f"CSV 为空或无法解析: {csv_path}")
        return rows

    def _load_split_lists(self, root: str) -> Tuple[Optional[set], Optional[set]]:
        train_list = None
        test_list = None
        train_path = os.path.join(root, "train_val_list.txt")
        test_path = os.path.join(root, "test_list.txt")
        if os.path.isfile(train_path) and os.path.isfile(test_path):
            with open(train_path, "r", encoding="utf-8") as f:
                train_list = set([ln.strip() for ln in f.read().splitlines() if ln.strip() != ""])
            with open(test_path, "r", encoding="utf-8") as f:
                test_list = set([ln.strip() for ln in f.read().splitlines() if ln.strip() != ""])
        return train_list, test_list

    def _patient_wise_split(self, rows: List[Dict[str, str]], seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        by_pid: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            pid = r.get("Patient ID", "").strip()
            if pid == "":
                pid = "unknown"
            by_pid.setdefault(pid, []).append(r)
        pids = list(by_pid.keys())
        rng = np.random.RandomState(int(seed))
        rng.shuffle(pids)
        n_train = int(round(0.8 * len(pids)))
        train_pids = set(pids[:n_train])
        train_rows: List[Dict[str, str]] = []
        test_rows: List[Dict[str, str]] = []
        for pid, group in by_pid.items():
            if pid in train_pids:
                train_rows.extend(group)
            else:
                test_rows.extend(group)
        return train_rows, test_rows

    def _make_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return transform_train, transform_test

    def _load_datasets(self):
        root = self._resolve_root()
        csv_path = self._find_csv_path(root)
        rows = self._load_rows(csv_path)
        train_list, test_list = self._load_split_lists(root)

        if train_list is not None and test_list is not None:
            tr_rows = [r for r in rows if r.get("Image Index", "").strip() in train_list]
            te_rows = [r for r in rows if r.get("Image Index", "").strip() in test_list]
        else:
            tr_rows, te_rows = self._patient_wise_split(rows, seed=int(self.seed) + 7)

        labels_vocab = [
            "No Finding",
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]
        label_to_idx = {name: i for i, name in enumerate(labels_vocab)}

        transform_train, transform_test = self._make_transforms()
        self._image_dir_candidates = self._list_image_dir_candidates(root)
        self._path_cache: Dict[str, str] = {}

        self.train_dataset = _ChestXray8TorchDataset(
            rows=tr_rows,
            label_to_idx=label_to_idx,
            resolve_image_path=self._resolve_image_path,
            parse_label=self._parse_labels,
            transform=transform_train,
        )
        self.test_dataset = _ChestXray8TorchDataset(
            rows=te_rows,
            label_to_idx=label_to_idx,
            resolve_image_path=self._resolve_image_path,
            parse_label=self._parse_labels,
            transform=transform_test,
        )
        print(
            f"Loaded ChestX-ray8: {len(self.train_dataset)} train, {len(self.test_dataset)} test samples (classes={self.num_classes})"
        )

    def _partition_non_iid(self) -> List[Dict[str, Subset]]:
        if str(getattr(self, "data_partition", "dirichlet")) == "uniform":
            labels = self._get_train_labels()
            client_indices = self._build_uniform_client_indices(labels)
            return self._make_client_datasets_from_indices(labels, client_indices)

        num_classes = int(self.num_classes)
        num_samples = len(self.train_dataset)

        if hasattr(self.train_dataset, "targets"):
            labels = np.asarray(self.train_dataset.targets)
        else:
            labels = np.array([self.train_dataset[i][1] for i in range(num_samples)])

        client_indices = [[] for _ in range(self.num_clients)]
        for class_idx in range(num_classes):
            class_indices = np.where(labels == class_idx)[0]
            np.random.shuffle(class_indices)
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            proportions = (proportions * len(class_indices)).astype(int)
            proportions[-1] = len(class_indices) - sum(proportions[:-1])
            start_idx = 0
            for client_idx in range(self.num_clients):
                end_idx = start_idx + proportions[client_idx]
                client_indices[client_idx].extend(class_indices[start_idx:end_idx])
                start_idx = end_idx

        for client_idx in range(self.num_clients):
            np.random.shuffle(client_indices[client_idx])

        seen: set[int] = set()
        for client_idx in range(self.num_clients):
            idxs = list(client_indices[client_idx])
            dup = seen.intersection(idxs)
            if len(dup) > 0:
                example = next(iter(dup))
                raise ValueError(f"{self.dataset_name} 分配出现跨客户端重叠: client={client_idx} example_idx={example}")
            seen.update(idxs)

        client_datasets: List[Dict[str, Subset]] = []
        train_ratio = self.splits[0]
        for client_idx in range(self.num_clients):
            indices = client_indices[client_idx]
            num_client_samples = len(indices)
            split_point = int(num_client_samples * train_ratio)
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]
            overlap = set(train_indices).intersection(test_indices)
            if len(overlap) > 0:
                example = next(iter(overlap))
                raise ValueError(f"{self.dataset_name} 客户端本地 train/test 出现重叠: client={client_idx} example_idx={example}")
            client_datasets.append({"train": Subset(self.train_dataset, train_indices), "test": Subset(self.train_dataset, test_indices)})

        self._print_partition_stats(labels, client_indices)
        return client_datasets

    def _print_partition_stats(self, labels: np.ndarray, client_indices: List[List[int]]):
        print("\n" + "=" * 60)
        if str(getattr(self, "data_partition", "dirichlet")) == "uniform":
            title = f"Uniform Partition Statistics ({self.dataset_name}, class-balanced IID)"
        else:
            title = f"Non-IID Partition Statistics ({self.dataset_name}, Dirichlet alpha={self.alpha})"
        print(title)
        print("=" * 60)
        for client_idx in range(self.num_clients):
            client_labels = labels[client_indices[client_idx]]
            unique, counts = np.unique(client_labels, return_counts=True)
            label_dist = dict(zip(unique, counts))
            total = len(client_labels)
            if total == 0:
                print(f"Client {client_idx}: 0 samples")
                continue
            dist_str = ", ".join([f"Class {k}: {v} ({100*v/total:.1f}%)" for k, v in sorted(label_dist.items())])
            print(f"Client {client_idx}: {total} samples")
            print(f"  Distribution: {dist_str}")
        print("=" * 60 + "\n")

    def get_client_dataset(self, client_idx: int, split: str = "train") -> Subset:
        if client_idx < 0 or client_idx >= self.num_clients:
            raise ValueError(f"Client index must be in [0, {self.num_clients})")
        if split not in self.client_datasets[client_idx]:
            raise ValueError(f"Unknown split '{split}', expected one of {list(self.client_datasets[client_idx].keys())}")
        return self.client_datasets[client_idx][split]

    def get_test_dataset(self) -> Dataset:
        return self.test_dataset

    def get_client_dataloader(
        self,
        client_idx: int,
        batch_size: int = 32,
        shuffle: bool = True,
        split: str = "train",
        fraction: float = 1.0,
    ) -> DataLoader:
        base_dataset = self.get_client_dataset(client_idx, split=split)
        if 0.0 < fraction < 1.0:
            num_samples = len(base_dataset)
            subset_size = max(1, int(num_samples * fraction))
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            subset_indices = all_indices[:subset_size]
            dataset = Subset(base_dataset, subset_indices)
        else:
            dataset = base_dataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

    def get_test_dataloader(self, batch_size: int = 100) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


class _ChestXray8TorchDataset(Dataset):
    def __init__(
        self,
        *,
        rows: List[Dict[str, str]],
        label_to_idx: Dict[str, int],
        resolve_image_path,
        parse_label,
        transform,
    ):
        self.rows = list(rows)
        self.label_to_idx = dict(label_to_idx)
        self.resolve_image_path = resolve_image_path
        self.parse_label = parse_label
        self.transform = transform

        self.targets: List[int] = []
        for r in self.rows:
            raw = r.get("Finding Labels", "")
            lab = self.parse_label(raw)
            if lab not in self.label_to_idx:
                lab = "No Finding"
            self.targets.append(int(self.label_to_idx[lab]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[int(idx)]
        fname = r.get("Image Index", "").strip()
        if fname == "":
            raise ValueError("样本缺少 Image Index")
        path = self.resolve_image_path(fname)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = int(self.targets[int(idx)])
        return img, y
