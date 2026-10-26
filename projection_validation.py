"""Measurement-only forks and deterministic long-dormancy schedules."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def make_pressure_trace(records, seed, hold_rounds=3):
    """Four banks of three Transformer blocks; nine dormant rounds on return."""
    first = records[0]
    banks = {}
    pool = sorted({name for row in records for names in row["allocation"].values() for name in names})
    for cid, names in first["allocation"].items():
        # The first ordinary allocation omits some blocks; obtain the complete
        # pool from the union across all recorded rounds for this client.
        groups = {}
        for name in pool:
            match = re.search(r"^(.*\.layer\.(\d+))\.", name)
            if not match:
                raise ValueError(f"not a Transformer block name: {name}")
            groups.setdefault(int(match.group(2)), []).append(name)
        if len(groups) != 12 or any(len(names) != 2 for names in groups.values()):
            raise ValueError(f"expected 12 complete query/value blocks for client {cid}")
        perm = list(sorted(groups))
        random.Random(20261022 + int(seed) * 1009 + int(cid)).shuffle(perm)
        banks[str(cid)] = [sorted(name for gid in perm[i:i+3] for name in groups[gid])
                           for i in range(0, 12, 3)]
    output = []
    for record in records:
        rd = int(record["round"])
        bank_index = ((rd - 1) // int(hold_rounds)) % 4
        output.append({
            "round": rd,
            "selected": [int(cid) for cid in record["selected"]],
            "allocation": {cid: choices[bank_index] for cid, choices in banks.items()},
        })
    audit_pressure_trace(output)
    return output


def audit_pressure_trace(records):
    last = {}
    gaps = []
    for row in records:
        rd = int(row["round"])
        for cid, names in row["allocation"].items():
            if len(names) != 6:
                raise ValueError(f"pressure allocation is not six LoRA modules: {rd}, {cid}")
            for name in names:
                key = (str(cid), name)
                if key in last and rd - last[key] > 1:
                    gap = rd - last[key] - 1
                    if gap < 8:
                        raise ValueError(f"pressure dormancy shorter than eight rounds: {key}, {gap}")
                    gaps.append(gap)
                last[key] = rd
    return {"reactivated_module_events": len(gaps), "minimum_dormant_rounds": min(gaps) if gaps else None}


def _rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def _restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(state[3])


def _rng_equal(a, b):
    return (a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
            and a[1][2:] == b[1][2:] and torch.equal(a[2], b[2])
            and len(a[3]) == len(b[3]) and all(torch.equal(x, y) for x, y in zip(a[3], b[3])))


def _snapshot(client):
    lora_ids = {id(t) for layer in client.lora_layers.values() for t in (layer.lora_A, layer.lora_B)}
    params = dict(client.model.named_parameters())
    mutable = {name: p.detach().clone() for name, p in params.items() if p.requires_grad or id(p) in lora_ids}
    return {
        "params": mutable,
        "flags": {name: p.requires_grad for name, p in params.items()},
        "grads": {name: p.grad.detach().clone() for name, p in params.items() if p.grad is not None},
        "buffers": {name: b.detach().clone() for name, b in client.model.named_buffers()},
        "modes": {name: module.training for name, module in client.model.named_modules()},
        "rng": _rng_state(),
        "optimizer": getattr(client, "optimizer", None),
    }


def _restore(client, state):
    with torch.no_grad():
        for name, p in client.model.named_parameters():
            if name in state["params"]:
                p.copy_(state["params"][name])
            p.requires_grad_(state["flags"][name])
            p.grad = state["grads"][name].clone() if name in state["grads"] else None
        for name, b in client.model.named_buffers():
            b.copy_(state["buffers"][name])
    for name, module in client.model.named_modules():
        module.training = state["modes"][name]
    client.optimizer = state["optimizer"]
    _restore_rng(state["rng"])


def _verify_restored(client, state):
    params = dict(client.model.named_parameters())
    assert all(torch.equal(params[name], value) for name, value in state["params"].items())
    assert all(p.requires_grad == state["flags"][name] for name, p in params.items())
    assert all((p.grad is None and name not in state["grads"]) or
               (p.grad is not None and name in state["grads"] and torch.equal(p.grad, state["grads"][name]))
               for name, p in params.items())
    assert all(torch.equal(b, state["buffers"][name]) for name, b in client.model.named_buffers())
    assert all(module.training == state["modes"][name] for name, module in client.model.named_modules())
    assert client.optimizer is state["optimizer"]
    assert _rng_equal(state["rng"], _rng_state())


def _fixed_holdout(client, count):
    dataset = client.test_loader.dataset
    n = min(int(count), len(dataset))
    if n < 1:
        raise ValueError("empty client holdout")
    samples = [dataset[index] for index in range(n)]
    inputs = {key: torch.stack([sample[0][key] for sample in samples]) for key in samples[0][0]}
    targets = torch.stack([sample[1] for sample in samples])
    train_dataset = client.train_loader.dataset
    train_indices = set(getattr(train_dataset, "provenance_indices", []))
    test_indices = set(getattr(dataset, "provenance_indices", []))
    # These runners use the same encoded training object for train/holdout.
    same_source = getattr(train_dataset, "provenance_encoded_id", None) == getattr(dataset, "provenance_encoded_id", None)
    if same_source and train_indices and test_indices and train_indices & test_indices:
        raise ValueError("client training and heldout indices overlap")
    return inputs, targets


def _logits(client, inputs, batch_size=32):
    client.model.eval()
    result = []
    with torch.no_grad():
        n = len(next(iter(inputs.values())))
        for start in range(0, n, batch_size):
            out = client.model(**{key: value[start:start+batch_size].to(client.device) for key, value in inputs.items()})
            result.append(getattr(out, "logits", out).detach().float().cpu())
    return torch.cat(result)


def _evaluate(client, inputs, targets, teacher):
    logits = _logits(client, inputs)
    return {
        "loss": float(F.cross_entropy(logits, targets).item()),
        "accuracy": float((logits.argmax(dim=-1) == targets).float().mean().item() * 100),
        "teacher_kl": float(F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(teacher, dim=-1), reduction="batchmean").item()),
    }


def _gradient_dot(client, inputs, targets, deltas):
    client.model.eval()
    layers = sorted(deltas)
    b_params = [client.lora_layers[name].lora_B for name in layers]
    for param in b_params:
        param.requires_grad_(True)
    dot = 0.0
    grad_sq = 0.0
    delta_sq = sum(float(value.square().sum().item()) for value in deltas.values())
    n = int(len(targets))
    for start in range(0, n, 32):
        end = min(start + 32, n)
        out = client.model(**{key: value[start:end].to(client.device) for key, value in inputs.items()})
        loss = F.cross_entropy(getattr(out, "logits", out), targets[start:end].to(client.device))
        gradients = torch.autograd.grad(loss, b_params, allow_unused=True)
        for name, grad in zip(layers, gradients):
            if grad is not None:
                dot += float((grad.detach() * deltas[name]).sum().item()) * (end-start) / n
                grad_sq += float(grad.detach().square().sum().item()) * (end-start) / n
    return {"grad_dot_delta": dot, "delta_B_norm": delta_sq ** 0.5,
            "rms_batch_gradient_norm": grad_sq ** 0.5}


def run_fork_probe(client, round_num, cid, allocated_layers, old_A, old_B, stale_B,
                   projected_B, target_layers, output_dir, seed, steps=5, eval_samples=128):
    """Replay the same five ordinary minibatches from the same firing state.

    All branches start with a fresh Adam, as in ordinary local training. The
    measurement restores parameters, gradients, modes, optimizer identity and
    Python/NumPy/CPU/CUDA RNG states exactly before returning.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {"seed": int(seed), "round": int(round_num), "client": int(cid),
              "k": int(steps), "target_layers": sorted(target_layers)}
    deltas = {name: projected_B[name] - stale_B[name] for name in sorted(target_layers)}
    delta_norm = sum(float(value.square().sum().item()) for value in deltas.values()) ** 0.5
    if delta_norm <= 1e-12 or len(client.train_loader) < steps:
        record.update(status="skipped", reason="zero_delta" if delta_norm <= 1e-12 else "fewer_than_k_train_batches")
        with (output_dir / "fork_skipped.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return False
    state = _snapshot(client)
    try:
        inputs_eval, targets_eval = _fixed_holdout(client, eval_samples)
        record["eval_samples"] = len(targets_eval)
        eval_digest = hashlib.sha256()
        for key in sorted(inputs_eval):
            eval_digest.update(inputs_eval[key].contiguous().numpy().tobytes())
        eval_digest.update(targets_eval.contiguous().numpy().tobytes())
        record["heldout_samples_sha256"] = eval_digest.hexdigest()
        iterator = iter(client.train_loader)
        try:
            minibatches = [next(iterator) for _ in range(steps)]
        finally:
            if hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()
            del iterator
        branch_rng = _rng_state()
        digest = hashlib.sha256()
        for batch, target in minibatches:
            for key in sorted(batch):
                digest.update(batch[key].contiguous().numpy().tobytes())
            digest.update(target.contiguous().numpy().tobytes())
        record["minibatch_sha256"] = digest.hexdigest()

        client.set_A_matrices(old_A)
        client.set_B_matrices(old_B)
        teacher = _logits(client, inputs_eval)
        _restore(client, state)
        client.set_B_matrices(stale_B)
        record["gradient_probe"] = _gradient_dot(client, inputs_eval, targets_eval, deltas)

        random_B = {name: value.clone() for name, value in stale_B.items()}
        random_delta_norms = {}
        for index, name in enumerate(sorted(deltas)):
            generator = torch.Generator(device=deltas[name].device)
            generator.manual_seed(int(seed) * 1000003 + int(round_num) * 101 + int(cid) * 1009 + index)
            noise = torch.randn(deltas[name].shape, generator=generator, device=deltas[name].device, dtype=deltas[name].dtype)
            noise *= torch.linalg.vector_norm(deltas[name]) / torch.linalg.vector_norm(noise).clamp_min(1e-30)
            random_B[name] = stale_B[name] + noise
            random_delta_norms[name] = float(torch.linalg.vector_norm(noise).item())
        record["per_block_projection_delta_norm"] = {name: float(torch.linalg.vector_norm(delta).item()) for name, delta in deltas.items()}
        record["per_block_random_delta_norm"] = random_delta_norms
        record["branches"] = {}
        for branch, b_values in (("stale", stale_B), ("projected", projected_B), ("random_equal_B_norm", random_B)):
            _restore(client, state)
            client.set_B_matrices(b_values)
            client._setup_layer_trainability(set(allocated_layers))
            optimizer = client._create_optimizer(warmup_mode=False)
            if optimizer is None:
                raise ValueError("fork branch has no optimizer")
            _restore_rng(branch_rng)
            curve = [{"step": 0, **_evaluate(client, inputs_eval, targets_eval, teacher)}]
            for step, (inputs, targets) in enumerate(minibatches, 1):
                client.model.train()
                optimizer.zero_grad()
                out = client.model(**{key: value.to(client.device) for key, value in inputs.items()})
                loss = F.cross_entropy(getattr(out, "logits", out), targets.to(client.device))
                loss.backward()
                optimizer.step()
                curve.append({"step": step, "train_loss": float(loss.item()),
                              **_evaluate(client, inputs_eval, targets_eval, teacher)})
            record["branches"][branch] = {
                "curve": curve,
                "heldout_loss_auc_0_to_k": sum((curve[j-1]["loss"] + curve[j]["loss"]) / 2 for j in range(1, steps+1)) / steps,
                "mean_heldout_loss_steps_1_to_k": sum(row["loss"] for row in curve[1:]) / steps,
            }
            del optimizer
        record["status"] = "complete"
    finally:
        _restore(client, state)
        _verify_restored(client, state)
    record["state_restored_exactly"] = True
    with (output_dir / "fork_events.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    return True
