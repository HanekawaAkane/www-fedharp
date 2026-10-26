"""Small PubMedQA task adapter for the FedHARP server.

Each client consumes its fixed training order once. A local minibatch is one
communication round; only final, fixed client test subsets are evaluated.
Only the closed-form projection transport arm is exercised here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

from factor_projection import project_factor_dicts

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
LABELS = ('yes', 'no', 'maybe')
INSTRUCTION = ('Use only the following PubMed abstract to answer the question. '
               'Respond with exactly one word: yes, no, or maybe.')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def append_json(path, obj):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def read_data(path):
    result = []
    for row in pd.read_parquet(path).to_dict('records'):
        ctx = row['context']; parts = ctx.get('contexts', []) if isinstance(ctx, dict) else []
        result.append({'pubid': str(row['pubid']), 'question': str(row['question']),
                       'abstract': ' '.join(str(x) for x in parts),
                       'gold': str(row['final_decision']).strip().lower()})
    assert all(r['gold'] in LABELS for r in result)
    assert len({r['pubid'] for r in result}) == len(result)
    return result


def dirichlet_partition(rows, clients, alpha, seed, min_size=24):
    rng = np.random.RandomState(seed)
    for attempt in range(1000):
        groups = [[] for _ in range(clients)]
        for label in LABELS:
            indices = [i for i, row in enumerate(rows) if row['gold'] == label]
            rng.shuffle(indices)
            proportions = rng.dirichlet([alpha] * clients)
            sizes = rng.multinomial(len(indices), proportions)
            offset = 0
            for cid, n in enumerate(sizes):
                groups[cid].extend(indices[offset:offset + int(n)]); offset += int(n)
        if min(map(len, groups)) >= min_size:
            for cid, g in enumerate(groups):
                np.random.RandomState(seed + 1000 + cid).shuffle(g)
            flat = [i for g in groups for i in g]
            assert sorted(flat) == list(range(len(rows)))
            return groups, attempt + 1
    raise ValueError('Could not obtain a nonempty client partition')


def make_manifest(args, splits):
    train_groups, attempts = dirichlet_partition(splits['train'], args.clients, args.alpha, args.seed + 11)
    # Disjoint test pools follow each client's training label proportions.
    # Every class keeps smoothing so all clients have adequate test coverage.
    rng = np.random.RandomState(args.seed + 29)
    pools = [[] for _ in range(args.clients)]
    hist = [Counter(splits['train'][i]['gold'] for i in g) for g in train_groups]
    for label in LABELS:
        ids = [i for i, r in enumerate(splits['test']) if r['gold'] == label]
        rng.shuffle(ids)
        weights = np.asarray([h[label] + 5 for h in hist], dtype=float)
        sizes = rng.multinomial(len(ids), weights / weights.sum())
        offset = 0
        for cid, n in enumerate(sizes):
            pools[cid].extend(ids[offset:offset + int(n)]); offset += int(n)
    assert min(map(len, pools)) >= args.eval_per_client, [len(p) for p in pools]
    test_groups = []
    for cid, pool in enumerate(pools):
        srng = np.random.RandomState(args.seed + 2000 + cid)
        picked = srng.choice(pool, size=args.eval_per_client, replace=False)
        test_groups.append([int(i) for i in picked])
    rounds = max(math.ceil(len(g) / args.batch_size) for g in train_groups)
    manifest = {'seed': args.seed, 'clients': args.clients, 'alpha': args.alpha,
                'ordinary_epochs': 1, 'batch_size': args.batch_size,
                'round_definition': 'one next minibatch per nonexhausted client, then synchronization',
                'rounds': rounds, 'partition_attempts': attempts,
                'test_sampling': f'{args.eval_per_client} fixed random examples per disjoint client pool; no repeated inference',
                'test_per_client': args.eval_per_client,
                'train_indices': train_groups, 'test_indices': test_groups,
                'test_pool_indices': pools,
                'train_pubids': [[splits['train'][i]['pubid'] for i in g] for g in train_groups],
                'test_pubids': [[splits['test'][i]['pubid'] for i in g] for g in test_groups],
                'client_train_histograms': [dict(h) for h in hist],
                'client_test_histograms': [dict(Counter(splits['test'][i]['gold'] for i in g)) for g in test_groups],
                'data_sha256': {s: file_hash(args.data_dir / f'{s}.parquet') for s in splits}}
    manifest['manifest_sha256'] = digest(manifest)
    return manifest


def template_ids(tokenizer, question, abstract):
    content = f'{INSTRUCTION}\n\nAbstract:\n{abstract}\n\nQuestion: {question}\nAnswer:'
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                       tokenize=True, add_generation_prompt=True,
                                       enable_thinking=False)
    if hasattr(ids, 'input_ids'): ids = ids.input_ids
    elif hasattr(ids, 'get'): ids = ids['input_ids']
    if hasattr(ids, 'tolist'): ids = ids.tolist()
    if ids and isinstance(ids[0], list): ids = ids[0]
    result = [int(i) for i in ids]
    assert result
    return result


def tokenize_rows(tokenizer, rows, max_length, max_candidate_tokens):
    prepared = []
    cap = max_length - max_candidate_tokens - 1
    for row in rows:
        full = template_ids(tokenizer, row['question'], row['abstract'])
        ids = full
        if len(full) > cap:
            # Cut only the abstract; always preserve the question and instruction.
            tokens = tokenizer.encode(row['abstract'], add_special_tokens=False)
            lo, hi = 0, len(tokens)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                abridged = tokenizer.decode(tokens[:mid], skip_special_tokens=True)
                trial = template_ids(tokenizer, row['question'], abridged)
                if len(trial) <= cap: lo = mid
                else: hi = mid - 1
            ids = template_ids(tokenizer, row['question'], tokenizer.decode(tokens[:lo], skip_special_tokens=True))
        assert len(ids) <= cap, 'Question/instruction alone exceeds token budget'
        prepared.append(dict(row, prompt_ids=ids, prompt_tokens=len(ids),
                             original_prompt_tokens=len(full), context_truncated=len(full) > cap,
                             prompt_sha256=digest(ids)))
    return prepared


def batch_tensors(rows, candidates, pad_id, device, end_id=None):
    sequences, targets = [], []
    for row, candidate in rows:
        suffix = list(candidates[candidate]) + ([end_id] if end_id is not None else [])
        sequences.append(row['prompt_ids'] + suffix)
        targets.append([-100] * len(row['prompt_ids']) + suffix)
    length = max(map(len, sequences))
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, (seq, lab) in enumerate(zip(sequences, targets)):
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        labels[i, :len(lab)] = torch.tensor(lab, device=device)
        attention[i, :len(seq)] = 1
    return {'input_ids': ids, 'attention_mask': attention, 'labels': labels}


def get_state(layers, factor):
    return {n: getattr(layer, f'get_{factor}')().detach().cpu().clone() for n, layer in layers.items()}


def set_state(layers, state, factor):
    for n, value in state.items(): getattr(layers[n], f'set_{factor}')(value)


def bytes_of(state):
    return sum(t.numel() * t.element_size() for t in state.values())


def optimize_batch(model, layers, selected, batch, lr):
    model.requires_grad_(False)
    for n, layer in layers.items():
        if n in selected:
            layer.unfreeze_B()
            layer.unfreeze_A()
    params = [p for p in model.parameters() if p.requires_grad]
    assert params
    opt = torch.optim.Adam(params, lr=lr)
    model.train(); opt.zero_grad(set_to_none=True)
    loss = model(**batch, use_cache=False).loss
    assert torch.isfinite(loss), 'Nonfinite training loss'
    loss.backward()
    sensitivity = {n: float(layer.lora_B.grad.detach().float().norm().item())
                   if layer.lora_B.grad is not None else 0.0 for n, layer in layers.items()}
    value = float(loss.item()); opt.step(); opt.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    return value, sensitivity


def metrics(predictions):
    cm = {g: {p: 0 for p in LABELS} for g in LABELS}
    for row in predictions: cm[row['gold']][row['prediction']] += 1
    per_class = {}
    for lab in LABELS:
        tp = cm[lab][lab]; fp = sum(cm[g][lab] for g in LABELS if g != lab)
        fn = sum(cm[lab][p] for p in LABELS if p != lab)
        per_class[lab] = {'f1': 2 * tp / max(1, 2 * tp + fp + fn),
                          'support': sum(cm[lab].values())}
    correct = sum(cm[l][l] for l in LABELS)
    return {'correct': correct, 'total': len(predictions),
            'accuracy': correct / len(predictions),
            'macro_f1': sum(per_class[l]['f1'] for l in LABELS) / 3,
            'confusion': cm, 'per_class': per_class}


def evaluate(model, tokenizer, candidates, samples, eval_indices, layers,
             global_A, personal_B, args, out):
    result = []; t0 = time.monotonic(); model.requires_grad_(False); model.eval()
    with torch.inference_mode(), (out / 'predictions.jsonl').open('w', encoding='utf-8') as sink:
        for cid, indices in enumerate(eval_indices):
            if layers is not None:
                set_state(layers, global_A, 'A'); set_state(layers, personal_B[cid], 'B')
            for offset in range(0, len(indices), args.eval_batch_size):
                chosen = indices[offset:offset + args.eval_batch_size]
                pairs = [(samples[i], lab) for i in chosen for lab in LABELS]
                batch = batch_tensors(pairs, candidates, tokenizer.pad_token_id, 'cuda')
                output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False)
                scores = []
                # Slice candidate positions before float/log_softmax to avoid a
                # batch x context x vocabulary fp32 allocation.
                for j, (row, lab) in enumerate(pairs):
                    p = len(row['prompt_ids']); suffix = candidates[lab]
                    logits = output.logits[j, p - 1:p - 1 + len(suffix), :].float()
                    tokens = torch.tensor(suffix, dtype=torch.long, device=logits.device)
                    scores.append(float(logits.log_softmax(-1).gather(1, tokens[:, None]).sum().item()))
                del output, batch
                for j, index in enumerate(chosen):
                    row = samples[index]; values = scores[3 * j:3 * j + 3]
                    pred = LABELS[int(np.argmax(values))]
                    rec = {'client': cid, 'pubid': row['pubid'], 'gold': row['gold'],
                           'prediction': pred, 'candidate_log_likelihood': dict(zip(LABELS, values)),
                           'prompt_sha256': row['prompt_sha256'], 'prompt_tokens': row['prompt_tokens'],
                           'context_truncated': row['context_truncated']}
                    sink.write(json.dumps(rec) + '\n'); result.append(rec)
            logging.info('eval client=%s n=%s elapsed=%.2fs', cid, len(indices), time.monotonic() - t0)
    ids = [r['pubid'] for r in result]
    assert len(set(ids)) == len(ids) == args.clients * args.eval_per_client
    summary = metrics(result)
    summary['clients'] = {str(c): metrics([r for r in result if r['client'] == c]) for c in range(args.clients)}
    summary['eval_seconds'] = time.monotonic() - t0
    summary['context_truncated_test'] = sum(r['context_truncated'] for r in result)
    summary['prediction_sha256'] = file_hash(out / 'predictions.jsonl')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='gemma-2b-it')
    ap.add_argument('--data-dir', type=Path, default=Path('data/pubmedqa/splits'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--clients', type=int, default=4)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--eval-per-client', type=int, default=50)
    ap.add_argument('--eval-batch-size', type=int, default=4)
    ap.add_argument('--max-length', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--projection-q', type=int, default=2)
    ap.add_argument('--projection-mode', choices=['none', 'projection_only'], default='projection_only')
    ap.add_argument('--projection-ridge', type=float, default=1e-6)
    ap.add_argument('--smoke-rounds', type=int, default=0)
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--save-final-adapters', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'DONE').exists(): raise SystemExit('Preserving completed campaign')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    splits = {s: read_data(args.data_dir / f'{s}.parquet') for s in ['train', 'valid', 'test']}
    ids = {s: {r['pubid'] for r in rows} for s, rows in splits.items()}
    assert not (ids['train'] & ids['test'] or ids['valid'] & ids['test'] or ids['train'] & ids['valid'])
    manifest = make_manifest(args, splits)
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['manifest_sha256', 'rounds', 'client_train_histograms', 'client_test_histograms']}), flush=True)
    if args.plan_only: return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(ROOT))
    from models import _extract_peft_lora_layers
    from server import FedHarpServer
    from peft import LoraConfig, get_peft_model, TaskType
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    candidates = {lab: list(tokenizer.encode(lab, add_special_tokens=False)) for lab in LABELS}
    # Training and evaluation share EXACT token IDs, template, context budget,
    # candidate boundary, and a deterministic fixed label order.
    prepared = {s: tokenize_rows(tokenizer, rows, args.max_length, max(map(len, candidates.values()))) for s, rows in splits.items()}
    prompt_manifest = {s: [{'pubid': r['pubid'], 'prompt_sha256': r['prompt_sha256'],
                            'prompt_tokens': r['prompt_tokens'], 'context_truncated': r['context_truncated']}
                           for r in rows] for s, rows in prepared.items()}
    write_json(args.output / 'prompt_manifest.json', prompt_manifest)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                local_files_only=True).to('cuda')
    model.config.use_cache = False
    # Only ordinary Linear modules in the language tower; exclude vision/audio
    # and Gemma4ClippableLinear, which PEFT does not support.
    targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
               and n.endswith(('.q_proj', '.v_proj')) and 'language_model' in n]
    if not targets:
        raise ValueError('No supported language-tower q/v Linear targets; inspect module names')
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                           lora_dropout=0.0, target_modules=targets, task_type=TaskType.CAUSAL_LM,
                           bias='none'))
    layers = _extract_peft_lora_layers(model)
    assert len(layers) == len(targets)
    initial_A, initial_B = get_state(layers, 'A'), get_state(layers, 'B')
    assert all(torch.count_nonzero(b) == 0 for b in initial_B.values())
    end_id = tokenizer.convert_tokens_to_ids('<turn|>')
    assert end_id is not None and end_id != tokenizer.unk_token_id
    source_hashes = {f: file_hash(ROOT / f) for f in ['main.py', 'client.py', 'server.py', 'models.py']}
    protocol = {'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'model_path': args.model_path, 'manifest_sha256': manifest['manifest_sha256'],
                'source_hashes': source_hashes, 'runner_sha256': file_hash(__file__),
                'candidate_token_ids': candidates, 'scoring': 'sum conditional label-token log probabilities; no length normalization',
                'chat_format': 'native template; enable_thinking=False', 'end_token': end_id,
                'targets': targets, 'module_count': len(layers),
                'projection_trigger': 'pre-round inactivity counter >= q on selected reactivated blocks',
                'projection_solver': 'B_new = B_old A_old A_new^T (A_new A_new^T + ridge I)^-1',
                'projection_optimizer': 'closed-form local transport; zero extra optimizer steps',
                'ordinary_optimizer': 'fresh Adam per client minibatch, no gradient accumulation',
                'fedharp_budget': 'All-Low 50 percent transformer blocks, fixed rank8',
                'aggregation': 'normalized n_k**0.5; active upload damping=1',
                'evaluation': 'one final pass: shared final A + each client private B; no validation selection',
                'test_sample_sha256': digest(manifest['test_pubids']),
                'full_model_checkpoints': 0}
    write_json(args.output / 'protocol.json', protocol)
    logging.info('loaded modules=%s candidates=%s prompt_truncation=%s time=%.2fs',
                 len(layers), candidates, {s: sum(r['context_truncated'] for r in rows) for s, rows in prepared.items()}, time.monotonic() - started)
    all_results = []; bundle = {'protocol': protocol, 'manifest': manifest, 'methods': {}}
    arms = ['fedharp_projection']
    if not args.smoke_rounds:
        out = args.output / 'zero_shot'; out.mkdir(exist_ok=True)
        with model.disable_adapter():
            summary = evaluate(model, tokenizer, candidates, prepared['test'], manifest['test_indices'],
                               None, None, None, args, out)
        summary.update(method='zero_shot', seed=args.seed, manifest_sha256=manifest['manifest_sha256'])
        write_json(out / 'summary.json', summary); (out / 'DONE').write_text('final fixed-subset evaluation completed\n')
        all_results.append(summary); logging.info('zero_shot accuracy=%.4f macro_f1=%.4f', summary['accuracy'], summary['macro_f1'])
    for arm in arms:
        t0 = time.monotonic(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        set_state(layers, initial_A, 'A'); set_state(layers, initial_B, 'B')
        out = args.output / arm; out.mkdir(exist_ok=True)
        for name in ['training.jsonl', 'projection_events.jsonl']:
            if (out / name).exists(): raise ValueError(f'Preserving existing partial output: {out / name}')
        server = FedHarpServer(model=model, lora_layers=layers, num_clients=args.clients,
                               allocation_ratio=0.5, aggregation_lr=1.0, struct_to_data_rounds=8,
                               seed=args.seed, method='fedharp')
        for cid in range(args.clients): server.client_layer_counts[cid] = int(0.5 * len(server.layer_names))
        private_A = {cid: {n: a.clone() for n, a in initial_A.items()} for cid in range(args.clients)}
        private_B = {cid: {n: b.clone() for n, b in initial_B.items()} for cid in range(args.clients)}
        seen = {cid: [] for cid in range(args.clients)}; counts = {cid: len(g) for cid, g in enumerate(manifest['train_indices'])}
        totals = {'ordinary_optimizer_steps': 0, 'projection_events': 0,
                  'projection_module_targets': 0, 'upload_tensor_bytes': 0,
                  'sensitivity_scalar_bytes': 0, 'download_tensor_bytes': 0}
        final_round = min(manifest['rounds'], args.smoke_rounds) if args.smoke_rounds else manifest['rounds']
        for round_num in range(1, final_round + 1):
            pre_round_staleness = {cid: dict(values) for cid, values in server.staleness.items()}
            allocation = server.start_round()
            # Restore the pre-round snapshot so projection triggers and
            # aggregation damping see the same tau. Counters advance after
            # local training and aggregation.
            server.staleness = {cid: dict(values) for cid, values in pre_round_staleness.items()}
            global_A = get_state(layers, 'A')
            updates = {}; active_map = {}; records = []
            for cid, order in enumerate(manifest['train_indices']):
                indices = order[(round_num - 1) * args.batch_size:round_num * args.batch_size]
                if not indices: continue
                selected = set(allocation[cid]); active_map[cid] = selected
                old_A = {n: a.clone() for n, a in private_A[cid].items()}
                old_B = {n: b.clone() for n, b in private_B[cid].items()}
                set_state(layers, global_A, 'A'); set_state(layers, private_B[cid], 'B')
                batch = batch_tensors([(prepared['train'][i], prepared['train'][i]['gold']) for i in indices],
                                       candidates, tokenizer.pad_token_id, 'cuda', end_id=end_id)
                targets_projection = {n for n in selected if int(pre_round_staleness[cid].get(n, 0)) >= int(args.projection_q)}
                if targets_projection:
                    projected_B, projection_stats = project_factor_dicts(
                        old_A, old_B, get_state(layers, 'A'), get_state(layers, 'B'),
                        targets_projection, ridge=float(args.projection_ridge))
                    set_state(layers, projected_B, 'B')
                    totals['projection_events'] += 1
                    totals['projection_module_targets'] += len(projection_stats)
                    append_json(out / 'projection_events.jsonl', {
                        'round': round_num, 'client': cid,
                        'pre_round_targets': sorted(targets_projection), 'q': int(args.projection_q),
                        'ridge': float(args.projection_ridge),
                        'residual_before': float(sum(v['residual_before'] for v in projection_stats.values())),
                        'residual_after': float(sum(v['residual_after'] for v in projection_stats.values())),
                        'layers': projection_stats,
                    })
                loss, sense = optimize_batch(model, layers, selected, batch, args.lr)
                after_A = get_state(layers, 'A')
                delta = {n: (after_A[n] - global_A[n]).to(layers[n].get_A().device) for n in selected}
                updates[cid] = delta; private_A[cid] = after_A; private_B[cid] = get_state(layers, 'B')
                server.update_client_sensitivities(cid, sense)
                seen[cid].extend(indices)
                up, down = bytes_of(delta), bytes_of(global_A)
                scalars = 4 * len(selected)
                totals['ordinary_optimizer_steps'] += 1; totals['upload_tensor_bytes'] += up
                totals['sensitivity_scalar_bytes'] += scalars; totals['download_tensor_bytes'] += down
                records.append({'client': cid, 'n': len(indices), 'loss': loss,
                                'pubids': [prepared['train'][i]['pubid'] for i in indices],
                                'allocated_modules': sorted(selected),
                                'upload_tensor_bytes': up, 'download_tensor_bytes': down,
                                'sensitivity_scalar_bytes': scalars})
            # Client simulation shares a physical model. Restore the server
            # snapshot BEFORE averaging, so the last client update is not added
            # twice. This implements production independent client/server state.
            set_state(layers, global_A, 'A')
            server.staleness = {cid: dict(values) for cid, values in pre_round_staleness.items()}
            server.aggregate_matrices(client_updates_A=updates, client_updates_B=None,
                                       client_sample_counts=counts, allocation_map=active_map)
            server.update_staleness(allocation)
            torch.cuda.synchronize()
            append_json(out / 'training.jsonl', {'round': round_num, 'clients': records,
                        'elapsed_seconds': time.monotonic() - t0, 'totals': dict(totals)})
            if round_num == 1 or round_num % 5 == 0 or round_num == final_round:
                logging.info('%s round=%s/%s samples=%s projection_events=%s elapsed=%.2fs', arm, round_num,
                             final_round, sum(map(len, seen.values())), totals['projection_events'], time.monotonic() - t0)
        final_A = get_state(layers, 'A')
        if args.smoke_rounds:
            assert totals['projection_events'] > 0
            summary = evaluate(model, tokenizer, candidates, prepared['test'], manifest['test_indices'],
                               layers, final_A, private_B, args, out)
            write_json(out / 'smoke_summary.json', dict(method=arm, rounds=final_round, **totals, **summary))
            continue
        for cid, order in enumerate(manifest['train_indices']): assert seen[cid] == order
        assert sum(map(len, seen.values())) == len(splits['train']) == 450
        assert totals['projection_events'] > 0
        train_seconds = time.monotonic() - t0
        summary = evaluate(model, tokenizer, candidates, prepared['test'], manifest['test_indices'],
                           layers, final_A, private_B, args, out)
        summary.update(method=arm, seed=args.seed, rounds=final_round, ordinary_epochs=1,
                       ordinary_training_samples=450, train_seconds=train_seconds,
                       manifest_sha256=manifest['manifest_sha256'], **totals)
        write_json(out / 'summary.json', summary)
        (out / 'DONE').write_text('one ordinary epoch; final fixed-subset evaluation completed\n')
        all_results.append(summary)
        if args.save_final_adapters:
            bundle['methods'][arm] = {'global_A': final_A, 'client_B': private_B, 'summary': summary}
            temp = args.output / 'final_adapters.tmp'; torch.save(bundle, temp)
            temp.replace(args.output / 'final_adapters.pt')
        logging.info('%s DONE accuracy=%.4f macro_f1=%.4f train_seconds=%.2f', arm, summary['accuracy'], summary['macro_f1'], train_seconds)
    if args.smoke_rounds:
        (args.output / 'SMOKE_DONE').write_text('projection arm: synchronization, transport, and state invariants passed\n')
    else:
        write_json(args.output / 'comparison.json', {'results': all_results,
                   'elapsed_seconds': time.monotonic() - started,
                   'manifest_sha256': manifest['manifest_sha256'], 'test_sample_sha256': protocol['test_sample_sha256'],
                   'single_seed': True, 'full_model_checkpoints': 0,
                   'final_adapter_bundle_bytes': (args.output / 'final_adapters.pt').stat().st_size if args.save_final_adapters else 0})
        (args.output / 'DONE').write_text('all final predictions and one-epoch invariants verified\n')


if __name__ == '__main__': main()
