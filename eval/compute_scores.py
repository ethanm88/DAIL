import os
import re
import json
import math
import argparse
import random
from typing import Dict, List, Optional, Tuple, DefaultDict, Any



from math import comb
import yaml


def pass_at_k(all_results, k: int) -> float:
    total = len(all_results)
    if total == 0:
        return 0.0
    pass_sum = 0.0
    for ex in all_results:
        samples = ex["samples"]
        n = len(samples)
        c = sum(1 for s in samples if s.get("correct", False))
        if c == 0:
            p = 0.0
        elif n - c < k:
            p = 1.0
        else:
            p_all_wrong = comb(n - c, k) / comb(n, k)
            p = 1 - p_all_wrong
        pass_sum += p
    return pass_sum / total


def _extract_answer_text(sample_dict):
    answer = str(sample_dict.get("full_output", ""))
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[-1]
    return answer.strip()


def _collect_all_answer_texts(all_results):
    texts = []
    for ex in all_results:
        for s in ex["samples"]:
            t = _extract_answer_text(s)
            if not t or t.strip() == "None": continue
            texts.append(t)
    return texts


def compute_for_model(raw_dir, model_id_plain, dataset_names, k_value, tokenizer):
    k_values = [2**i for i in range(0, int(math.log2(k_value)) + 1)]
    all_results = []
    num_files_total = 0
    
    model_dirs_to_load = []
    all_datasets_present = True
    for dataset_name in dataset_names:
        model_id = f"dataset={dataset_name}_{model_id_plain}"
        model_dir = os.path.join(raw_dir, model_id)
        print(f"Checking for model directory: {model_dir}")
        if not os.path.isdir(model_dir):
            all_datasets_present = False
            break
        model_dirs_to_load.append((model_dir, dataset_name))
    if not all_datasets_present:
        return {}

    for model_dir, dataset_name in model_dirs_to_load:
        num_files_dataset = 0
        print(model_dir)
        for fname in os.listdir(model_dir):
            if not fname.startswith("problem_") or not fname.endswith(".json"): continue
            cur_pid = (int)(fname.split("problem_")[-1].split(".json")[0])
            try:
                with open(os.path.join(model_dir, fname)) as f:
                    data = json.load(f)
            except:
                print(f'Failed to load {os.path.join(model_dir, fname)}')
                continue
            if "samples" in data: all_results.append({"samples": data["samples"]})
            num_files_dataset += 1
        num_files_total += num_files_dataset
        print(f"  > Loaded {num_files_dataset} problems from {model_dir}")
    if not all_results: return {}

    max_n = max(len(item["samples"]) for item in all_results)
    valid_ks = [k for k in k_values if k <= max_n]

    results = {}
    for k in valid_ks:
        results[f"pass@{k}"] = pass_at_k(all_results, k)
        
    results[f"num_problems"] = num_files_total

    return results


def _ensure_dir_for_file(path):
    if os.path.dirname(path): os.makedirs(os.path.dirname(path), exist_ok=True)


# Plotting removed per request — plotting functions were deleted.


def _load_models_yaml(path: str) -> List[dict]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"models_file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    out: List[dict] = []
    # 1) exact models
    for m in (data.get("models") or []):
        mid = m.get("id")
        if mid:
            out.append({
                "id": str(mid),
                "label": str(m.get("label", mid)),
                "enabled": bool(m.get("enabled", True)),
            })
    # 2) expand templates
    for t in (data.get("templates") or []):
        if not t.get("enabled", True): continue
        id_tmpl = t.get("id_template")
        lab_tmpl = t.get("label_template", "{id}")
        params = t.get("for") or {}
        keys = list(params.keys())
        if not id_tmpl or not keys: continue
        from itertools import product
        values_lists = [params[k] for k in keys]
        for combo in product(*values_lists):
            subs = dict(zip(keys, combo))
            mid = id_tmpl.format(**subs)
            label = lab_tmpl.format(**subs) if lab_tmpl else mid
            out.append({"id": mid, "label": label, "enabled": True})
        if "initial" in t:
            initial_model = t["initial"]
            out.append({"id": initial_model, "label": lab_tmpl.format(**{"epoch": 0}), "enabled": True})
    return out

def _prettify_id(model_id: str) -> str:
    s = model_id.replace("model=", "").replace("_", " ")
    s = re.sub(r"\s+enabling thinking=(True|False)", r" (thinking=\1)", s, flags=re.I)
    s = re.sub(r"\s+epochs[_=\s](\d+)", r" (epochs=\1)", s)
    return s.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', default='raw_eval_results')
    parser.add_argument('--models_file', default="score_configs/all.yaml")
    parser.add_argument('--max_k', type=int, default=128)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--dataset', nargs='+', default=['aime_2024', 'aime_2025'])
    parser.add_argument('--dataset_label', default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.raw_dir): return

    dataset_label = args.dataset_label if args.dataset_label else "+".join(sorted(args.dataset))
    yaml_models = _load_models_yaml(args.models_file)
    selected = [m for m in yaml_models if m.get("enabled", True)]
    if not selected: return
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=True)
    summary = {}
    for m in selected:
        label_plain = m.get("label") or _prettify_id(m["id"])
        print(f"Computing for: {label_plain}")
        res = compute_for_model(args.raw_dir, m["id"], args.dataset, args.max_k, None)
        if res: summary[label_plain] = res

    output_file = os.path.join(args.results_dir, f"{dataset_label}_summary_v2.json")
    _ensure_dir_for_file(output_file)
    with open(output_file, 'w') as out: json.dump(summary, out, indent=2)
    print(f"Saved summary to {output_file}")

if __name__ == '__main__':
    main()