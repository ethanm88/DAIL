import os
import json
import time
import math
import torch
from math import comb
import random
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations, islice
from vllm import LLM, SamplingParams

def pass_at_k(all_results, k: int) -> float:
    """
    Compute empirical pass@k over a list of examples.
    """
    total = len(all_results)
    if total == 0:
        return 0.0

    pass_sum = 0.0
    for ex in all_results:
        samples = ex["samples"]
        n = len(samples)
        c = sum(1 for s in samples if s["correct"])
        if c == 0:
            p = 0.0
        elif n - c < k:
            p = 1.0
        else:
            p_all_wrong = comb(n - c, k) / comb(n, k)
            p = 1 - p_all_wrong
        pass_sum += p

    return pass_sum / total

def init_llm_with_retry(**kwargs):
    time.sleep(random.uniform(0, 5))
    return LLM(**kwargs)

def run_eval(
    parquet_dir: str,
    model_name: str = "agentica-org/DeepScaleR-1.5B-Preview",
    eval_dir: str = "eval_results",
    num_samples: int = 32,
    batch_size: int = 32,
    enable_thinking: bool = False,
    problems: list = None,
    greedy: bool = False,
    max_tokens: int = 8000,
    max_model_len: int = 8000,
    temperature: float = 0.6,
    force_answer: bool = False
):
    os.makedirs(eval_dir, exist_ok=True)
    print('EVALUATION MODEL:', model_name)
    # Check if all problems already have results
    if problems is not None:
        existing_problem_files = {int(fname.split('_')[1].split('.')[0]) for fname in os.listdir(eval_dir) if fname.startswith("problem_") and fname.endswith(".json")}
        problems_to_run = [p for p in problems if p not in existing_problem_files]
        if not problems_to_run:
            print("All specified problems already have results. Skipping evaluation.")
            return
        else:
            print(f"Evaluating {len(problems_to_run)} out of {len(problems)} specified problems.")
            problems = problems_to_run
    
    print('Problems', problems)
    
    from datasets import load_dataset
    from more_itertools import chunked
    current_directory = os.getcwd()   
    from utils.math_eval import compute_score
    from tqdm.auto import tqdm
    # from datasets import Dataset, DatasetDict
    from utils.chat_template import UPDATED_CHAT_TEMPLATE
    
    # Load dataset
    ds = load_dataset("parquet", data_files=os.path.join(parquet_dir, "test.parquet"), features= None)

    test_ds = ds["test"] if "test" in ds else ds["train"]

    # Filter items by problem index (from extra_info)
    items = []
    for ex in test_ds:
        idx = (int)(ex.get("extra_info", {}).get("index"))
        if idx is None:
            continue
        if problems is None or idx in problems:
            items.append((idx, ex["prompt"], ex["reward_model"]["ground_truth"]))

    if not items:
        print("No matching problems to evaluate.")
        return

    # Initialize LLM
    if "72B" in model_name:
        tensor_parallel_size = 4
    else:
        tensor_parallel_size = 1

    llm = init_llm_with_retry(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.7,
        max_model_len=max_model_len
    )


    # Iterate in batches
    for batch_idx, batch in enumerate(tqdm(chunked(items, batch_size), desc="Batches", unit="batch")):
        pids, questions, ground_truths = zip(*batch)
        if greedy:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
            )
        else:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                n=num_samples,
                stop=["</think>"],
                presence_penalty=1.5, # add to avoid repetition
            )

        if 'Qwen3' in model_name:
            chat_template_kwargs = {'enable_thinking': enable_thinking}
            batch_responses = llm.chat(list(questions), sampling_params, chat_template_kwargs=chat_template_kwargs)
        else:
            batch_responses = llm.chat(list(questions), sampling_params)

        all_outputs_per_example = []
        for i in range(len(pids)):
            outputs_texts = [resp.text for resp in batch_responses[i].outputs]
            all_outputs_per_example.append(outputs_texts)

        # Identify samples missing </think> and prepare for a single batched repair call
        to_fix_prompts = []
        to_fix_refs = []

        if enable_thinking:
            for ex_i, outputs in enumerate(all_outputs_per_example):
                for sample_j, text in enumerate(outputs):
                    safe_text = text or ""
                    if '</think>' not in safe_text:
                        # Use continue_final_message logic by appending the closing tag to assistant content
                        user_message = questions[ex_i][0] 
                        if force_answer:
                            full_output_for_answer = [
                                user_message,
                                {"role": "assistant", "content": safe_text + "\n</think> \n The final answer (in \\boxed{}):"}
                            ]
                        else:
                            full_output_for_answer = [
                                user_message,
                                {"role": "assistant", "content": safe_text + "\n</think>"}
                            ]
                        to_fix_prompts.append(full_output_for_answer)
                        to_fix_refs.append((ex_i, sample_j))
                    elif force_answer:
                        # if forcing answer, need to generate the final answer even if final tokens started generating
                        full_output_for_answer = [
                            user_message,
                            {"role": "assistant", "content": safe_text + "\n The final answer (in \\boxed{}):"}
                        ]
                        to_fix_prompts.append(full_output_for_answer)
                        to_fix_refs.append((ex_i, sample_j))
        else:
            if force_answer:
                for ex_i, outputs in enumerate(all_outputs_per_example):
                    for sample_j, text in enumerate(outputs):
                        safe_text = text or ""
                        # check if \\boxed{} is missing
                        if '\\boxed{' not in safe_text:
                            full_output_for_answer = [
                                questions[ex_i][0],
                                {"role": "assistant", "content": safe_text + "\n The final answer (in \\boxed{}):"}
                            ]
                            to_fix_prompts.append(full_output_for_answer)
                            to_fix_refs.append((ex_i, sample_j))
        if to_fix_prompts:
            if force_answer:
                second_pass_params = SamplingParams(
                    max_tokens=100,
                    temperature=temperature,
                    presence_penalty=1.5,
                    top_p=0.95,
                    n=1,
                )
            else:
                second_pass_params = SamplingParams(
                    max_tokens=2048,
                    temperature=temperature,
                    presence_penalty=1.5,
                    top_p=0.95,
                    n=1,
                )

            # Perform one batched call for all missing answers across the entire batch
            fixed_responses = llm.chat(
                to_fix_prompts,
                second_pass_params,
                continue_final_message=True,
                add_generation_prompt=False,
                chat_template=UPDATED_CHAT_TEMPLATE,
                chat_template_kwargs={'enable_thinking': False}
            )

            # Map batched results back to their original example and sample indices
            for k, resp in enumerate(fixed_responses):
                ex_i, sample_j = to_fix_refs[k]
                if resp.outputs and resp.outputs[0].text:
                    fixed_text = resp.outputs[0].text
                    all_outputs_per_example[ex_i][sample_j] = (all_outputs_per_example[ex_i][sample_j] or "") + "\n</think>\n" + fixed_text
        # Final scoring and serialization
        for i, pid in enumerate(pids):
            samples = []
            cur_ground_truth = str(ground_truths[i]) if ground_truths[i] is not None else None

            for full_output in all_outputs_per_example[i]:
                is_correct = False
                extracted_answer = None

                if cur_ground_truth is not None and full_output is not None:
                    is_correct, extracted_answer = compute_score(full_output, cur_ground_truth, return_extracted=True)

                samples.append({
                    "full_output": full_output.split('</think>')[-1].strip() if '</think>' in (full_output or "") else (full_output or "").strip(),
                    "ground_truth": cur_ground_truth,
                    "pred_answer": str(extracted_answer),
                    "correct": is_correct,
                })

            question_text = questions[i][0]['content'] if isinstance(questions[i], list) and questions[i] and 'content' in questions[i][0] else str(questions[i])
            result = {"problem_index": pid, "question": question_text, "samples": samples}
            out_path = os.path.join(eval_dir, f"problem_{pid}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=4)
            print(f"Saved results for problem {pid} to {out_path}")

    # Optionally compute aggregated pass@k over all saved
    all_results = []
    for fname in os.listdir(eval_dir):
        if fname.startswith("problem_") and fname.endswith(".json"):
            with open(os.path.join(eval_dir, fname)) as f:
                all_results.append(json.load(f))

    k_values = [2**i for i in range(int(math.log2(num_samples)) + 1)]
    pass_at_k_results = {k: pass_at_k(all_results, k) for k in k_values}
    print("Pass@k results:", pass_at_k_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sft_human_distill_Qwen_Qwen3-4B_16K/model_merged")
    parser.add_argument("--dataset", type=str, default="aime_2025")
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--problems", type=int, nargs='+', default=None)
    parser.add_argument("--eval_dir", type=str, default="raw_eval_results")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--use_temperature", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--max_model_len", type=int, default=8000)
    parser.add_argument("--no_force_answer", action="store_true")
    args = parser.parse_args()


    if args.use_temperature:
        temperature_flag = f'_temperature={args.temperature}'
    else:
        temperature_flag = ''



    if not args.no_force_answer:
        full_eval_dir = os.path.join(args.eval_dir, f"dataset={args.dataset}_{args.max_tokens}_model={args.model_name.replace('/', '_')}_enabling_thinking={args.enable_thinking}_greedy={args.greedy}{temperature_flag}")
    else:
        full_eval_dir = os.path.join(args.eval_dir, f"dataset={args.dataset}_{args.max_tokens}_model={args.model_name.replace('/', '_')}_enabling_thinking={args.enable_thinking}_greedy={args.greedy}{temperature_flag}_no_force_answer")


    os.makedirs(full_eval_dir, exist_ok=True)

    parquet_dir = os.path.join(os.environ.get("DATA_DIR", ""), args.dataset)
    print(f"Using parquet directory from DATA_DIR: {parquet_dir}")
    
    # print save dir
    print(f"Saving evaluation results to: {full_eval_dir}")
    run_eval(
        parquet_dir=parquet_dir,
        model_name=args.model_name,
        eval_dir=full_eval_dir,
        num_samples=args.num_samples,
        enable_thinking=args.enable_thinking,
        problems=args.problems,
        greedy=args.greedy,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        force_answer=not args.no_force_answer
    )

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()