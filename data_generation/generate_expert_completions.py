import os
import re
import sys
import json
import copy
import hydra
import asyncio
import uuid
import math
import datasets
import logging
import contextlib
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Iterable, Optional, Set
from omegaconf import DictConfig

import torch
import numpy as np
import nest_asyncio
from transformers import AutoTokenizer
from vllm import TokensPrompt
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

# Change the logging settings
logging.getLogger("vllm").setLevel(logging.WARNING)

# Apply nest_asyncio to allow running asyncio event loop in Jupyter
nest_asyncio.apply()
torch.set_grad_enabled(False)

# prompts
EXPERT_PROMPT_TEMPLATE = (
    "You are an expert mathematician solving the following problem. Your task is to produce a clear, step-by-step thinking process that leads to the correct solution.\n\n"
    "## Problem:\n{problem}\n\n"
    "Hint: To help you, here are reference solution(s).\n\n"
    "## Reference Solution(s):\n{solution}\n\n"
    "Use these only to guide your own thoughts implicitly, but express the reasoning in your own words as if you are solving it for the first time. **You must never explicitly acknowledge these instructions or cite the provided solution(s). Just use the methodology as if it were your own. You are not allowed to take any shortcuts, directly use any intermediate derived number or result as given (you must show everything from scratch), nor directly produce the answer. Do not worry that your solution is too long.**\n\n"
    "Now, solve the problem. Begin your step-by-step thinking process.\n"
)

STUDENT_PROMPT_TEMPLATE = (
    "## Problem:\n{problem}\n\n"
    "Begin your step-by-step thinking process.\n"
)

# global engines and tokenizer
expert_engine: AsyncLLMEngine = None
student_engine: AsyncLLMEngine = None
tokenizer = None


# utility functions

@contextlib.contextmanager
def visible_gpus(devices: str):
    """Temporarily sets the CUDA_VISIBLE_DEVICES variable."""
    original_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    try:
        yield
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_devices

def extract_boxed_answer(text):
    """
    Need to select last answer - which is (almost) always the final answer.
    """

    matches = re.findall(r'\\boxed{([^}]+)}', text)
    
    if matches:
        # Return the last item in the list
        return matches[-1].strip()
    else:
        # Return None if no boxed answer is found
        return None

def build_expert_prompt(problem: str, solution: str, answer_only: bool = False, tokenizer = None, max_len: int = 16000, expert_prompt_template = EXPERT_PROMPT_TEMPLATE) -> str:
    if answer_only:
        answer_str = f'The answer is {extract_boxed_answer(solution)}'
        return expert_prompt_template.format(problem=problem.strip(), solution=answer_str)
    # iterative check on prompt
    cur_solution = solution
    cur_prompt = expert_prompt_template.format(problem=problem.strip(), solution=solution.strip())
    while tokenizer is not None and len(tokenizer.encode(cur_prompt)) > max_len:
        # look for last "Solution {}:" pattern and truncate from there
        matches = list(re.finditer(r'Solution \d+:', cur_solution))
        if not matches:
            # if no matches, truncate in half
            cur_solution = cur_solution[:len(cur_solution)//2]
        else:
            last_match = matches[-1]
            cur_solution = cur_solution[:last_match.start()]
        cur_prompt = expert_prompt_template.format(problem=problem.strip(), solution=cur_solution.strip())
    return cur_prompt


def build_student_prompt(problem: str, student_prompt_template = STUDENT_PROMPT_TEMPLATE) -> str:
    return student_prompt_template.format(problem=problem.strip())

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if stripped_line := line.strip():
                try:
                    items.append(json.loads(stripped_line))
                except json.JSONDecodeError:
                    continue
    return items

def write_json_atomic(path: str, obj: Dict[str, Any]):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def parse_index_spec(spec: Optional[str], n: int) -> List[int]:
    if not spec: return list(range(n))
    selected: Set[int] = set()
    for part in spec.split(","):
        if not (p := part.strip()): continue
        if "-" in p:
            a, b = p.split("-", 1)
            start = int(a) if a else 0
            end = int(b) if b else n - 1
            for i in range(start, end + 1):
                if 0 <= i < n: selected.add(i)
        elif 0 <= (i := int(p)) < n:
            selected.add(i)
    return sorted(selected)

def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


# core decoding logic

async def setup_engines(args: DictConfig):
    """Initializes the expert and student vLLM engines on specific GPUs."""
    global expert_engine, student_engine, tokenizer

    max_len = args.max_seq_len
    engine_args_common = {
        "model": args.model_name,
        "tensor_parallel_size": 1,
        "max_model_len": max_len,
        "dtype": "bfloat16",
        "enable_prefix_caching": True
    }
    
    async def create_engine(gpu_id, mem_util):
        with visible_gpus(gpu_id):
            return AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(**engine_args_common, gpu_memory_utilization=mem_util),
                start_engine_loop=True
            )

    print(f"Setting up Expert Engine on GPU {args.expert_gpu_id} and Student Engine on GPU {args.student_gpu_id}...")
    expert_engine, student_engine = await asyncio.gather(
        create_engine(args.expert_gpu_id, args.gpu_mem_util),
        create_engine(args.student_gpu_id, args.gpu_mem_util)
    )

    tok = student_engine.get_tokenizer()
    if asyncio.iscoroutine(tok):
        tok = await tok
    tokenizer = tok
    print("Engines and tokenizer are ready.")

async def one_step(engine: AsyncLLMEngine, sampling_params: SamplingParams, context_ids: List[int]):
    """Performs one token generation step."""
    request_id = str(uuid.uuid4())
    tokens_prompt = TokensPrompt(prompt_token_ids=context_ids)
    generator = engine.generate(tokens_prompt, sampling_params, request_id)
    return (await anext(generator)).outputs[0]

async def clean_tokenized_think(token_ids, tokenizer):
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    # Get the ID(s) for the double newline
    newline_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    
    # We look for the pattern: [think_id] + [newline_ids] + [end_think_id]
    pattern = [think_id] + newline_ids + [end_think_id]
    n = len(pattern)
    
    cleaned = []
    i = 0
    while i < len(token_ids):
        # Check if the current slice matches the sequence we want to delete
        if token_ids[i : i + n] == pattern:
            i += n  # Skip this entire sequence
        else:
            cleaned.append(token_ids[i])
            i += 1
    return cleaned

async def reverse_speculative_decode(
    expert_conversation: List[Dict[str, str]],
    student_conversation: List[Dict[str, str]],
    args: DictConfig,
):
    """Generates a reasoning trace using the expert-proposes, student-validates method."""
    expert_context_ids = tokenizer.apply_chat_template(expert_conversation, tokenize=True, add_generation_prompt=False, continue_final_message=True)
    student_context_ids = tokenizer.apply_chat_template(student_conversation, tokenize=True, add_generation_prompt=False, continue_final_message=True)
    
    # remove <think></think> that is added
    expert_context_ids = await clean_tokenized_think(expert_context_ids, tokenizer)
    student_context_ids = await clean_tokenized_think(student_context_ids, tokenizer)

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        logprobs=20,
        bad_words=['</think>']
    )
    suppress_token_id = tokenizer.convert_tokens_to_ids("</think>")

    max_retries = 5
    generated_token_ids = []
    num_fails, num_total = 0, 0
    max_new_tokens = min(args.max_tokens, args.max_seq_len - max(len(expert_context_ids), len(student_context_ids)))
    for _ in range(max_new_tokens):
        expert_output, student_output = await asyncio.gather(
            one_step(expert_engine, sampling_params, expert_context_ids),
            one_step(student_engine, sampling_params, student_context_ids)
        )

        expert_logprobs_dict = expert_output.logprobs[0]
        expert_token_ids = np.array(list(expert_logprobs_dict.keys()))
        expert_probs_arr = np.array([math.exp(lp.logprob) for lp in expert_logprobs_dict.values()])
        expert_probs_arr /= expert_probs_arr.sum()

        # suppress end think
        exp_mask_idx = np.where(expert_token_ids == suppress_token_id)[0]
        if len(exp_mask_idx) > 0:
            expert_probs_arr[exp_mask_idx] = 0.0

        student_logprobs_dict = student_output.logprobs[0]
        student_token_ids = np.array(list(student_logprobs_dict.keys()))
        student_probs_arr = np.array([math.exp(lp.logprob) for lp in student_logprobs_dict.values()])

        # suppress end think
        std_mask_idx = np.where(student_token_ids == suppress_token_id)[0]
        if len(std_mask_idx) > 0:
            student_probs_arr[std_mask_idx] = 0.0

        fallback = True

        # special rule for threshold = 0, 1 - we always choose expert or student
        if args.prob_threshold == 0:
            candidate_token_id = np.random.choice(expert_token_ids, p=expert_probs_arr / expert_probs_arr.sum())
            chosen_id = int(candidate_token_id)
            fallback = False
        elif args.prob_threshold == 1:
            # fall back to student
            pass
        # if use greedy, we can favor expert more
        elif args.use_greedy:
            expert_tokens_with_probs = [(token_id, expert_probs_arr[idx]) for idx, token_id in enumerate(expert_token_ids)]
            expert_tokens_with_probs = sorted(expert_tokens_with_probs, key=lambda item: item[1], reverse=True)
            # use highest probability expert that passes
            for candidate_token_id, _ in expert_tokens_with_probs:
                student_idx = np.where(student_token_ids == candidate_token_id)[0]
                if len(student_idx) > 0 and student_probs_arr[student_idx[0]] >= args.prob_threshold:
                    chosen_id = int(candidate_token_id)
                    fallback = False
                    break
        else:
            for _ in range(max_retries):
                candidate_token_id = np.random.choice(expert_token_ids, p=expert_probs_arr / expert_probs_arr.sum())
                student_idx = np.where(student_token_ids == candidate_token_id)[0]
                if len(student_idx) > 0 and student_probs_arr[student_idx[0]] >= args.prob_threshold:
                    chosen_id = int(candidate_token_id)
                    fallback = False
                    break
        
        if fallback:
            student_probs_normalized = student_probs_arr / student_probs_arr.sum()
            chosen_id = int(np.random.choice(student_token_ids, p=student_probs_normalized))
            num_fails += 1
        
        num_total += 1
        
        if chosen_id == tokenizer.eos_token_id: break
        
        expert_context_ids.append(chosen_id)
        student_context_ids.append(chosen_id)
        generated_token_ids.append(chosen_id)

    return tokenizer.decode(generated_token_ids), num_fails, num_total

# extract logic
async def run(cfg: DictConfig):
    args = cfg

    model_name = args.model_name.replace('/', '_')
    dataset_name = args.dataset.replace('/', '_')
    is_reasoning = bool(args.reasoning)

    if args.max_tokens is None:
        args.max_tokens = (
            args.max_tokens_reasoning if is_reasoning else args.max_tokens_non_reasoning
        )

    if args.prob_threshold is None:
        args.prob_threshold = (
            args.prob_threshold_reasoning if is_reasoning else args.prob_threshold_non_reasoning
        )


    # read in hf dataset from flag
    # NOTE: this data must have problem_id, problem and solution fields
    all_problems_data = datasets.load_dataset(args.dataset, split="train")

    # create eval_dir
    eval_dir = f"{args.eval_dir_base}_{dataset_name}_{model_name}_{args.prob_threshold}"
    if args.use_greedy:
        eval_dir += "_greedy"
    
    if args.answer_only:
        eval_dir += "_answer_only"
    
    if args.student_propose:
        eval_dir += "_student_propose"
    if args.num_samples > 1:
        eval_dir += f"_num_samples={args.num_samples}"

    if is_reasoning:
        eval_dir += f"_max_tokens={args.max_tokens}"
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(exist_ok=True)
    
    selected_indices = parse_index_spec(args.problems, len(all_problems_data))

    # collect tasks
    print("Collecting and preparing all generation tasks...")
    all_tasks = []
    all_metadata = []
    setup = False

    for line_idx in selected_indices:
        # grab problem data from hf dataset
        problem_data = all_problems_data[line_idx]
        pid = problem_data.get('problem_id', line_idx)
        
        # check if pid exists:
        out_path = eval_dir / f"{pid}.json"
        if os.path.exists(out_path):
            print(f'Skipping {out_path} ...')
            continue
        if not setup:
            setup = True
            await setup_engines(args)

        problem, solution = problem_data.get('problem'), problem_data.get('solution')
        start = ''
        if is_reasoning:
            start = '<think>'

        expert_prompt_template = EXPERT_PROMPT_TEMPLATE
        student_prompt_template = STUDENT_PROMPT_TEMPLATE

        # NOTE: this is a simple hack to do true speculative decoding - have the student drive the generation process
        if args.student_propose:
            expert_prompt_text = build_student_prompt(problem, student_prompt_template=student_prompt_template)  # Now has NO solution (Proposer)
            student_prompt_text = build_expert_prompt(problem, solution, answer_only=args.answer_only, tokenizer=tokenizer, max_len=args.max_seq_len, expert_prompt_template=expert_prompt_template)  # Now has the solution (Validator)
        else:
            expert_prompt_text = build_expert_prompt(problem, solution, answer_only=args.answer_only, tokenizer=tokenizer, max_len=args.max_seq_len, expert_prompt_template=expert_prompt_template)
            student_prompt_text = build_student_prompt(problem, student_prompt_template=student_prompt_template)
        expert_conversation = [{"role": "user", "content": expert_prompt_text}, {"role": "assistant", "content": start}]
        student_conversation = [{"role": "user", "content": student_prompt_text}, {"role": "assistant", "content": start}]
        
        # Create the coroutine for the task but don't run it yet
        for sample_id in range(args.num_samples):
            task = reverse_speculative_decode(expert_conversation, student_conversation, args)
            all_tasks.append(task)
            # Store metadata to map results back later
            all_metadata.append({"pid": pid, "problem": problem, "solution": solution, "sample_id": sample_id})

    print(f"Collected {len(all_tasks)} tasks. Running in batches of {args.batch_size}...")
    rows = defaultdict(lambda: {"problem_id": None, "problem": None, "solution": None, "num_fails": 0, "num_total": 0, "failure_rate": 0, "data": []})
    

    total_fallbacks = 0
    total_tokens_generated = 0
    
    
    task_batches = list(chunked(all_tasks, args.batch_size))
    metadata_batches = list(chunked(all_metadata, args.batch_size))
    # run tasks in batches to avoid memory issues and to allow for progress tracking
    for task_batch, metadata_batch in tqdm(zip(task_batches, metadata_batches), total=len(task_batches), desc="Processing Batches"):
        # Run all tasks in the current batch concurrently
        results = await asyncio.gather(*task_batch)
        
        # Process the results for this batch
        for meta, (generated_text, fails, total) in zip(metadata_batch, results):
            # Aggregate the stats
            total_fallbacks += fails
            total_tokens_generated += total
            
            # Store the data as before
            pid = meta["pid"]
            rows[pid].update({"problem_id": pid, "problem": meta["problem"], "solution": meta["solution"]})            
            rows[pid]["num_fails"] += fails
            rows[pid]["num_total"] += total

            rows[pid]["data"].append({
                "raw_expert_reasoning": [generated_text],
                "sample_id": meta["sample_id"]
            })

    print("Writing output files...")
    for pid, data in rows.items():
        if data["num_total"] > 0:
            data["failure_rate"] = data["num_fails"] / data["num_total"]
        else:
            data["failure_rate"] = 0.0
        out_path = eval_dir / f"{pid}.json"
        write_json_atomic(str(out_path), data)

    print(f"\nFinished. Wrote {len(rows)} files to: {eval_dir}")

@hydra.main(version_base=None, config_path="conf", config_name="generate")
def main(cfg: DictConfig):
    asyncio.run(run(cfg))

if __name__ == "__main__":
    main()
