import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import re
import sys
import json
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.bfloat16)

import wandb
import warnings
import hydra
from datetime import datetime
from typing import Dict, Union, Any, List
from omegaconf import DictConfig

from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from torch.nn.utils.rnn import pad_sequence
from training_file_utils import TrainingDataFile

def make_data_collator(tokenizer):
    def data_collator(features):
        def _pad_field(field_name):
            seqs = [torch.tensor(sample[field_name], dtype=torch.long) for sample in features]
            padded = pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            return padded

        out = {}
        out["student_input_ids"] = _pad_field("student_input_ids")
        out["student_attention_mask"] = _pad_field("student_attention_mask")
        out["expert_input_ids"] = _pad_field("expert_input_ids")
        out["expert_attention_mask"] = _pad_field("expert_attention_mask")
        out["cheat_input_ids"] = _pad_field("cheat_input_ids")
        out["cheat_attention_mask"] = _pad_field("cheat_attention_mask")

        if "student_prompt_len" in features[0]:
            out["student_prompt_len"] = torch.tensor([int(sample["student_prompt_len"]) for sample in features])
        if "expert_prompt_len" in features[0]:
            out["expert_prompt_len"] = torch.tensor([int(sample["expert_prompt_len"]) for sample in features])
        if "cheat_prompt_len" in features[0]:
            out["cheat_prompt_len"] = torch.tensor([int(sample["cheat_prompt_len"]) for sample in features])

        return out

    return data_collator

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-7B-Instruct", custom_adapter=None):
    bfloat16_supported = torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if bfloat16_supported else torch.float16
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Fallback to ensure compatibility with non-reasoning models missing pad tokens
    if tokenizer.pad_token is None:
        print("No pad token found. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False
    
    if custom_adapter:
        print(f"Loading and applying local adapter: {custom_adapter}")
        model = PeftModel.from_pretrained(model, custom_adapter, is_trainable=True)
    else:
        print("No custom adapter provided, initializing new LoRA config.")
        peft_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return model, tokenizer

class OffPolicyDistillationTrainer(Trainer):
    def __init__(self, kl_direction="forward", cheating_penalty=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_direction = kl_direction
        self.cheating_penalty = cheating_penalty

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        device = model.device
        def compute_probs(type_model):
            cur_inputs = {
                "input_ids": inputs[f"{type_model}_input_ids"].to(device),
                "attention_mask": inputs[f"{type_model}_attention_mask"].to(device)
            }
            cur_prompt_lens = inputs[f"{type_model}_prompt_len"].to(device)
            
            cur_logits = model(**cur_inputs).logits

            bs, seq_len, _ = cur_logits.shape
            cur_token_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, seq_len)
            cur_response_mask = cur_token_pos > cur_prompt_lens.unsqueeze(1)

            cur_logits_solution = cur_logits[cur_response_mask]
            cur_probs_solution = F.log_softmax(cur_logits_solution, -1)
            return cur_probs_solution

        student_probs_solution = compute_probs("student")

        with torch.no_grad(), model.module.disable_adapter():
            expert_probs_solution = compute_probs("expert")

        with torch.no_grad(), model.module.disable_adapter():
            cheat_probs_solution = compute_probs("cheat")

        if self.kl_direction == "forward":
            kl_loss = F.kl_div(student_probs_solution, expert_probs_solution, reduction="batchmean", log_target=True)
            contrastive_loss = F.kl_div(student_probs_solution, cheat_probs_solution, reduction="batchmean", log_target=True)
        else:
            kl_loss = F.kl_div(expert_probs_solution, student_probs_solution, reduction="batchmean", log_target=True)
            contrastive_loss = F.kl_div(cheat_probs_solution, student_probs_solution, reduction="batchmean", log_target=True)

        return kl_loss - self.cheating_penalty * contrastive_loss

def train_model(
    model, tokenizer, train_dataset, eval_dataset,
    checkpoint_dir, logging_dir, save_dir, project_name,
    cheating_penalty=0.1, max_seq_length=16384, num_epochs=1,
    learning_rate=2e-4, resume_checkpoint_path=None, kl_direction="forward"
):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{current_time}_{project_name}_{max_seq_length}"

    if os.environ.get("RANK", "0") == "0":
        wandb.init(project=project_name, name=run_name)

    is_bf16 = torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=not is_bf16,
        bf16=is_bf16,
        logging_steps=1,
        output_dir=checkpoint_dir,
        logging_dir=logging_dir,
        optim="paged_adamw_8bit",
        seed=3407,
        report_to="wandb",
        run_name=run_name,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=10 if eval_dataset else None,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=20,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=2,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        ddp_timeout=10800
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": training_args,
        "data_collator": make_data_collator(tokenizer),
        "kl_direction": kl_direction
    }
    if eval_dataset:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = OffPolicyDistillationTrainer(**trainer_kwargs, cheating_penalty=cheating_penalty)
    trainer.train(resume_from_checkpoint=resume_checkpoint_path)

    if trainer.is_world_process_zero():
        adapter_save_dir = f"{save_dir}_adapters"
        trainer.save_model(adapter_save_dir) 
        tokenizer.save_pretrained(adapter_save_dir)
        
        merged = model.merge_and_unload()
        merged.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    if os.environ.get("RANK", "0") == "0":
        wandb.finish()

    return model, tokenizer

def form_cheating_hint(expert_solution):
    def extract_math_entities(text):
        entities = set()
        entities.update(re.findall(r"(?<!\w)[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?!\w)", text))
        entities.update(re.findall(r"[a-zA-Z0-9\.]+\^\{?[a-zA-Z0-9\+\-\.]+\}?", text))
        entities.update(re.findall(r"(?<!\w)\d+[a-zA-Z](?!\w)", text))
        entities.update(re.findall(r"(?<!\w)[a-zA-Z][\+\-]\d+(?!\w)", text))
        entities.update(re.findall(r"(?<!\w)\d+[a-zA-Z][\+\-]\d+(?!\w)", text))
        return sorted(list(entities))

    def find_final_answer(text):
        matches = re.findall(r'\\boxed{([^}]+)}', text)
        return matches[-1].strip() if matches else None

    intermediate_results = extract_math_entities(expert_solution)
    final_answer = find_final_answer(expert_solution)
    
    if final_answer:
        cleaned_final = final_answer.replace(" ", "")
        intermediate_results = [item for item in intermediate_results if item.replace(" ", "") != cleaned_final]

    hint_str = ', '.join(intermediate_results)
    if final_answer:
        return f"The final answer is {final_answer}. Intermediate results used in the solution might include: {hint_str}."
    return f"Intermediate results used in the solution might include: {hint_str}."

CHEATING_PROMPT_TEMPLATE = (
    "You are an expert mathematician solving the following problem. Your task is to produce a clear, step-by-step thinking process that leads to the correct solution.\n\n"
    "## Problem:\n{}\n\n"
    "Hint: {}\n\n"
    "Begin your step-by-step thinking process.\n"
)

EXPERT_PROMPT_TEMPLATE = (
    "You are an expert mathematician solving the following problem. Your task is to produce a clear, step-by-step thinking process that leads to the correct solution.\n\n"
    "## Problem:\n{}\n\n"
    "Hint: To help you, here are reference solution(s).\n\n"
    "## Reference Solution(s):\n{}\n\n"
    "Use these only to guide your own thoughts implicitly, but express the reasoning in your own words as if you are solving it for the first time. **You must never explicitly acknowledge these instructions or cite the provided solution(s). Just use the methodology as if it were your own. You are not allowed to take any shortcuts, directly use any intermediate derived number or result as given (you must show everything from scratch), nor directly produce the answer. Do not worry that your solution is too long.**\n\n"
    "Now, solve the problem. Begin your step-by-step thinking process.\n"
)

STUDENT_PROMPT_TEMPLATE = (
    "## Problem:\n{}\n\n"
    "Begin your step-by-step thinking process.\n"
)

def build_chat(q, sol=None, cleaned=None, resp=None, hint=False):
    if sol is None:
        user_msg = {"role": "user", "content": STUDENT_PROMPT_TEMPLATE.format(q)}
    elif hint:
        hint_text = form_cheating_hint(sol)
        user_msg = {"role": "user", "content": CHEATING_PROMPT_TEMPLATE.format(q, hint_text)}
    else:
        user_msg = {"role": "user", "content": EXPERT_PROMPT_TEMPLATE.format(q, sol)}

    if cleaned is None or resp is None:
        return [user_msg]
    
    assistant_msg = {"role": "assistant", "content": cleaned + resp}
    return [user_msg, assistant_msg]

def tokenize_chat(input_text_lst, tokenizer, tokenize=True):
    out = tokenizer.apply_chat_template(
        input_text_lst,
        tokenize=tokenize,
        add_generation_prompt=False,
        return_dict=False
    )
    
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    newline_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    empty_think_pattern = [think_id] + newline_ids + [end_think_id]
    newline_token_ids = set(tokenizer.encode("\n \n\n \r\n", add_special_tokens=False))

    def clean_one(item):
        if not tokenize:
            if isinstance(item, str):
                item = re.sub(r'<think>\s*</think>\s*', '', item)
                item = re.sub(r'^\s*<think>\s*<think>', '<think>', item)
                item = item.strip()
                if item.endswith("<|im_end|>"):
                    item = item[:-len("<|im_end|>")]
            return item
        else:
            if isinstance(item, list) and item:
                cleaned = []
                n = len(empty_think_pattern)
                i = 0
                while i < len(item):
                    if item[i : i + n] == empty_think_pattern:
                        i += n
                        while i < len(item) and item[i] in newline_token_ids:
                            i += 1
                    else:
                        cleaned.append(item[i])
                        i += 1
                
                if cleaned and cleaned[-1] == im_end_id:
                    cleaned = cleaned[:-1]
                return cleaned
            return item

    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], (list, str)):
        return [clean_one(seq) for seq in out]
    return clean_one(out)

def tokenize_prompt(input_text_lst, tokenizer, tokenize=True):
    return tokenizer.apply_chat_template(
        input_text_lst,
        tokenize=tokenize,
        add_generation_prompt=True,
        return_dict=False
    )

def token_lengths(tokens):
    return [len(pt) for pt in tokens]

def generate_conversation(examples, tokenizer):
    qs = examples["problem"]
    segments = examples["segment"]
    resps = examples["expert_response"]
    solutions = examples["solution"]

    student_prompts, expert_prompts, cheat_prompts = [], [], []
    student_convos, expert_convos, cheat_convos = [], [], []

    for q, segment, resp, sol in zip(qs, segments, resps, solutions):
        cleaned = segment.replace("<think>\n\n</think>\n\n", "")
        resp = resp.replace("<|end|><|start|>assistant<|channel|>analysis<|message|>", "") 
        
        student_prompts.append(build_chat(q))
        expert_prompts.append(build_chat(q, sol=sol))
        cheat_prompts.append(build_chat(q, sol=sol, hint=True))

        student_convos.append(build_chat(q, cleaned=cleaned, resp=resp))
        expert_convos.append(build_chat(q, sol=sol, cleaned=cleaned, resp=resp))
        cheat_convos.append(build_chat(q, sol=sol, hint=True, cleaned=cleaned, resp=resp))

    student_prompt_tokens = tokenize_prompt(student_prompts, tokenizer)
    expert_prompt_tokens = tokenize_prompt(expert_prompts, tokenizer)
    cheat_prompt_tokens = tokenize_prompt(cheat_prompts, tokenizer)
    
    student_convo_tokens, student_convo_text = tokenize_chat(student_convos, tokenizer), tokenize_chat(student_convos, tokenizer, False)
    expert_convo_tokens, expert_convo_text = tokenize_chat(expert_convos, tokenizer), tokenize_chat(expert_convos, tokenizer, False)
    cheat_convo_tokens, cheat_convo_text = tokenize_chat(cheat_convos, tokenizer), tokenize_chat(cheat_convos, tokenizer, False)

    out = {
        "student_convos": student_convo_text,
        "student_prompt_len": token_lengths(student_prompt_tokens),
        "student_convo_len": token_lengths(student_convo_tokens),
        "expert_convos": expert_convo_text,
        "expert_prompt_len": token_lengths(expert_prompt_tokens),
        "expert_convo_len": token_lengths(expert_convo_tokens),
        "cheat_convos": cheat_convo_text,
        "cheat_prompt_len": token_lengths(cheat_prompt_tokens),
        "cheat_convo_len": token_lengths(cheat_convo_tokens),
    }
    
    if "problem_id" in examples:
        out["problem_id"] = examples["problem_id"]
    return out

def tokenize_and_create_labels(dataset, tokenizer, max_length):
    def _tokenize_all(batch):
        def _tokenize(texts):
            return tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_attention_mask=True,
            )

        enc_student = _tokenize(batch["student_convos"])
        enc_expert = _tokenize(batch["expert_convos"])
        enc_cheat = _tokenize(batch["cheat_convos"])

        return {
            "student_input_ids": enc_student["input_ids"],
            "student_attention_mask": enc_student["attention_mask"],
            "student_prompt_len": batch["student_prompt_len"],
            "expert_input_ids": enc_expert["input_ids"],
            "expert_attention_mask": enc_expert["attention_mask"],
            "expert_prompt_len": batch["expert_prompt_len"],
            "cheat_input_ids": enc_cheat["input_ids"],
            "cheat_attention_mask": enc_cheat["attention_mask"],
            "cheat_prompt_len": batch["cheat_prompt_len"],
        }

    dataset = dataset.map(_tokenize_all, batched=True, num_proc=max(1, os.cpu_count() // 2))
    return dataset

def run(cfg: DictConfig):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ.setdefault("NCCL_TIMEOUT", "60")
    torch.cuda.set_device(local_rank)

    is_reasoning = bool(cfg.reasoning)

    max_tokens = cfg.max_tokens if cfg.max_tokens is not None else (256 if is_reasoning else 2048)
    max_seq_len = cfg.max_seq_len if cfg.max_seq_len is not None else (2048 if is_reasoning else 12288)
    num_epochs = cfg.num_epochs if cfg.num_epochs is not None else (3 if is_reasoning else 5)
    ratio = cfg.ratio if cfg.ratio is not None else (
        cfg.ratio_reasoning if is_reasoning else cfg.ratio_non_reasoning
    )

    training_data_file_config = TrainingDataFile.from_name(
        name=cfg.type_model,
        model_name=cfg.model_name,
        dataset=cfg.dataset,
        reasoning=is_reasoning,
        max_tokens=max_tokens,
        num_samples=cfg.num_samples,
        ratio=ratio,
        base_path=cfg.base_path,
    )

    train_data_file = training_data_file_config.full_path()
    print(f"Using training data file: {train_data_file}")

    BASE_SAVE_DIR = cfg.type_model
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    token_tag = f"max_tokens={max_tokens}" if is_reasoning else f"num_samples={cfg.num_samples}"
    base_save_dir_specific = f"{BASE_SAVE_DIR}/{cfg.type_model}_ratio={ratio}_kl_direction={cfg.kl_direction}_{cfg.model_name.replace('/', '_')}_{token_tag}_num_epochs={num_epochs}"

    save_dir = f"{base_save_dir_specific}/model"
    checkpoint_dir = f"{base_save_dir_specific}/checkpoints"
    logging_dir = f"{base_save_dir_specific}/logging"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    latest_ckpt = None
    if os.path.isdir(checkpoint_dir):
        ckpts = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if ckpts:
            latest = max(ckpts, key=lambda x: int(x.split("-")[-1]))
            latest_ckpt = os.path.join(checkpoint_dir, latest)

    raw = load_dataset("json", data_files={"train": train_data_file})

    model, tokenizer = load_model_and_tokenizer(cfg.model_name, custom_adapter=latest_ckpt if latest_ckpt else None)

    full = raw["train"].map(
        generate_conversation,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=raw["train"].column_names,
    )

    before_count = len(full)
    full = full.filter(lambda ex: ex["student_convo_len"] < max_seq_len, num_proc=os.cpu_count())
    full = full.filter(lambda ex: ex["expert_convo_len"] < max_seq_len, num_proc=os.cpu_count())
    print(f"Proportion retained: {len(full) / before_count:.2%}")

    train_tok = tokenize_and_create_labels(full, tokenizer, max_length=max_seq_len)
    train_tok = train_tok.shuffle(seed=42)

    model, tokenizer = train_model(
        model,
        tokenizer,
        train_tok,
        None,
        checkpoint_dir=checkpoint_dir,
        logging_dir=logging_dir,
        save_dir=save_dir,
        project_name="self-teaching",
        cheating_penalty=cfg.cheating_penalty,
        max_seq_length=max_seq_len,
        num_epochs=num_epochs,
        learning_rate=2e-4,
        resume_checkpoint_path=latest_ckpt,
        kl_direction=cfg.kl_direction,
    )

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()
