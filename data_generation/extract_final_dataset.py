import os
import re
import json
import random
import hydra
from tqdm import tqdm
from collections import defaultdict
from omegaconf import DictConfig

from grade_and_save import grade_and_copy

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

def save_training_file_non_reasoning(file_path, save_dir="training_data"):
    # Configuration
    disallowed_contests_years = {"2024", "2025"}
    random.seed(42)

    # Collate data from files
    collated_data = defaultdict(list)
    model_name = "Qwen" + file_path.split('_Qwen')[-1]

    print(f"Processing files from: {file_path}")
    for f_name in tqdm(os.listdir(file_path)):
        if any(d in f_name for d in disallowed_contests_years):
            continue

        try:
            with open(os.path.join(file_path, f_name), "r", encoding="utf-8") as f:
                cur_data = json.load(f)
                problem_id = cur_data.get("problem_id")
                problem = cur_data.get("problem")
                solution = cur_data.get("solution")
                data = cur_data.get("data", [])
                answer = extract_boxed_answer(solution)
                for idx, entry in enumerate(data):
                    segment = entry.get("segment")
                    expert_response = entry.get("raw_expert_reasoning", [None])[0]
                    if expert_response and 'Qwen2.5' in model_name:
                        expert_response = expert_response.split("<|im_start|>user\n")[-1]
                    
                    found = False
                    expert_answer = extract_boxed_answer(expert_response) if expert_response else None
                    if answer is not None and expert_answer is not None and answer == expert_answer:
                        found = True
                    if found:
                        collated_data[problem_id].append(
                            {
                                "problem_id": problem_id,
                                "problem": problem,
                                "solution": solution,
                                "segment_idx": idx,
                                "segment": segment,
                                "expert_response": expert_response,
                            }
                        )
                        break # Only need one correct expert response per problem
        except Exception as e:
            print(f"Error processing file {f_name}: {e}")

    # Prepare data for saving
    os.makedirs(save_dir, exist_ok=True)
    base_name = re.sub(r"_num_samples=\d+", "", os.path.basename(file_path))
    training_data_dir = os.path.join(save_dir, base_name)
    os.makedirs(training_data_dir, exist_ok=True)

    all_variants = []
    print("Collating and shuffling variants...")
    for problem_id, variants in tqdm(collated_data.items()):
        random.shuffle(variants) # Shuffle variants within each problem
        all_variants.extend(variants)
    # Shuffle all variants from all problems together
    random.shuffle(all_variants)

    # Save to a single file
    training_data_file = os.path.join(training_data_dir, "train.jsonl")
    num_saved_entries = 0
    
    print(f"Saving data to {training_data_file}...")
    with open(training_data_file, "w", encoding="utf-8") as f:
        for entry in all_variants:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            num_saved_entries += 1

    print(f"Saved {num_saved_entries} total entries to {training_data_file}")

def save_training_file_reasoning(file_path, save_dir="training_data"):
    # Configuration
    random.seed(42)

    # Collate data from files
    collated_data = defaultdict(list)
    model_name = "Qwen" + file_path.split('_Qwen')[-1]

    print(f"Processing files from: {file_path}")
    for f_name in tqdm(os.listdir(file_path)):
        try:
            with open(os.path.join(file_path, f_name), "r", encoding="utf-8") as f:
                cur_data = json.load(f)
                problem_id = cur_data.get("problem_id")
                problem = cur_data.get("problem")
                solution = cur_data.get("solution")
                data = cur_data.get("data", [])
                for idx, entry in enumerate(data):
                    segment = entry.get("segment")
                    expert_response = entry.get("raw_expert_reasoning", [None])[0]
                    expert_response = expert_response.split('</think>')[0]
                    expert_response = '<think>' + expert_response
                    collated_data[problem_id].append(
                        {
                            "problem_id": problem_id,
                            "problem": problem,
                            "solution": solution,
                            "segment_idx": idx,
                            "segment": '', # keep empty field; entries are multiple solution variants
                            "expert_response": expert_response,
                        }
                    )
        except Exception as e:
            print(f"Error processing file {f_name}: {e}")

    # Prepare data for saving
    os.makedirs(save_dir, exist_ok=True)
    base_name = re.sub(r"_num_samples=\d+", "", os.path.basename(file_path))
    training_data_dir = os.path.join(save_dir, base_name)
    os.makedirs(training_data_dir, exist_ok=True)

    all_variants = []
    print("Collating and shuffling variants...")
    for problem_id, variants in tqdm(collated_data.items()):
        random.shuffle(variants)  # Shuffle variants within each problem
        if variants:
            all_variants.append(random.choice(variants))

    # Shuffle all variants from all problems together
    random.shuffle(all_variants)

    # Save to a single file
    training_data_file = os.path.join(training_data_dir, "train.jsonl")
    num_saved_entries = 0
    
    print(f"Saving data to {training_data_file}...")
    with open(training_data_file, "w", encoding="utf-8") as f:
        for entry in all_variants:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            num_saved_entries += 1

    print(f"Saved {num_saved_entries} total entries to {training_data_file}")

def get_raw_data_dir(
    base_dir=".",
    dataset="emendes3/e1-proof",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    ratio=0.8,
    student_propose=False,
    num_samples=32,
    answer_only=False,
    max_tokens=2048,
    reasoning=False,
):
    dataset_name = dataset.replace("/", "_")
    base_file_name = f"{base_dir}/expert_completions_{dataset_name}_{model_name.replace('/', '_')}_{ratio}"
    if answer_only:
        base_file_name += "_answer_only"
    if student_propose:
        base_file_name += "_student_propose"
    if num_samples > 1:
        base_file_name += f"_num_samples={num_samples}"
    if reasoning:
        base_file_name += f"_max_tokens={max_tokens}"
    return base_file_name

@hydra.main(version_base=None, config_path="conf", config_name="extract")
def main(cfg: DictConfig):
    # Validate dataset argument for consistency
    allowed_datasets = {"emendes3/e1-proof", "emendes3/e1-verifiable"}
    if cfg.dataset not in allowed_datasets:
        raise ValueError(
            f"Invalid dataset '{cfg.dataset}'. Allowed datasets: {', '.join(sorted(allowed_datasets))}"
        )

    is_reasoning = bool(cfg.reasoning)

    max_tokens = cfg.max_tokens
    if max_tokens is None:
        max_tokens = 256 if is_reasoning else 2048

    raw_data_dir = get_raw_data_dir(
        base_dir=".",
        dataset=cfg.dataset,
        model_name=cfg.model_name,
        ratio=cfg.ratio,
        student_propose=cfg.student_propose,
        num_samples=cfg.num_samples,
        answer_only=cfg.answer_only,
        max_tokens=max_tokens,
        reasoning=is_reasoning,
    )

    grade_and_copy(
        input_dir=raw_data_dir,
        output_dir="correct_only_sampled",
        reasoning=is_reasoning,
    )

    processed_dir = os.path.join("correct_only_sampled", os.path.basename(raw_data_dir))

    if is_reasoning:
        save_training_file_reasoning(processed_dir)
    else:
        save_training_file_non_reasoning(processed_dir)

if __name__ == "__main__":
    main()
