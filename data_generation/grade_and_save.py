import os
import json
import re
import shutil
import argparse
from typing import Dict, List, Optional

BASE_DIR = "."
DEFAULT_OUTPUT_DIR = "correct_only_sampled"
DEFAULT_SPECIFIED_DIRS: List[str] = []
def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Finds all occurrences of \boxed{...} expressions in the text
    and returns the content inside the braces of the *last* one.
    """

    matches = re.findall(r'\\boxed{([^}]+)}', text)
    
    if matches:
        # Return the last item in the list
        return matches[-1].strip()
    else:
        # Return None if no boxed answer is found
        return None

def _hyperparam_label(dir_name: str) -> str:
    if 'answer' in dir_name:
        hyperparam = dir_name.split('_answer_only')[0].split('_')[-1] + " (answer_only)"
    else:
        hyperparam = dir_name.split('_')[-1]
    if 'student' in dir_name:
        hyperparam += " (student_propose)"
    return hyperparam

def _should_skip_dir(dir_name: str, dir_path: str) -> bool:
    if not os.path.isdir(dir_path) or '_' not in dir_name or 'expert_completions' not in dir_name:
        return True
    if dir_name in {"correct_only", "correct_only_sampled"}:
        return True
    return False

def _copy_all_json_files(dir_path: str, target_dir_path: str) -> Dict[str, int]:
    counts = {"copied": 0, "total": 0}
    for filename in os.listdir(dir_path):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(dir_path, filename)
        counts["total"] += 1
        dest_file_path = os.path.join(target_dir_path, filename)
        try:
            shutil.copy2(file_path, dest_file_path)
            counts["copied"] += 1
        except Exception as e:
            print(f"  [Error] Failed to copy {file_path} to {dest_file_path}: {e}")
    return counts

def grade_and_copy(
    base_dir: str = BASE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    specified_dirs: Optional[List[str]] = None,
    reasoning: bool = False,
    input_dir: Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}

    if input_dir:
        base_dir = os.path.dirname(input_dir) or "."
        specified_dirs = [os.path.basename(input_dir)]
    if specified_dirs is None:
        specified_dirs = DEFAULT_SPECIFIED_DIRS

    print(f"Scanning subdirectories in: {os.path.abspath(base_dir)}\n")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Correct files will be saved in: {os.path.abspath(output_dir)}\n")

    dirs = os.listdir(base_dir) if len(specified_dirs) == 0 else specified_dirs
    skip_grading = reasoning
    for dir_name in dirs:
        dir_path = os.path.join(base_dir, dir_name)

        if _should_skip_dir(dir_name, dir_path):
            continue

        hyperparam = _hyperparam_label(dir_name)
        if hyperparam not in results:
            results[hyperparam] = {'correct': 0, 'total': 0}
            print(f"Found new hyperparameter: {hyperparam} (from dir: {dir_name})")

        target_dir_path = os.path.join(output_dir, dir_name)
        os.makedirs(target_dir_path, exist_ok=True)

        if skip_grading:
            counts = _copy_all_json_files(dir_path, target_dir_path)
            results[hyperparam]['total'] += counts["total"]
            results[hyperparam]['correct'] += counts["copied"]
            continue

        for filename in os.listdir(dir_path):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(dir_path, filename)
            results[hyperparam]['total'] += 1

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                solution_text = data.get("solution", "")
                expert_data = data.get("data", [])
                if not expert_data or not isinstance(expert_data, list) or \
                   "raw_expert_reasoning" not in expert_data[0] or \
                   not expert_data[0]["raw_expert_reasoning"]:
                    print(f"  [Warning] Skipping {file_path}: Missing 'raw_expert_reasoning'")
                    continue

                expert_texts = [expert_data[i]["raw_expert_reasoning"][0] for i in range(len(expert_data)) if "raw_expert_reasoning" in expert_data[i]]

                solution_answer = extract_boxed_answer(solution_text)
                expert_answers = [extract_boxed_answer(text) for text in expert_texts]

                if solution_answer is None:
                    print(f"  [Warning] Skipping {file_path}: No boxed answer found in 'solution' field.")
                elif not expert_answers or all(ans is None for ans in expert_answers):
                    print(f"  [Warning] Skipping {file_path}: No boxed answer found in 'expert_reasoning' field.")
                elif any(solution_answer == ans for ans in expert_answers):
                    results[hyperparam]['correct'] += 1
                    dest_file_path = os.path.join(target_dir_path, filename)
                    try:
                        shutil.copy2(file_path, dest_file_path)
                        print(f"  [Correct] Copied {filename} to correct_only/{dir_name}/")
                    except Exception as e:
                        print(f"  [Error] Failed to copy {file_path} to {dest_file_path}: {e}")
                else:
                    print(f"  [Mismatch] {filename}: Sol='{solution_answer}', Exp='{', '.join([str(ans) for ans in expert_answers])}'")

            except Exception as e:
                print(f"  [Error] Could not process {file_path}: {e}")

    print("\n--- Final Results ---")

    for hyperparam in sorted(results.keys()):
        counts = results[hyperparam]
        total = counts['total']
        correct = counts['correct']

        if total > 0:
            accuracy = (correct / total) * 100
        else:
            accuracy = 0.0

        print(f"\n## Hyperparameter: {hyperparam}")
        print(f"  Total Files:    {total}")
        print(f"  Correct Files:  {correct}")
        print(f"  Accuracy:       {accuracy:.2f}%")
        print("-----------------------")

    return results

def main():
    parser = argparse.ArgumentParser(description="Grade and copy expert completions.")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR, help="Directory to scan for expert completion folders.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save copied files.")
    parser.add_argument("--dirs", nargs="+", default=None, help="Specific subdirectories to process.")
    parser.add_argument("--input_dir", type=str, default=None, help="Process a single directory.")
    parser.add_argument("--reasoning", action="store_true", help="Skip grading and copy all files.")
    args = parser.parse_args()

    grade_and_copy(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        specified_dirs=args.dirs,
        reasoning=args.reasoning,
        input_dir=args.input_dir,
    )

if __name__ == "__main__":
    main()
