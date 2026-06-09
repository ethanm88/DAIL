import os
from dataclasses import dataclass

@dataclass
class TrainingDataFile:
    """
    Holds the configuration and constructs file paths for training data.
    """

    model_name: str
    dataset: str
    ratio: float
    student_propose: bool
    answer_only: bool
    use_solution: bool = False
    distillation: bool = False
    reasoning: bool = False

    base_path: str = "../data_generation/training_data"
    max_tokens: int = 256
    num_samples: int = 32

    @classmethod
    def from_name(
        cls,
        name: str,
        model_name: str,
        dataset: str,
        reasoning: bool,
        max_tokens: int,
        num_samples: int,
        base_path: str = "../data_generation/training_data",
        ratio: float = 0.8,
    ):
        common_params = {
            "model_name": model_name,
            "dataset": dataset,
            "max_tokens": max_tokens,
            "num_samples": num_samples,
            "base_path": base_path,
            "reasoning": reasoning,
        }

        if name in [
            "off_policy_distillation",
        ]:
            return cls(
                ratio=ratio,
                student_propose=True,
                answer_only=False,
                use_solution=False,
                **common_params,
            )
        if name in ["dail"]:
            return cls(
                ratio=1.0,
                student_propose=True,
                answer_only=False,
                use_solution=False,
                **common_params,
            )
        raise ValueError(f"Unknown data file name configuration: {name}")

    def filename_suffix(self):
        dataset_name = self.dataset.replace("/", "_")
        suffix = f"expert_completions_{dataset_name}_{self.model_name.replace('/', '_')}_{self.ratio}"
        if self.student_propose:
            suffix += "_student_propose"
        if self.answer_only:
            suffix += "_answer_only"
        if self.reasoning:
            suffix += f"_max_tokens={self.max_tokens}"
        elif self.num_samples > 1:
            suffix += f"_num_samples={self.num_samples}"
        return suffix

    def full_path(self):
        full_path = os.path.join(self.base_path, self.filename_suffix(), "train.jsonl")
        print(f"Constructed training data file path: {full_path}")
        return full_path
