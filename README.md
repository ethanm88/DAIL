# Making Expert Reasoning Learnable with Self-Distillation

<a href='https://arxiv.org/abs/2602.02405'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/datasets/emendes3/e1-proof'><img src='https://img.shields.io/badge/🤗-e1_proof-blue'></a> <a href='https://huggingface.co/datasets/emendes3/e1-verifiable'><img src='https://img.shields.io/badge/🤗-e1_verifiable-blue'></a>


## Overview

This repository contains the official code for the DAIL algorithm presented in the paper "Making Expert Reasoning Learnable with Self-Distillation". The following instructions walk through running on default reasoning model settings. Non-reasoning models can be run by omitting the `--reasoning` and changing the model name.



> [!IMPORTANT]
> As described in the paper as an offline method, DAIL incorporates embarrassingly parallel data generation, and as such, assumes access to a SLURM cluster. YMMV if you are running on a different setup.

## Setup

### Setup Environment
Create and activate the conda environment, then install requirements:

```
conda create -n dail python=3.10 -y
conda activate dail
bash install_dependencies.sh
```

### Download datasets
First download the training datasets from huggingface in python shell.
```python
from datasets import load_dataset
load_dataset("emendes3/e1-proof")
load_dataset("emendes3/e1-verifiable")
```

### Download model
Download the model once to avoid repeated Hugging Face Hub requests from array jobs:

```
cd data_generation
python download_model.py --model_name Qwen/Qwen3-8B
```

## Data Generation

1. Submit the Slurm job:

```
cd data_generation && \
sbatch \
  --account=<ACCOUNT> \
  --partition=<PARTITION> \
  --gres=gpu:<GPU_TYPE>:<NUM_GPUS> \
  --cpus-per-task=<CPUS> \
  --array <ARRAY_RANGE> \
  generate_expert_completions.sbatch \
  --model_name <model> \
  --dataset <dataset> \
  --reasoning \
  --dataset_size <N>
```

The sbatch wrapper accepts `--model_name`, `--dataset`, `--reasoning`, and optional `--dataset_size`. If `--dataset_size` is omitted, the script loads the Hugging Face dataset and computes its length on the fly. It then splits the problems across array jobs and passes a `problems` range to the Python script. Please adjust the cluster specific sbatch parameters (account, partition, GPUs, CPUs and array range) as needed.

Use `--reasoning` for reasoning models; omit it for non-reasoning. If `prob_threshold` is unset in `conf/generate.yaml`, the script defaults to `prob_threshold_reasoning` for reasoning and `prob_threshold_non_reasoning` otherwise. 

Output is written under `data_generation/` in a folder named:
`expert_completions_<MODEL>_<PROB_THRESHOLD>[_greedy][_answer_only][_student_propose][_num_samples=N][_max_tokens=M]`,
containing one `<problem_id>.json` per problem. Slurm logs go to `data_generation/job-outputs/%x-%A_%a.out` and `.err`.


> [!NOTE]
> You can override the default hyperparameters and settings in the `conf` directory

2. Extract the final dataset (this runs grade/copy automatically). 


```
python extract_final_dataset.py model_name=<model> --reasoning
```

This copies/filters into `data_generation/correct_only_sampled/<expert_completions_...>/` and writes the final training file to
`data_generation/training_data/<expert_completions_...>/train.jsonl`.



## Train


1. Submit the Slurm job:

```
cd train && \
sbatch \ 
  --account=<ACCOUNT> \
  --partition=<PARTITION> \
  --gres=gpu:<GPU_TYPE>:<NUM_GPUS> \
  --cpus-per-task=<CPUS> \
launch_training.sbatch \
  --model_name <model> \
  --reasoning
```

The sbatch wrapper accepts `--model_name` and `--reasoning`. Again, omit `--reasoning` for non-reasoning runs. Logs go to `train/job-outputs/`.

> [!IMPORTANT]
> The current settings in ddp_config.yaml are for 4 GPUs. You will need to adjust this config per your hardware.


## Evaluation

From the repo root:

1. Set DATA_DIR environment variable to the directory you want to store the evaluation datasets.

  ```
  export DATA_DIR=<path_to_data_dir>
  ```

2. Download evaluation datasets from Hugging Face.

  ```python
  from datasets import load_dataset
  load_dataset("emendes3/aime_2024")
  load_dataset("emendes3/aime_2025")
  load_dataset("emendes3/beyond_aime")
  load_dataset("emendes3/imo_answer")
  ```

3. Preload the model before distributed inference:

  ```
  cd eval && \
  python preload_model.py --model_name Qwen/Qwen3-8B
  ```

4. Adjust your cluster specific sbatch parameters in ``eval_array.sbatch``

5. Submit inference jobs:

  * For reasoning sweeps:

    ```
    cd eval
    bash run_eval_dataset_reasoning.sh <dataset_name>
    ```
  * For non-reasoning:

    ```
    cd eval
    bash run_eval_dataset.sh <dataset_name>
    ```

Outputs are written under `eval/raw_eval_results_answer/` in folders named:
`dataset=<DATASET>_<MAX_TOKENS>_model=<MODEL>_enabling_thinking=<BOOL>_greedy=<BOOL>[_temperature=<T>][_no_force_answer]`,
with one `problem_<id>.json` per problem. Slurm logs are by default saved to `eval/job-outputs/`.

6. Compute metrics:

```
bash collate_reasoning.sh
```

This computes scores for all models specified in `score_configs/reasoning_models.yaml` and writes summaries to `results/<dataset>_summary.json`.


## Citation
```
@inproceedings{
    mendes2026selfdistill,
    title={Making Expert Reasoning Learnable with Self-Distillation},
    author={Mendes, Ethan and Park, Jungsoo and Ritter, Alan},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=JG6f02X29a}
  }
```
