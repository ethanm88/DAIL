#!/bin/bash

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
# Locked to vllm 0.11.0 and PyTorch 2.8.0
pip install --no-cache-dir "vllm==0.11.0" "torch==2.8.0" "tensordict==0.10.0" torchdata more-itertools


echo "2. install basic packages"
pip install "transformers[hf_xet]==4.57.3" accelerate datasets peft hf-transfer \
    "numpy==2.2.6" "pyarrow==20.0.0" pandas \
    codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler bitsandbytes nest-asyncio \
    pytest py-spy pyext pre-commit ruff

echo "3. install FlashAttention and FlashInfer"
# Python 3.10, PyTorch 2.8 wheel
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Matching FlashInfer wheel for PyTorch 2.8
wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.8-cp38-abi3-linux_x86_64.whl && \
    pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.8-cp38-abi3-linux_x86_64.whl

# Remove the downloaded wheel files
rm flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl flashinfer_python-0.2.2.post1+cu124torch2.8-cp38-abi3-linux_x86_64.whl

echo "Successfully installed all packages"