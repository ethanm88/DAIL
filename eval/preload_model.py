#!/usr/bin/env python3
import argparse
import time

from vllm import LLM


def main():
    parser = argparse.ArgumentParser(description="Warm up model weights before distributed eval.")
    parser.add_argument("--model_name", required=True, help="Model name or local path.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--max_model_len", type=int, default=8000)
    parser.add_argument("--hold_seconds", type=int, default=0)
    args = parser.parse_args()

    print(f"Preloading model: {args.model_name}")
    LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    print("Preload complete.")

    if args.hold_seconds > 0:
        time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
