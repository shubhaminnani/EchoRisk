#!/usr/bin/env bash
# Train Tasks 1, 2, 3 sequentially on GPU 0.
# Usage:
#   bash train_all.sh              # fresh run
#   bash train_all.sh --resume     # continue each from its last.pth

set -e
GPU=0
RESUME=${1:-}

echo "=========================================="
echo " Training on CUDA_VISIBLE_DEVICES=$GPU"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python src_echoprime/train_task1.py $RESUME
echo ""
CUDA_VISIBLE_DEVICES=$GPU python src_echoprime/train_task2.py $RESUME
echo ""
CUDA_VISIBLE_DEVICES=$GPU python src_echoprime/train_task3.py $RESUME

echo ""
echo "All tasks finished."
