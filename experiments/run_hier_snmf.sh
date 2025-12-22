#!/bin/bash


PYTHONPATH=. python experiments/train/train_hier.py \
    --sparsity 0.01 \
    --ranks 400,200,100,50 \
    --max-iterations-per-layer 2000 \
    --patience 1500 \
    --ft-lr 1e-3 \
    --ft-iters 500 \
    --fine-tune \
    --model-name gpt2-small \
    --factorization-mode mlp \
    --layers 0,3,6,9,11 \
    --data-path data/final_dataset_20_concepts.json \
    --model-device cuda \
    --data-device cpu \
    --fitting-device cuda \
    --base-path . \
    --save-path experiments/artifacts/hier/ \
    --seed 42

    echo "Finished."

