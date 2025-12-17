PYTHONPATH=. python experiments/train/train.py \
    --sparsity 0.01 \
    --ranks 400,200,100,50 \
    --max-iterations-per-layer 2000 \
    --patience 1500 \
    --model-name gpt2-small \
    --factorization-mode mlp \
    --layers 0 \
    --data-path data/final_dataset_20_concepts.json \
    --model-device cuda \
    --data-device cpu \
    --fitting-device cuda \
    --base-path . \
    --save-path experiments/artifacts/ \
    --seed 42

echo "Finished."
