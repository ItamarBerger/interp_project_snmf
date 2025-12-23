
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


PYTHONPATH=. python experiments/snmf_interp/generate_concept_context.py \
  --models-dir experiments/artifacts/hier \
  --output-json experiments/artifacts/concept_contexts.json \
  --layers 0,3,6,9,11 \
  --ranks 400,200,100,50 \
  --num-samples-per-factor 25 \
  --context-window 15 \
  --sparsity 0.01 \
  --seed 42 \
  --model-name "gpt2-small" \
  --factor-mode mlp \
  --data-path data/final_dataset_20_concepts.json \
  --model-device cuda \
  --data-device cpu

# PYTHONPATH=. python experiments/causal/generate_causal_output.py \
#   --model-name gpt2-small \
#   --layers 0,3,6,9,11 \
#   --ranks 400,200,100,50 \
#   --sparsity 0.01 \
#   --factorization-base-path experiments/artifacts/hier \
#   --save-path experiments/artifacts/causal_output.json \
#   --device cuda

echo "Finished."
