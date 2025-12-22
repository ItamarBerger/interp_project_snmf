PYTHONPATH=. python experiments/causal/generate_causal_output.py \
  --model-name gpt2-small \
  --layers 0 \
  --ranks 400,200,100,50 \
  --sparsity 0.01 \
  --factorization-base-path experiments/artifacts/hier \
  --save-path experiments/artifacts/causal_output.json \
  --device cuda

echo "Finished."
