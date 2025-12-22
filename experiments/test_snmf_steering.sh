
PYTHONPATH=. python experiments/snmf_interp/generate_concept_context.py \
--models-dir experiments/artifacts/hier/ \
--output-json experiments/artifacts/concept_contexts.json \
--layers 0 \
--ranks 400,200,100,50 \
--num-samples-per-factor 25 \
--context-window 15 \
--sparsity 0.01 \
--seed 42 \
--model-name "gpt2-small" \
--factor-mode mlp \
--data-path data/final_dataset_20_concepts.json \
--model-device cudas \
--data-device cuda

echo "Finished."
