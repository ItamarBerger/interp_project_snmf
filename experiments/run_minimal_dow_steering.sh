#!/usr/bin/env bash
# Load OPENAI_API_KEY from a private user env file if present
if [ -f "$HOME/.openai.env" ]; then
  source "$HOME/.openai.env"
fi

# Ensure OPENAI_API_KEY is available for LLM steps
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set. Create $HOME/.openai.env with: export OPENAI_API_KEY=..." >&2
  exit 1
fi

# Minimal test: days_of_week with just 3 ranks
# This should cost ~$1-2 and complete quickly

echo "==== Minimal SNMF Steering Test (days_of_week, 3 ranks) ===="
echo "Estimated cost: ~$1-2"
echo "Estimated time: ~30-45 minutes"
echo ""

# Training step (will skip if model exists)
PYTHONPATH=. python experiments/train/train.py \
    --sparsity 0.01 \
    --ranks 3 \
    --max-iterations-per-layer 1000 \
    --patience 500 \
    --model-name gpt2-small \
    --factorization-mode mlp \
    --layers 0 \
    --data-path data/days_of_week_dataset.json \
    --model-device cuda \
    --data-device cpu \
    --fitting-device cuda \
    --base-path . \
    --save-path experiments/results/ \
    --seed 42

# Step 1: Generate concept context
PYTHONPATH=. python experiments/snmf_interp/generate_concept_context.py \
  --models-dir experiments/results \
  --output-json experiments/results/dow_concept_contexts.json \
  --layers 0 \
  --ranks 3 \
  --num-samples-per-factor 10 \
  --context-window 10 \
  --sparsity 0.01 \
  --seed 42 \
  --model-name "gpt2-small" \
  --factor-mode mlp \
  --data-path data/days_of_week_dataset.json \
  --model-device cuda \
  --data-device cpu

# Step 2: Generate input descriptions (THIS COSTS MONEY)
PYTHONPATH=. python experiments/snmf_interp/generate_input_descriptions.py \
  --input-json experiments/results/dow_concept_contexts.json \
  --output-json experiments/results/dow_input_descriptions.json \
  --model gpt-4o-mini \
  --env-var OPENAI_API_KEY \
  --layers 0 \
  --k-values 3 \
  --top-m 3 \
  --max-tokens 200 \
  --concurrency 2 \
  --retries 3

# Step 3: Generate causal output
PYTHONPATH=. python experiments/causal/generate_causal_output.py \
  --model-name gpt2-small \
  --layers 0 \
  --ranks 3 \
  --sparsity 0.01 \
  --factorization-base-path experiments/results \
  --save-path experiments/results/dow_causal_output.json \
  --device cuda

# Step 4: Score causal inputs (THIS COSTS MONEY)
PYTHONPATH=. python experiments/causal/input_score_llm_judge.py \
  --input experiments/results/dow_causal_output.json \
  --concepts experiments/results/dow_input_descriptions.json \
  --output experiments/results/dow_causal_results_in.json \
  --model gpt-4o-mini \
  --ranks 3 \
  --layers 0 \
  --concurrency 2

echo ""
echo "==== Test Complete ===="
echo "Results saved to experiments/results/dow_*"
echo ""
echo "Key outputs:"
echo "  - dow_input_descriptions.json: What each factor represents"
echo "  - dow_causal_results_in.json: How well steering worked (THIS IS THE VALIDATION)"
echo ""
