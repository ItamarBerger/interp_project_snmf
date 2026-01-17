#!/bin/bash

set -euo pipefail

echo "Starting hierarchical SNMF steering experiment..."

STEPS="all"
DRY_RUN=0
LAYERS="0,3,6,9,11"

# Get args to control which steps to run
# If STEPS is "all", run all steps
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      IFS=',' read -r -a STEPS <<< "$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    --causal-save-path)
      CAUSAL_OUTPUT_PATH="$2"
      shift 2
      ;;
    --output-score-results)
      OUTPUT_SCORE_RESULTS="$2"
      shift 2
      ;;
    --input-score-results)
      INPUT_SCORE_RESULTS="$2"
      shift 2
      ;;
    --concepts-context-file)
      CONCEPTS_CONTEXT_FILE="$2"
      shift 2
      ;;
    --act-model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# Return error status code if model name is not provided
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "ERROR: --act-model-name is required" >&2
  exit 1
fi


if [[ -z "${CAUSAL_OUTPUT_PATH:-}" ]]; then
  CAUSAL_OUTPUT_PATH="experiments/artifacts/$MODEL_NAME/causal_output.json"
fi

FACTORIZATION_BASE_PATH="experiments/artifacts/$MODEL_NAME/hier"
OUTPUT_SCORE_RESULTS="experiments/artifacts/$MODEL_NAME/causal_results_out.json"
INPUT_SCORE_RESULTS="experiments/artifacts/$MODEL_NAME}/causal_results_in.json"
CONCEPTS_CONTEXT_FILE="experiments/artifacts/$MODEL_NAME/concept_contexts.json"
INPUT_DESCRIPTIONS_FILE="experiments/artifacts/$MODEL_NAME/input_descriptions.json"
VOCAB_PROJ_FILE="experiments/artifacts/$MODEL_NAME/vocab_proj.json"
OUTPUT_DESCRIPTIONS_FILE="experiments/artifacts/$MODEL_NAME/output_descriptions.json"


# If STEPS is "all", set it to run all steps
if [[ "${STEPS[0]}" == "all" ]]; then
  STEPS=("train" "generate_concept_context" "generate_input_descriptions" "generate_vocab_proj" "generate_output_descriptions" "generate_causal_output" "input_score_judge" "output_score_judge")
fi

echo "========== Overview =========="
echo "Steps to run: ${STEPS[*]}"
echo "Layers: $LAYERS"
echo "save path for causal output: $CAUSAL_OUTPUT_PATH"
echo "Using factorization base path: $FACTORIZATION_BASE_PATH"

echo "========== Starting Steps =========="

if [[ " ${STEPS[*]} " == *" train "* ]]; then
  echo "Running training step..."
  if [[ $DRY_RUN -eq 0 ]]; then
     PYTHONPATH=. python experiments/train/train_hier.py \
      --sparsity 0.01 \
      --ranks 400,200,100,50 \
      --max-iterations-per-layer 2000 \
      --patience 1500 \
      --ft-lr 1e-3 \
      --ft-iters 500 \
      --fine-tune \
      --model-name $MODEL_NAME \
      --factorization-mode mlp \
      --layers $LAYERS \
      --data-path data/hier_concepts.json \
      --model-device cuda \
      --data-device cpu \
      --fitting-device cuda \
      --base-path . \
      --save-path $FACTORIZATION_BASE_PATH \
      --seed 42
    fi
  echo "Training step completed."
fi

if [[ " ${STEPS[*]} " == *" generate_concept_context "* ]]; then
  echo "Generating concept contexts..."
  if [[ $DRY_RUN -eq 0 ]]; then
    PYTHONPATH=. python experiments/snmf_interp/generate_concept_context.py \
    --models-dir $FACTORIZATION_BASE_PATH \
    --output-json $CONCEPTS_CONTEXT_FILE \
    --data-path data/hier_concepts.json \
    --layers $LAYERS \
    --ranks 400,200,100,50 \
    --num-samples-per-factor 25 \
    --context-window 15 \
    --sparsity 0.01 \
    --seed 42 \
    --model-name $MODEL_NAME \
    --factor-mode mlp \
    --model-device cuda \
    --data-device cpu
  fi
  echo "Concept contexts generated."
fi

if [[ " ${STEPS[*]} " == *" generate_input_descriptions "* ]]; then
  echo "Generating input descriptions..."
  if [[ $DRY_RUN -eq 0 ]]; then
    PYTHONPATH=. python experiments/snmf_interp/generate_input_descriptions.py \
    --input-json $CONCEPTS_CONTEXT_FILE \
    --output-json $INPUT_DESCRIPTIONS_FILE \
    --model gemini-2.0-flash \
    --env-var GEMINI_API_KEY \
    --layers $LAYERS \
    --k-values 400,200,100,50 \
    --top-m 10 \
    --max-tokens 200 \
    --concurrency 50 \
    --retries 5
  fi
  echo "Input descriptions generated."
fi

if [[ " ${STEPS[*]} " == *" generate_vocab_proj "* ]]; then
  echo "Generating vocabulary projections..."
  if [[ $DRY_RUN -eq 0 ]]; then
   PYTHONPATH=. python experiments/snmf_interp/generate_vocab_proj.py\
    --model-name $MODEL_NAME \
    --base-path . \
    --factorization-base-path $FACTORIZATION_BASE_PATH \
    --output-path $VOCAB_PROJ_FILE \
    --layers $LAYERS \
    --ranks 400,200,100,50 \
    --top-k 75 \
    --sparsity 0.01 \
    --device cuda \
    --seed 123
  fi
  echo "Vocabulary projections generated."
fi

if [[ " ${STEPS[*]} " == *" generate_output_descriptions "* ]]; then
  echo "Generating output descriptions..."
  if [[ $DRY_RUN -eq 0 ]]; then
    PYTHONPATH=. python experiments/snmf_interp/generate_output_centric_descriptions.py \
  --input $VOCAB_PROJ_FILE \
  --output $OUTPUT_DESCRIPTIONS_FILE \
  --model gemini-2.0-flash \
  --layers $LAYERS \
  --ranks 400,200,100,50 \
  --concurrency 25 \
  --top-m 25 \
  --max-tokens 5000
  fi
  echo "Output descriptions generated."
fi

if [[ " ${STEPS[*]} " == *" generate_causal_output "* ]]; then
  echo "Generating causal output..."
  if [[ $DRY_RUN -eq 0 ]]; then
    PYTHONPATH=. python experiments/causal/generate_causal_output.py \
   --model-name $MODEL_NAME \
   --layers $LAYERS \
   --ranks 400,200,100,50 \
   --sparsity 0.01 \
   --factorization-base-path $FACTORIZATION_BASE_PATH \
   --save-path $CAUSAL_OUTPUT_PATH \
   --device cuda
  fi
  echo "Causal output generated."
fi



if [[ " ${STEPS[*]} " == *" input_score_judge "* ]]; then
  echo "Starting input score judging..."
  if [[ $DRY_RUN -eq 0 ]]; then
    PYTHONPATH=. python experiments/causal/input_score_llm_judge.py \
   --input $CAUSAL_OUTPUT_PATH \
   --concepts $INPUT_DESCRIPTIONS_FILE \
   --output $INPUT_SCORE_RESULTS \
   --model gemini-2.0-flash \
   --ranks 400,200,100,50 \
   --layers $LAYERS \
   --concurrency 25
  fi
  echo "Input score judging completed."
fi

if [[ " ${STEPS[*]} " == *" output_score_judge "* ]]; then
  echo "Starting output score judging..."
  if [[ $DRY_RUN -eq 0 ]]; then
    PYTHONPATH=. python experiments/causal/output_score_llm_judge.py \
    --input $CAUSAL_OUTPUT_PATH \
   --concepts $OUTPUT_DESCRIPTIONS_FILE \
   --output $OUTPUT_SCORE_RESULTS \
   --layers $LAYERS \
   --ranks 400,200,100,50 \
   --model gemini-2.0-flash \
   --concurrency 25 \
   --attempts 2 \
   --sparsity 0.01
  fi
    echo "Output score judging completed."
fi

echo "Finished."