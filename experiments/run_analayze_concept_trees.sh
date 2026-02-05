#!/bin/bash

STEPS="all"
DRY_RUN=0
LAYERS="0,1,2,3,6,9,11"
RANKS="400,200,100,50"
MODEL_NAME="gpt2-small"
BASE_DIR="experiments/artifacts"

# Get args to control which steps to run
# If STEPS is "all", run all steps
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      IFS=',' read -r -a STEPS <<< "$2"
      shift 2
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    --ranks)
      RANKS="$2"
      shift 2
      ;;
    --act-model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --base-dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --output-path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_PATH:-}" ]]; then
  echo "Error: --output-path argument is required." >&2
  exit 1
fi

FACTORIZATION_BASE_PATH="$BASE_DIR/$MODEL_NAME/hier"

# If STEPS is "all", set it to run all steps
if [[ "${STEPS[0]}" == "all" ]]; then
  STEPS=("train" "analyze_concept_trees" "concept_trees_analysis_to_csv")
fi


if [[ " ${STEPS[@]} " =~ " train " ]]; then
    echo "Running concept tree analysis step..."
    #Train hier SNMF for specified model and layers - only if artifacts/hier directory is empty or non existing
    PYTHONPATH=. python experiments/train/train_hier.py \
        --sparsity 0.01 \
        --ranks $RANKS \
        --max-iterations-per-layer 2000 \
        --patience 1500 \
        --ft-lr 1e-3 \
        --ft-iters 500 \
        --fine-tune \
        --overwrite\
        --model-name $MODEL_NAME \
        --factorization-mode mlp \
        --layers $LAYERS \
        --data-path data/hier_concepts.json \
        --model-device cuda \
        --data-device cpu \
        --fitting-device cuda \
        --base-path . \
        --save-path $FACTORIZATION_BASE_PATH \
        --seed 42 \
        --wandb-mode disabled
        echo "Finished training hier SNMF models."
fi

if [[ " ${STEPS[@]} " =~ " analyze_concept_trees " ]]; then
    echo "Running concept tree analysis step..."
    #Analyze Concept Trees
    #Create concept tree analysis in the format of jsons
    PYTHONPATH=. python experiments/concept_trees/analayze_concept_trees.py \
      --factorization-base-path $FACTORIZATION_BASE_PATH \
      --model-name $MODEL_NAME \
      --layers $LAYERS \
      --ranks $RANKS \
      --output-path experiments/artifacts/concept_trees/

    echo "Finished analyzing concept trees."
fi

if [[ " ${STEPS[@]} " =~ " concept_trees_analysis_to_csv " ]]; then
    #Extract Key Findings into csv tables for easier visualization
    echo "Running concept trees analysis to csv step..."
    PYTHONPATH=. python experiments/concept_trees/concept_trees_analaysis_to_csv.py \
      --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
      --concept-trees-analysis-path experiments/artifacts/concept_trees/concept_trees_analysis \
      --output-path experiments/artifacts/concept_trees/concept_trees_analysis_visualizations/
    echo "Finished extracting concept trees analysis to csv."
fi


# Note : The following visualization scripts are commented out since they have only been locally tested on windows (seems like slurm had an issue with it).
# Save Visualization of Key Findings across levels
# export PYTHONPATH=.
# python experiments/artifacts/concept_trees/concept_trees_analysis_visualizations/visualize_key_findings_across_levels.py


# # Save Visualization of Key Findings per layer
# export PYTHONPATH=.
# python experiments/artifacts/concept_trees/concept_trees_analysis_visualizations/visualize_key_findings_per_layer.py
