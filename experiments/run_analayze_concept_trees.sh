#!/bin/bash


#Train hier SNMF for specified model and layers - only if artifacts/hier directory is empty or non existing
PYTHONPATH=. python experiments/train/train_hier.py \
    --sparsity 0.01 \
    --ranks 400,200,100,50 \
    --max-iterations-per-layer 2000 \
    --patience 1500 \
    --ft-lr 1e-3 \
    --ft-iters 500 \
    --fine-tune \
    --overwrite\
    --model-name gpt2-small \
    --factorization-mode mlp \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --data-path data/hier_concepts.json \
    --model-device cpu \
    --data-device cpu \
    --fitting-device cuda \
    --base-path . \
    --save-path experiments/artifacts/hier/ \
    --seed 42 \
    --wandb-mode disabled
    echo "Finished."


#Create concept tree analysis in the format of jsons
export PYTHONPATH=.
 python experiments/analayze_concept_trees.py \
  --factorization-base-path experiments/artifacts/hier/ \
  --model-name gpt2-small \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
  --ranks 400,200,100,50 \
  --output-path experiments/artifacts/concept_trees/


#Extract Key Findings into csv tables for easier visualization
export PYTHONPATH=.
 python experiments/concept_trees_analaysis_to_csv.py \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
  --concept-trees-analysis-path experiments/artifacts/concept_trees/concept_trees_analysis \
  --output-path experiments/artifacts/concept_trees/concept_trees_analysis_visualizations/


# Note : The following visualization scripts are commented out since they have only been locally tested on windows (seems like slurm had an issue with it).
# Save Visualization of Key Findings across levels
# export PYTHONPATH=.
# python experiments/artifacts/concept_trees/concept_trees_analysis_visualizations/visualize_key_findings_across_levels.py


# # Save Visualization of Key Findings per layer
# export PYTHONPATH=.
# python experiments/artifacts/concept_trees/concept_trees_analysis_visualizations/visualize_key_findings_per_layer.py
