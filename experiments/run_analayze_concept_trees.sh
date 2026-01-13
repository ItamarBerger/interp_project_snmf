#!/bin/bash

# export PYTHONPATH=.
#  python experiments/analayze_concept_trees.py \
#   --factorization-base-path experiments/artifacts/hier/ \
#   --model-name gpt2-small \
#   --layers 0,3,6,9,11 \
#   --ranks 400,200,100,50 \
#   --output-path experiments/artifacts/concept_trees/


export PYTHONPATH=.
 python experiments/visualize_concept_trees_analaysis.py \
  --layers 0,3,6,9,11 \
  --concept-trees-analysis-path experiments/artifacts/concept_trees/concept_trees_analysis \
  --output-path experiments/artifacts/concept_trees/concept_trees_analysis_visualizations/
