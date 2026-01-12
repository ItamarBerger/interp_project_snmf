#!/bin/bash

export PYTHONPATH=.
 python experiments/analayze_concept_trees.py \
  --factorization-base-path experiments/artifacts/hier/ \
  --model-name gpt2-small \
  --layers 0,3,6,9,11 \
  --ranks 400,200,100,50 \
  --output-path experiments/artifacts/concept_trees/