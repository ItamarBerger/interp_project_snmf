# Hierarchical Concepts in Transformer MLPs: Causal Relevance and Structure

This repository contains the code for the paper: **"Hierarchical Concepts in Transformer MLPs: Causal Relevance and Structure"** by Tslil Shani, Itamar Berger, and Ohad Sharet.

This project is a fork of and builds upon the codebase from [Shafran et al. (2025)](https://github.com/ordavid-s/snmf-mlp-decomposition), which introduced Semi-Nonnegative Matrix Factorization (SNMF) for decomposing MLP activations in transformers into interpretable features and demonstrated the emergence of hierarchical structure within those features.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Our Contribution](#our-contribution)
3. [Setup](#setup)
4. [Data](#data)
5. [Experiments](#experiments)
6. [Results](#results)
7. [Repository Structure](#repository-structure)
8. [Tutorials](#tutorials)

---

## Motivation

Understanding how large language models represent information is a central challenge in mechanistic interpretability. A key principle in this endeavor is **causal relevance** - features that influence model outputs in predictable ways provide stronger evidence of functional importance than those that are merely correlational.

Shafran et al. (2025) showed that applying SNMF to MLP activations yields interpretable features, and that recursively decomposing the resulting feature matrix produces hierarchical concept trees - where higher-level concepts emerge as compositions of lower-level sub-concepts. For example, fine-grained concepts like "Saturday" and "Sunday" can merge into a broader feature representing "Weekend."

While these findings are promising, the **causal relevance** of hierarchical concepts and their **structural organization** across layers remained largely unexplored. Prior work provided illustrative examples but did not rigorously evaluate whether these features causally affect model outputs.

---

## Our Contribution

This work investigates hierarchical features discovered via SNMF along two axes:

1. **Causal Steerability**: We evaluate whether hierarchical features at different levels of abstraction can causally steer model outputs through activation interventions, and how steering effectiveness varies with hierarchy depth.

2. **Structural Analysis**: We characterize the structural properties of discovered hierarchies. Including tree depth, branching factor, and compositionality patterns across different layers of the model.

### Key Findings

- Features across hierarchy levels in early-to-mid layers exhibit a **decline in steering performance as depth increases**.
- Structural analysis reveals **layer-specific organization**: some layers exhibit deep, multi-level hierarchies while others show shallower compositions.
- Concept descriptions tend to converge at higher hierarchy levels, suggesting that recursive decomposition surfaces increasingly shared semantic content.
- Steering score trajectories within concept trees show that domain-specific root concepts tend to exhibit lower scores, while morphosyntactic root concepts tend to retain higher scores across levels.

### Models Evaluated

- **GPT-2 Small** 
- **Gemma-2-2B** 
- **Llama-3.1-8B** 

---

## Setup

### Installation

```bash
git clone https://github.com/ItamarBerger/interp_project_snmf.git
cd interp_project_snmf
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the root directory with the following API key:

```bash
GEMINI_API_KEY=ABC....
```

**API Key Usage:**
- **GEMINI_API_KEY**: Used throughout the hierarchical steering pipeline - for generating concept descriptions (via Gemini 2.5 Flash), output-centric descriptions (via Gemini 2.0 Flash), and for LLM-as-judge scoring of steering results (via Gemini 2.0 Flash).

---

## Data

The `data/` directory contains datasets compatible with the repository's dataset abstraction:

1. **final_dataset_20_concepts.json** - Dataset of randomly sampled concepts used to train SNMF in the paper
2. **hier_concepts.json** - Dataset constructed with ConceptNet containing hierarchical concepts used for hierarchical SNMF- decomposition.

---

## Experiments

All experiment scripts are located in the `experiments/` directory.

### Training Hierarchical SNMF - `run_hier_snmf.sh`

Trains hierarchical SNMF models across multiple transformer layers. For each layer, the script applies recursive SNMF decomposition, producing a multi-level factorization where each level represents concepts at a different granularity. The trained models are saved as artifacts for downstream analysis and steering.

```bash
cd experiments
./run_hier_snmf.sh
```

### Hierarchical Steering - `run_hier_snmf_steering.sh`

This is the central experiment of the paper. It runs the full causal steering pipeline for hierarchical SNMF features. From training through activation intervention to LLM-judged evaluation. The pipeline consists of the following steps:

1. **Train** - Train hierarchical SNMF models (if not already available).
2. **Generate Concept Contexts** - Extract top activating token contexts for each factor at every hierarchy level.
3. **Generate Input Descriptions** - Use an LLM to produce concise concept labels from the extracted contexts.
4. **Generate Vocabulary Projections** - Project each factor onto the model's vocabulary to characterize its output-side behavior.
5. **Generate Output Descriptions** - Use an LLM to produce output-centric descriptions from vocabulary projections.
6. **Generate Causal Output** - Perform activation steering interventions by amplifying each factor and recording the model's modified outputs.
7. **Input Score Judge** - An LLM judge scores how well the steered output reflects the input-based concept description.
8. **Output Score Judge** - An LLM judge scores how well the steered output reflects the output-based concept description.

Individual steps can be selected via the `--steps` flag, allowing partial reruns without repeating the full pipeline.

### Analyzing Concept Trees - `run_analayze_concept_trees.sh`

Analyzes the structural properties of the hierarchical decompositions:

1. **Train** - Optionally trains hierarchical SNMF models (if not already available).
2. **Analyze Concept Trees** - Builds concept trees from the trained hierarchical models by tracing how top-level factors decompose into sub-concepts across levels. Computes structural statistics such as tree depth, branching factor, and leaf distribution per layer.
3. **Export Statistics** - Extracts key findings into tabular format for visualization.

---

## Results

The `experiments/results/` directory contains steering evaluation results organized by hierarchy configuration and model.

### Hierarchy Configurations

We evaluate two decomposition depths:

| Configuration | Levels | Ranks per Level |
|---|---|---|
| **k400** (4-level) | 4 | 400 → 200 → 100 → 50 |
| **k800** (5-level) | 5 | 800 → 400 → 200 → 100 → 50 |

The `k400` configuration provides 4 levels of abstraction, while `k800` adds a finer-grained base level (800 factors) for a 5-level hierarchy. Both configurations share the same top-level granularity (50 factors).

### Directory Layout

Results are organized by configuration and model:

```
experiments/results/
├── k400/                       # 4-level decomposition
│   ├── gemma-2-2b/
│   ├── gpt2-small/
│   └── meta-llama/Llama-3.1-8B/
└── k800/                       # 5-level decomposition
    ├── gemma-2-2b/
    └── gpt2-small/
```

Each model directory contains:
- **results_in.json** - Raw steering evaluation scores across layers and hierarchy levels.
- **plots_in/barplots/** - Mean steering scores across hierarchy levels and layers.
- **plots_in/boxplots/** - Score distributions showing variance across factors and layers.

---

## Repository Structure

```
.
├── data/                           # Concept datasets
├── data_utils/                     # Dataset loading and processing
├── factorization/                  # SNMF and Hierarchical SNMF implementations
├── llm_utils/                      # Transformer activation generation
├── intervention/                   # Activation steering tools
├── experiments/
│   ├── train/                      # Training scripts (standard and hierarchical)
│   ├── snmf_interp/                # Factor interpretation (descriptions, vocab projection)
│   ├── causal/                     # Causal steering pipeline
│   ├── evaluation/
│   │   └── concept_trees/          # Concept tree construction and analysis
│   ├── results/                    # Evaluation results and visualizations
│   ├── run_hier_snmf.sh            # Train hierarchical SNMF
│   ├── run_hier_snmf_steering.sh   # Full causal steering pipeline
│   └── run_analayze_concept_trees.sh  # Concept tree structural analysis
├── tracker/                        # Experiment tracking (W&B integration)
└── utils/                          # Logging and general utilities
```

---

## Tutorials

Two Jupyter notebooks demonstrate the core SNMF functionality:

1. **snmf_tutorial.ipynb** - Train standard SNMF end-to-end: process data, generate MLP activations, factorize, and analyze discovered factors.
2. **hierarchial_nmf_tutorial.ipynb** - Train recursive (hierarchical) SNMF and visualize the concept trees identified in the MLP.

---

**Questions or issues?** Please open an issue and we will do our best to respond in a timely manner.
