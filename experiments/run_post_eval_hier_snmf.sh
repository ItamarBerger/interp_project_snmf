#!/bin/bash


# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Input paths
INPUT_IN_RESULTS_JSON="${1:-experiments/artifacts/causal_results_in.json}"
INPUT_OUT_RESULTS_JSON="${2:-experiments/artifacts/causal_results_out.json}"

# Output paths for aggregated concepts
OUTPUT_IN_CONCEPTS_JSON="experiments/artifacts/in_aggregated_causal_results.json"
OUTPUT_OUT_CONCEPTS_JSON="experiments/artifacts/out_aggregated_causal_results.json"

# Output paths for layer aggregations
LAYERS_OUTPUT_IN_JSON="experiments/artifacts/in_aggregated_causal_results_by_layer.json"
LAYERS_OUTPUT_OUT_JSON="experiments/artifacts/out_aggregated_causal_results_by_layer.json"

# Output paths for layer-level summaries (new)
LAYER_LEVEL_IN_JSON="experiments/artifacts/in_layer_level_summary.json"
LAYER_LEVEL_OUT_JSON="experiments/artifacts/out_layer_level_summary.json"

# Output directories for visualizations
VIZ_OUTPUT_IN_DIR="experiments/artifacts/visualizations_in"
VIZ_OUTPUT_OUT_DIR="experiments/artifacts/visualizations_out"

# Print configuration
echo "=========================================="
echo "Running Post-Evaluation Analysis Pipeline"
echo "=========================================="
echo "Processing input-centric and output-centric causal results"
echo ""

# Run the aggregation script for input-centric results
echo "[1/4] Aggregating input-centric results..."
python3 "$SCRIPT_DIR/evaluation/aggregate_causal_results.py" \
    --input "$INPUT_IN_RESULTS_JSON" \
    --concepts_output "$OUTPUT_IN_CONCEPTS_JSON" \
    --layers_output "$LAYERS_OUTPUT_IN_JSON" \
    --layer-level-output "$LAYER_LEVEL_IN_JSON"

if [ $? -ne 0 ]; then
    echo "✗ Input-centric aggregation failed!"
    exit 1
fi

# Run the aggregation script for output-centric results
echo ""
echo "[2/4] Aggregating output-centric results..."
python3 "$SCRIPT_DIR/evaluation/aggregate_causal_results.py" \
    --input "$INPUT_OUT_RESULTS_JSON" \
    --concepts_output "$OUTPUT_OUT_CONCEPTS_JSON" \
    --layers_output "$LAYERS_OUTPUT_OUT_JSON" \
    --layer-level-output "$LAYER_LEVEL_OUT_JSON"

if [ $? -ne 0 ]; then
    echo "✗ Output-centric aggregation failed!"
    exit 1
fi

# Generate visualizations for input-centric results
echo ""
echo "[3/4] Generating visualizations for input-centric results..."
python3 "$SCRIPT_DIR/evaluation/viauslaize_aggregated_causal_results.py" \
    --layer-level-input "$LAYER_LEVEL_IN_JSON" \
    --concepts-input "$OUTPUT_IN_CONCEPTS_JSON" \
    --output-dir "$VIZ_OUTPUT_IN_DIR"

if [ $? -ne 0 ]; then
    echo "✗ Input-centric visualization failed!"
    exit 1
fi

# Generate visualizations for output-centric results
echo ""
echo "[4/4] Generating visualizations for output-centric results..."
python3 "$SCRIPT_DIR/evaluation/viauslaize_aggregated_causal_results.py" \
    --layer-level-input "$LAYER_LEVEL_OUT_JSON" \
    --concepts-input "$OUTPUT_OUT_CONCEPTS_JSON" \
    --output-dir "$VIZ_OUTPUT_OUT_DIR"

if [ $? -ne 0 ]; then
    echo "✗ Output-centric visualization failed!"
    exit 1
fi

# All steps completed successfully
echo ""
echo "=========================================="
echo "✓ Post-Evaluation Analysis Complete!"
echo "=========================================="
echo ""
echo "Output files generated:"
echo "  Input-centric:"
echo "    - Concepts: $OUTPUT_IN_CONCEPTS_JSON"
echo "    - Layers: $LAYERS_OUTPUT_IN_JSON"
echo "    - Layer-Level Summary: $LAYER_LEVEL_IN_JSON"
echo "    - Visualizations: $VIZ_OUTPUT_IN_DIR/"
echo ""
echo "  Output-centric:"
echo "    - Concepts: $OUTPUT_OUT_CONCEPTS_JSON"
echo "    - Layers: $LAYERS_OUTPUT_OUT_JSON"
echo "    - Layer-Level Summary: $LAYER_LEVEL_OUT_JSON"
echo "    - Visualizations: $VIZ_OUTPUT_OUT_DIR/"
echo ""