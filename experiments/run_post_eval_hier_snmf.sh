#!/bin/bash


# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

INPUT_IN_RESULTS_JSON="${1:-experiments/artifacts/causal_results_in.json}"
OUTPUT_IN_JSON="${2:-experiments/artifacts/in_aggregated_causal_results.json}"
$LAYERS_OUTPUT_IN_JSON="${3:-experiments/artifacts/in_aggregated_causal_results_by_layer.json}"

INPUT_OUT_RESULTS_JSON="${1:-experiments/artifacts/out_causal_results.json}"
OUTPUT_OUT_JSON="${2:-experiments/artifacts/out_aggregated_causal_results.json}"
$LAYERS_OUTPUT_OUT_JSON="${3:-experiments/artifacts/out_aggregated_causal_results_by_layer.json}"

# Print configuration
echo "=========================================="
echo "Running Aggregate Of Causal experiment by aggregating concept scores as max of all kl/sign combinations"
echo "=========================================="
# echo "Input:    $INPUT_IN_RESULTS_JSON"
# echo "Output:   $OUTPUT_IN_JSON"
# echo "Sort by:  $SORT_BY"
# echo "=========================================="
# echo ""

# Run the aggregation script
python3 "$SCRIPT_DIR/evaluation/aggregate_causal_results.py" \
    --input "$INPUT_IN_RESULTS_JSON" \
    --concepts_output "$OUTPUT_IN_JSON" \
    --layers_output "$LAYERS_OUTPUT_IN_JSON"

python3 "$SCRIPT_DIR/evaluation/aggregate_causal_results.py" \
    --input "$INPUT_OUT_JSON" \
    --concepts_output "$OUTPUT_OUT_JSON" \
    --layers_output "$LAYERS_OUTPUT_OUT_JSON"

# Generate visualizations of aggregated resuls of the causal experiment


# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Aggregation completed successfully!"
else
    echo ""
    echo "✗ Aggregation failed!"
    exit 1
fi