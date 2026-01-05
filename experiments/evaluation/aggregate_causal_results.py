import json
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple


def parse_judge_output(input_file: str) -> List[Dict]:
    """Load the JSON output from input_score_llm_judge.py or output_score_llm_judge.py"""
    with open(input_file, 'r') as f:
        return json.load(f)


def aggregate_by_feature(entries: List[Dict]) -> Tuple[Dict[Tuple, Dict], List[Dict]]:
    """
    Group entries by (layer, level, h_row) concept and compute max scores over kl/sign combination.
    For each concept:
      1. Group by (kl, intervention_sign) combination
      2. Compute average of sentence_results for each combination
      3. Take maximum across all combinations

    Returns:
      - A dict mapping (layer, level, h_row) -> aggregated data
      - A list of skipped concepts (where description is null)
    """
    # Group by concept identifier (layer, level, h_row)
    concept_groups = defaultdict(list)
    skipped_concepts = []

    for entry in entries:
        # Extract concept identifier
        layer = entry["layer"]
        level = entry.get("level", 0)
        h_row = entry["h_row"]

        concept_key = (layer, level, h_row)
        concept_groups[concept_key].append(entry)

    # Compute max scores for each concept
    aggregated = {}

    for concept_key, group_entries in concept_groups.items():
        layer, level, h_row = concept_key

        # Check if description is null - skip if so
        first_entry = group_entries[0]
        description = first_entry.get("description")

        if description is None:
            skipped_concepts.append({
                "layer": layer,
                "level": level,
                "h_row": h_row,
                "reason": "description is null"
            })
            continue

        # Group by (kl, intervention_sign) combination
        kl_sign_groups = defaultdict(list)

        for entry in group_entries:
            kl = entry.get("kl")
            intervention_sign = entry.get("intervention_sign")
            kl_sign_key = (kl, intervention_sign)
            kl_sign_groups[kl_sign_key].append(entry)

        # For each kl/sign combination, compute average scores
        combination_avg_scores = []

        for kl_sign_key, kl_sign_entries in kl_sign_groups.items():
            # Collect all scores from sentence results for this combination
            combo_concept_scores = []
            combo_final_scores = []
            combo_fluency_scores = []

            for entry in kl_sign_entries:
                for sent_result in entry.get("sentence_results", []):
                    concept_score = sent_result.get("concept_score")
                    final_score = sent_result.get("final_score")
                    fluency_score = sent_result.get("fluency_score")

                    if concept_score is not None:
                        combo_concept_scores.append(concept_score)
                    if final_score is not None:
                        combo_final_scores.append(final_score)
                    if fluency_score is not None:
                        combo_fluency_scores.append(fluency_score)

            # Compute averages for this combination
            avg_concept = sum(combo_concept_scores) / len(combo_concept_scores) if combo_concept_scores else None
            avg_final = sum(combo_final_scores) / len(combo_final_scores) if combo_final_scores else None
            avg_fluency = sum(combo_fluency_scores) / len(combo_fluency_scores) if combo_fluency_scores else None

            if avg_final is not None:  # Only include if we have valid scores
                combination_avg_scores.append({
                    "kl": kl_sign_key[0],
                    "intervention_sign": kl_sign_key[1],
                    "avg_concept_score": avg_concept,
                    "avg_final_score": avg_final,
                    "avg_fluency_score": avg_fluency,
                    "num_sentences": len(combo_final_scores)
                })

        # Find the maximum average final score across all combinations
        if combination_avg_scores:
            best_combination = max(combination_avg_scores, key=lambda x: x["avg_final_score"])

            # Collect metadata
            K = first_entry.get("K", "SAE")
            sparsity = first_entry.get("sparsity")

            # Collect all unique kl and intervention_sign values
            all_kls = sorted(set(entry.get("kl") for entry in group_entries if entry.get("kl") is not None))
            all_signs = sorted(set(entry.get("intervention_sign") for entry in group_entries if entry.get("intervention_sign") is not None))

            # Important: Recall that we mainly care about max scores here. becuase this is the score from the originalpaper.
            aggregated[concept_key] = {
                "K": K,
                "layer": layer,
                "level": level,
                "h_row": h_row,
                "description": description,
                "sparsity": sparsity,
                "max_avg_concept_score": best_combination["avg_concept_score"],
                "max_avg_final_score": best_combination["avg_final_score"],
                "max_avg_fluency_score": best_combination["avg_fluency_score"],
                "best_kl": best_combination["kl"],
                "best_intervention_sign": best_combination["intervention_sign"],
                "num_sentences_best": best_combination["num_sentences"],
                "num_kl_sign_combinations": len(combination_avg_scores),
                "all_kl_values": all_kls,
                "all_intervention_signs": all_signs
            }

    return aggregated, skipped_concepts
def aggregate_by_layer(aggregated_concepts: List[Dict]) -> Dict[int, Dict]:
    """
    Aggregate concepts by layer by averaging their max scores.

    Returns:
      - A dict mapping layer -> aggregated statistics
    """
    layer_groups = defaultdict(list)

    for concept in aggregated_concepts:
        layer = concept["layer"]
        layer_groups[layer].append(concept)

    aggregated_by_layer = {}

    for layer, concepts in layer_groups.items():
        concept_scores = [
            c["max_avg_concept_score"]
            for c in concepts
            if c.get("max_avg_concept_score") is not None
        ]
        final_scores = [
            c["max_avg_final_score"]
            for c in concepts
            if c.get("max_avg_final_score") is not None
        ]
        fluency_scores = [
            c["max_avg_fluency_score"]
            for c in concepts
            if c.get("max_avg_fluency_score") is not None
        ]

        aggregated_by_layer[layer] = {
            "layer": layer,
            "num_concepts": len(concepts),
            "avg_max_concept_score": (
                sum(concept_scores) / len(concept_scores)
                if concept_scores else None
            ),
            "avg_max_final_score": (
                sum(final_scores) / len(final_scores)
                if final_scores else None
            ),
            "avg_max_fluency_score": (
                sum(fluency_scores) / len(fluency_scores)
                if fluency_scores else None
            ),
            "num_valid_concept_scores": len(concept_scores),
            "num_valid_final_scores": len(final_scores),
            "num_valid_fluency_scores": len(fluency_scores),
        }

    return aggregated_by_layer


def aggregate_by_layer_and_level(aggregated_concepts: List[Dict]) -> Dict:
    """
    Aggregate concepts by layer and level.
    
    For each layer, group concepts by level and compute average max scores.
    
    Returns:
      - A dict where keys are layers, and values contain level-wise statistics
        Structure: {
            layer: {
                "layer": layer,
                "levels": {
                    level: {
                        "level": level,
                        "final_score": avg_max_final_score,
                        "concept_score": avg_max_concept_score,
                        "fluency_score": avg_max_fluency_score,
                        "num_concepts": count
                    }
                }
            }
        }
    """
    # Group by (layer, level)
    layer_level_groups = defaultdict(lambda: defaultdict(list))
    
    for concept in aggregated_concepts:
        layer = concept["layer"]
        level = concept.get("level", 0)
        layer_level_groups[layer][level].append(concept)
    
    # Compute statistics for each layer and level
    result = {}
    
    for layer, level_dict in layer_level_groups.items():
        levels_data = {}
        
        for level, concepts in level_dict.items():
            # Extract scores
            concept_scores = [
                c["max_avg_concept_score"]
                for c in concepts
                if c.get("max_avg_concept_score") is not None
            ]
            final_scores = [
                c["max_avg_final_score"]
                for c in concepts
                if c.get("max_avg_final_score") is not None
            ]
            fluency_scores = [
                c["max_avg_fluency_score"]
                for c in concepts
                if c.get("max_avg_fluency_score") is not None
            ]
            
            levels_data[level] = {
                "level": level,
                "final_score": (
                    sum(final_scores) / len(final_scores)
                    if final_scores else None
                ),
                "concept_score": (
                    sum(concept_scores) / len(concept_scores)
                    if concept_scores else None
                ),
                "fluency_score": (
                    sum(fluency_scores) / len(fluency_scores)
                    if fluency_scores else None
                ),
                "num_concepts": len(concepts),
                "num_valid_final_scores": len(final_scores),
                "num_valid_concept_scores": len(concept_scores),
                "num_valid_fluency_scores": len(fluency_scores)
            }
        
        result[layer] = {
            "layer": layer,
            "levels": levels_data
        }
    
    return result

    


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LLM judge scores by feature, taking maximum scores across all entries."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON (output from input_score_llm_judge.py or output_score_llm_judge.py)"
    )
    parser.add_argument(
        "--concepts_output",
        required=True,
        help="Path to write aggregated JSON"
    )
    parser.add_argument(
        "--sort-by",
        choices=["max_avg_concept_score", "max_avg_final_score", "max_avg_fluency_score"],
        default="max_avg_final_score",
        help="Sort results by this metric (default: max_avg_final_score)"
    )
    parser.add_argument(
        "--layers-output",
        required=True,
        help="Path to write aggregated JSON by layers"
    )
    parser.add_argument(
        "--layer-level-output",
        required=False,
        default=None,
        help="Path to write layer-level summary JSON (each layer with its levels and scores)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("AGGREGATE MAX SCORES BY FEATURE")
    print("="*80)

    # Load input
    print(f"\n[STEP 1/3] Loading input from {args.input}...")
    entries = parse_judge_output(args.input)
    print(f"  ✓ Loaded {len(entries)} entries")

    # Aggregate
    print(f"\n[STEP 2/3] Aggregating scores by concept...")
    aggregated, skipped_concepts = aggregate_by_feature(entries)
    print(f"  ✓ Found {len(aggregated)} unique concepts")
    print(f"  ✓ Skipped {len(skipped_concepts)} concepts (description is null)")

    # Log skipped concepts
    if skipped_concepts:
        print("\n  Skipped concepts:")
        for skipped in skipped_concepts:
            print(f"    - Layer {skipped['layer']}, Level {skipped['level']}, h_row {skipped['h_row']}: {skipped['reason']}")

    # Convert to list and sort
    result_list = list(aggregated.values())

    # Sort by specified metric (descending, None values last)
    sort_key = args.sort_by
    result_list.sort(
        key=lambda x: (x[sort_key] is None, -(x[sort_key] if x[sort_key] is not None else 0))
    )

    print(f"  ✓ Sorted by {sort_key} (descending)")

    # Save output
    print(f"\n[STEP 3.1/3] Saving results to {args.concepts_output}...")
    with open(args.concepts_output, 'w') as f:
        json.dump(result_list, f, indent=2)

    # Save aggregation where each entry is a layer with the scores obtained for each level
    print(f"\n[STEP 3.2/4] Saving results to {args.layers_output}...")

    aggregated_by_layer = aggregate_by_layer(result_list)
    with open(args.layers_output, 'w') as f:
        json.dump(aggregated_by_layer, f, indent=2)

    print(f"  ✓ Saved layer aggregation to {args.layers_output}")

    # Save layer-level aggregation (new format for visualization)
    if args.layer_level_output:
        print(f"\n[STEP 3.3/4] Saving layer-level summary to {args.layer_level_output}...")
        layer_level_data = aggregate_by_layer_and_level(result_list)
        with open(args.layer_level_output, 'w') as f:
            json.dump(layer_level_data, f, indent=2)
        print(f"  ✓ Saved {len(layer_level_data)} layers with level breakdowns")

    print(f"  ✓ Saved {len(result_list)} aggregated concept entries")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    valid_concept_scores = [e["max_avg_concept_score"] for e in result_list if e["max_avg_concept_score"] is not None]
    valid_final_scores = [e["max_avg_final_score"] for e in result_list if e["max_avg_final_score"] is not None]
    valid_fluency_scores = [e["max_avg_fluency_score"] for e in result_list if e["max_avg_fluency_score"] is not None]

    if valid_concept_scores:
        print(f"\nMax Average Concept Scores:")
        print(f"  Mean: {sum(valid_concept_scores) / len(valid_concept_scores):.3f}")
        print(f"  Max: {max(valid_concept_scores):.3f}")
        print(f"  Min: {min(valid_concept_scores):.3f}")

    if valid_final_scores:
        print(f"\nMax Average Final Scores:")
        print(f"  Mean: {sum(valid_final_scores) / len(valid_final_scores):.3f}")
        print(f"  Max: {max(valid_final_scores):.3f}")
        print(f"  Min: {min(valid_final_scores):.3f}")

    if valid_fluency_scores:
        print(f"\nMax Average Fluency Scores:")
        print(f"  Mean: {sum(valid_fluency_scores) / len(valid_fluency_scores):.3f}")
        print(f"  Max: {max(valid_fluency_scores):.3f}")
        print(f"  Min: {min(valid_fluency_scores):.3f}")

    print(f"\n{'='*80}")
    print("✓ DONE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()