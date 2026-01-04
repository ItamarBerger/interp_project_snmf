import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict
import os


def load_aggregated_results(input_file: str) -> List[Dict]:
    """Load the aggregated results JSON"""
    with open(input_file, 'r') as f:
        return json.load(f)


def load_layer_level_summary(input_file: str) -> Dict:
    """Load the layer-level summary JSON"""
    with open(input_file, 'r') as f:
        return json.load(f)


def group_by_level(entries: List[Dict]) -> Dict[int, List[Dict]]:
    """Group entries by level"""
    level_groups = defaultdict(list)
    for entry in entries:
        level = entry.get("level", 0)
        level_groups[level].append(entry)
    return dict(level_groups)


def compute_level_statistics(level_groups: Dict[int, List[Dict]]) -> Dict[int, Dict]:
    """Compute statistics for each level"""
    stats = {}

    for level, entries in level_groups.items():
        # Extract scores
        concept_scores = [e["max_avg_concept_score"] for e in entries if e["max_avg_concept_score"] is not None]
        final_scores = [e["max_avg_final_score"] for e in entries if e["max_avg_final_score"] is not None]
        fluency_scores = [e["max_avg_fluency_score"] for e in entries if e["max_avg_fluency_score"] is not None]

        stats[level] = {
            "num_concepts": len(entries),
            "concept_scores": {
                "mean": np.mean(concept_scores) if concept_scores else 0,
                "std": np.std(concept_scores) if concept_scores else 0,
                "median": np.median(concept_scores) if concept_scores else 0,
                "min": np.min(concept_scores) if concept_scores else 0,
                "max": np.max(concept_scores) if concept_scores else 0,
                "count": len(concept_scores)
            },
            "final_scores": {
                "mean": np.mean(final_scores) if final_scores else 0,
                "std": np.std(final_scores) if final_scores else 0,
                "median": np.median(final_scores) if final_scores else 0,
                "min": np.min(final_scores) if final_scores else 0,
                "max": np.max(final_scores) if final_scores else 0,
                "count": len(final_scores)
            },
            "fluency_scores": {
                "mean": np.mean(fluency_scores) if fluency_scores else 0,
                "std": np.std(fluency_scores) if fluency_scores else 0,
                "median": np.median(fluency_scores) if fluency_scores else 0,
                "min": np.min(fluency_scores) if fluency_scores else 0,
                "max": np.max(fluency_scores) if fluency_scores else 0,
                "count": len(fluency_scores)
            }
        }

    return stats


def print_level_analysis(stats: Dict[int, Dict]):
    """Print detailed analysis for each level"""
    print("\n" + "="*80)
    print("LEVEL-BY-LEVEL ANALYSIS")
    print("="*80)

    sorted_levels = sorted(stats.keys())

    for level in sorted_levels:
        level_stats = stats[level]
        print(f"\n{'='*80}")
        print(f"LEVEL {level}")
        print(f"{'='*80}")
        print(f"Total Concepts: {level_stats['num_concepts']}")

        print(f"\n  Concept Scores (n={level_stats['concept_scores']['count']}):")
        print(f"    Mean:   {level_stats['concept_scores']['mean']:.4f}")
        print(f"    Median: {level_stats['concept_scores']['median']:.4f}")
        print(f"    Std:    {level_stats['concept_scores']['std']:.4f}")
        print(f"    Min:    {level_stats['concept_scores']['min']:.4f}")
        print(f"    Max:    {level_stats['concept_scores']['max']:.4f}")

        print(f"\n  Final Scores (n={level_stats['final_scores']['count']}):")
        print(f"    Mean:   {level_stats['final_scores']['mean']:.4f}")
        print(f"    Median: {level_stats['final_scores']['median']:.4f}")
        print(f"    Std:    {level_stats['final_scores']['std']:.4f}")
        print(f"    Min:    {level_stats['final_scores']['min']:.4f}")
        print(f"    Max:    {level_stats['final_scores']['max']:.4f}")

        print(f"\n  Fluency Scores (n={level_stats['fluency_scores']['count']}):")
        print(f"    Mean:   {level_stats['fluency_scores']['mean']:.4f}")
        print(f"    Median: {level_stats['fluency_scores']['median']:.4f}")
        print(f"    Std:    {level_stats['fluency_scores']['std']:.4f}")
        print(f"    Min:    {level_stats['fluency_scores']['min']:.4f}")
        print(f"    Max:    {level_stats['fluency_scores']['max']:.4f}")


def print_comparative_analysis(stats: Dict[int, Dict], entries: List[Dict], output_dir: str = "experiments/artifacts/"):
    """
    Write comparative analysis across levels and per-(level,layer) summaries to JSON files.

    - Writes the overall `level_comparative_analysis.json` (containing `stats`).
    - Writes one JSON file per (level, layer) in `output_dir`, named `level_{level}_layer_{layer}.json`.

    :param stats: Level-level aggregated statistics (used for overall summary)
    :param entries: Original entries list (used to compute per-(level,layer) summaries)
    :param output_dir: Directory where per-(level,layer) files will be written
    """
    os.makedirs(output_dir, exist_ok=True)

    # Write overall stats
    overall_path = os.path.join(output_dir, 'level_comparative_analysis.json')
    with open(overall_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Build per-(level, layer) aggregates
    lvl_lyr = defaultdict(lambda: defaultdict(list))
    for e in entries:
        level = e.get('level', 0)
        layer = e.get('layer', 'none')
        key = (level, layer)
        lvl_lyr[key]['concept_scores'].append(e.get('max_avg_concept_score'))
        lvl_lyr[key]['final_scores'].append(e.get('max_avg_final_score'))
        lvl_lyr[key]['fluency_scores'].append(e.get('max_avg_fluency_score'))

    index = {}
    for (level, layer), measures in lvl_lyr.items():
        # Filter None values
        concept_scores = [v for v in measures['concept_scores'] if v is not None]
        final_scores = [v for v in measures['final_scores'] if v is not None]
        fluency_scores = [v for v in measures['fluency_scores'] if v is not None]

        summary = {
            'level': level,
            'layer': layer,
            'num_concepts': len(concept_scores) if concept_scores else 0,
            'concept_scores': {
                'mean': float(np.mean(concept_scores)) if concept_scores else 0.0,
                'std': float(np.std(concept_scores)) if concept_scores else 0.0,
                'min': float(np.min(concept_scores)) if concept_scores else 0.0,
                'max': float(np.max(concept_scores)) if concept_scores else 0.0,
                'count': len(concept_scores)
            },
            'final_scores': {
                'mean': float(np.mean(final_scores)) if final_scores else 0.0,
                'std': float(np.std(final_scores)) if final_scores else 0.0,
                'min': float(np.min(final_scores)) if final_scores else 0.0,
                'max': float(np.max(final_scores)) if final_scores else 0.0,
                'count': len(final_scores)
            },
            'fluency_scores': {
                'mean': float(np.mean(fluency_scores)) if fluency_scores else 0.0,
                'std': float(np.std(fluency_scores)) if fluency_scores else 0.0,
                'min': float(np.min(fluency_scores)) if fluency_scores else 0.0,
                'max': float(np.max(fluency_scores)) if fluency_scores else 0.0,
                'count': len(fluency_scores)
            }
        }

        safe_layer = str(layer).replace('/', '_').replace('\\', '_')
        fname = f'level_{level}_layer_{safe_layer}.json'
        fpath = os.path.join(output_dir, fname)
        with open(fpath, 'w') as f:
            json.dump(summary, f, indent=2)

        index_key = f"{level}:{layer}"
        index[index_key] = fpath

    # Write an index file mapping (level:layer) -> file path
    index_path = os.path.join(output_dir, 'level_layer_index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    # Print a concise confirmation message
    print(f"  ✓ Wrote overall comparative stats to: {overall_path}")
    print(f"  ✓ Wrote {len(index)} per-(level,layer) summary files to: {output_dir}")


def create_layer_level_visualizations(layer_level_data: Dict, output_dir: str):
    """
    Create visualizations for each layer comparing its levels.
    Each layer gets its own figure with subplots for the three score types.
    
    Args:
        layer_level_data: Dict with structure {layer: {"layer": layer, "levels": {...}}}
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    print(f"\n  Creating per-layer visualizations...")
    
    for layer_key, layer_info in sorted(layer_level_data.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
        layer = layer_info["layer"]
        levels_dict = layer_info["levels"]
        
        # Sort levels
        sorted_levels = sorted([int(k) for k in levels_dict.keys()])
        
        if len(sorted_levels) == 0:
            print(f"    ⚠ Layer {layer}: No levels found, skipping...")
            continue
        
        # Extract data for this layer
        level_labels = [f"Level {level}" for level in sorted_levels]
        
        concept_scores = []
        final_scores = []
        fluency_scores = []
        
        for level in sorted_levels:
            level_data = levels_dict[str(level)]
            concept_scores.append(level_data.get("concept_score", 0) or 0)
            final_scores.append(level_data.get("final_score", 0) or 0)
            fluency_scores.append(level_data.get("fluency_score", 0) or 0)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_levels)))
        
        # Subplot 1: Concept Scores
        bars1 = axes[0].bar(range(len(sorted_levels)), concept_scores,
                           color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[0].set_xlabel('Level', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Mean Concept Score', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Concept Scores', fontsize=14, fontweight='bold')
        axes[0].set_xticks(range(len(sorted_levels)))
        axes[0].set_xticklabels([str(l) for l in sorted_levels], fontsize=11)
        axes[0].set_ylim(0, max(2, max(concept_scores) * 1.2 if concept_scores else 2))
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars1, concept_scores):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Subplot 2: Final Scores
        bars2 = axes[1].bar(range(len(sorted_levels)), final_scores,
                           color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[1].set_xlabel('Level', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Mean Final Score', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Final Scores', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(len(sorted_levels)))
        axes[1].set_xticklabels([str(l) for l in sorted_levels], fontsize=11)
        axes[1].set_ylim(0, max(2, max(final_scores) * 1.2 if final_scores else 2))
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars2, final_scores):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Subplot 3: Fluency Scores
        bars3 = axes[2].bar(range(len(sorted_levels)), fluency_scores,
                           color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[2].set_xlabel('Level', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Mean Fluency Score', fontsize=12, fontweight='bold')
        axes[2].set_title(f'Fluency Scores', fontsize=14, fontweight='bold')
        axes[2].set_xticks(range(len(sorted_levels)))
        axes[2].set_xticklabels([str(l) for l in sorted_levels], fontsize=11)
        axes[2].set_ylim(0, max(2, max(fluency_scores) * 1.2 if fluency_scores else 2))
        axes[2].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars3, fluency_scores):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Overall title
        plt.suptitle(f'Layer {layer}: Score Comparison Across Levels', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'layer_{layer}_levels_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Layer {layer}: Saved {output_path}")
        plt.close()


def create_visualizations(stats: Dict[int, Dict], output_dir: str):
    """Create matplotlib visualizations comparing levels"""
    # save figures by their corresponding layer (output_dir/layer_{layer}.png)

    os.makedirs(output_dir, exist_ok=True)

    sorted_levels = sorted(stats.keys())

    # Set up style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_levels)))

    # Figure 1: Concept Scores
    fig, ax = plt.subplots(figsize=(10, 6))

    means = [stats[level]['concept_scores']['mean'] for level in sorted_levels]
    stds = [stats[level]['concept_scores']['std'] for level in sorted_levels]

    bars = ax.bar(range(len(sorted_levels)), means, yerr=stds,
                   color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Concept Score', fontsize=14, fontweight='bold')
    ax.set_title('Max Average Concept Scores by Level', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(sorted_levels)))
    ax.set_xticklabels([f'Level {l}' for l in sorted_levels], fontsize=12)
    ax.set_ylim(0, 2)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_scores_by_level.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {os.path.join(output_dir, 'concept_scores_by_level.png')}")
    plt.close()

    # Figure 2: Final Scores
    fig, ax = plt.subplots(figsize=(10, 6))

    means = [stats[level]['final_scores']['mean'] for level in sorted_levels]
    stds = [stats[level]['final_scores']['std'] for level in sorted_levels]

    bars = ax.bar(range(len(sorted_levels)), means, yerr=stds,
                   color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Final Score', fontsize=14, fontweight='bold')
    ax.set_title('Max Average Final Scores by Level', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(sorted_levels)))
    ax.set_xticklabels([f'Level {l}' for l in sorted_levels], fontsize=12)
    ax.set_ylim(0, 2)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_scores_by_level.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {os.path.join(output_dir, 'final_scores_by_level.png')}")
    plt.close()

    # Figure 3: Fluency Scores
    fig, ax = plt.subplots(figsize=(10, 6))

    means = [stats[level]['fluency_scores']['mean'] for level in sorted_levels]
    stds = [stats[level]['fluency_scores']['std'] for level in sorted_levels]

    bars = ax.bar(range(len(sorted_levels)), means, yerr=stds,
                   color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Fluency Score', fontsize=14, fontweight='bold')
    ax.set_title('Max Average Fluency Scores by Level', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(sorted_levels)))
    ax.set_xticklabels([f'Level {l}' for l in sorted_levels], fontsize=12)
    ax.set_ylim(0, 2)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fluency_scores_by_level.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {os.path.join(output_dir, 'fluency_scores_by_level.png')}")
    plt.close()

    # Figure 4: Combined comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    score_types = ['concept_scores', 'final_scores', 'fluency_scores']
    titles = ['Concept Scores', 'Final Scores', 'Fluency Scores']

    for idx, (score_type, title) in enumerate(zip(score_types, titles)):
        means = [stats[level][score_type]['mean'] for level in sorted_levels]
        stds = [stats[level][score_type]['std'] for level in sorted_levels]

        bars = axes[idx].bar(range(len(sorted_levels)), means, yerr=stds,
                            color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)

        axes[idx].set_xlabel('Level', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(f'Mean {title}', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{title}', fontsize=14, fontweight='bold')
        axes[idx].set_xticks(range(len(sorted_levels)))
        axes[idx].set_xticklabels([f'{l}' for l in sorted_levels], fontsize=11)
        axes[idx].set_ylim(0, 2)
        axes[idx].grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                          f'{mean:.2f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Comparison of All Score Types Across Levels', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_scores_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {os.path.join(output_dir, 'all_scores_comparison.png')}")
    plt.close()


def analyze_by_layer(entries: List[Dict]) -> Dict:
    """Additional analysis: performance by layer"""
    layer_groups = defaultdict(list)

    for entry in entries:
        layer = entry.get("layer")
        if layer is not None:
            layer_groups[layer].append(entry)

    layer_stats = {}
    for layer, layer_entries in layer_groups.items():
        final_scores = [e["max_avg_final_score"] for e in layer_entries if e["max_avg_final_score"] is not None]
        layer_stats[layer] = {
            "count": len(layer_entries),
            "mean_final_score": np.mean(final_scores) if final_scores else 0,
            "max_final_score": np.max(final_scores) if final_scores else 0
        }

    return layer_stats


def print_layer_analysis(layer_stats: Dict):
    """Print analysis by layer"""
    print("\n" + "="*80)
    print("ANALYSIS BY LAYER")
    print("="*80)

    sorted_layers = sorted(layer_stats.keys())

    print(f"\n{'Layer':<10} {'Count':<10} {'Mean Final Score':<20} {'Max Final Score':<20}")
    print("-"*80)

    for layer in sorted_layers:
        stats = layer_stats[layer]
        print(f"{layer:<10} {stats['count']:<10} {stats['mean_final_score']:<20.4f} {stats['max_final_score']:<20.4f}")

    best_layer = max(sorted_layers, key=lambda l: layer_stats[l]['mean_final_score'])
    print(f"\nBest performing layer: {best_layer} (mean final score = {layer_stats[best_layer]['mean_final_score']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze aggregated max scores with focus on layer-level comparison"
    )
    parser.add_argument(
        "--layer-level-input",
        required=True,
        help="Path to layer-level summary JSON (output from aggregate_causal_results.py --layer-level-output)"
    )
    parser.add_argument(
        "--concepts-input",
        required=False,
        default=None,
        help="Path to aggregated concepts JSON (optional, for additional analyses)"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/artifacts/analysis_results",
        help="Directory to save analysis figures (default: experiments/artifacts/analysis_results)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ANALYZING AGGREGATED MAX SCORES BY LAYER AND LEVEL")
    print("="*80)

    # Load layer-level data
    print(f"\n[STEP 1/2] Loading layer-level summary from {args.layer_level_input}...")
    layer_level_data = load_layer_level_summary(args.layer_level_input)
    print(f"  ✓ Loaded {len(layer_level_data)} layers")
    
    for layer_key, layer_info in sorted(layer_level_data.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
        num_levels = len(layer_info.get("levels", {}))
        print(f"    Layer {layer_info['layer']}: {num_levels} levels")

    # Create per-layer visualizations
    print(f"\n[STEP 2/2] Creating per-layer visualizations...")
    create_layer_level_visualizations(layer_level_data, args.output_dir)

    # Optional: Load concepts data for additional analysis
    if args.concepts_input:
        print(f"\n[ADDITIONAL] Loading concept-level data from {args.concepts_input}...")
        entries = load_aggregated_results(args.concepts_input)
        print(f"  ✓ Loaded {len(entries)} concepts")
        
        # Group by level for cross-layer analysis
        print(f"\n[ADDITIONAL] Creating cross-layer level analysis...")
        level_groups = group_by_level(entries)
        print(f"  ✓ Found {len(level_groups)} levels across all layers")
        
        # Compute statistics
        stats = compute_level_statistics(level_groups)
        
        # Print analyses
        print_level_analysis(stats)
        print_comparative_analysis(stats, entries, args.output_dir)
        
        # Additional analysis by layer
        layer_stats = analyze_by_layer(entries)
        print_layer_analysis(layer_stats)
        
        # Create cross-layer visualizations
        print(f"\n[ADDITIONAL] Creating cross-layer visualizations...")
        create_visualizations(stats, args.output_dir)

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nFigures saved to: {args.output_dir}")
    print(f"  - Per-layer comparison figures: layer_<N>_levels_comparison.png")
    if args.concepts_input:
        print(f"  - Cross-layer analysis figures:")
        print(f"    * concept_scores_by_level.png")
        print(f"    * final_scores_by_level.png")
        print(f"    * fluency_scores_by_level.png")
        print(f"    * all_scores_comparison.png")
    print("\n")


if __name__ == "__main__":
    main()