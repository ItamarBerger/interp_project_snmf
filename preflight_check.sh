#!/bin/bash
echo "========================================="
echo "Pre-flight Check for SNMF Steering Pipeline"
echo "========================================="
echo ""

errors=0
warnings=0

# Check 1: Results directory writable
echo "[1/8] Checking experiments/results/ is writable..."
if [ -w experiments/results/ ]; then
  echo "  ✓ PASS: Directory is writable"
else
  echo "  ✗ FAIL: Directory is not writable"
  ((errors++))
fi
echo ""

# Check 2: Test file creation
echo "[2/8] Testing file creation in results/..."
if echo "test" > experiments/results/.test 2>/dev/null && rm experiments/results/.test 2>/dev/null; then
  echo "  ✓ PASS: Can create and delete files"
else
  echo "  ✗ FAIL: Cannot create files"
  ((errors++))
fi
echo ""

# Check 3: NMF model readable
echo "[3/8] Checking NMF model in artifacts/..."
if [ -r experiments/artifacts/0/50/nmf-l0-r50.pkl ]; then
  echo "  ✓ PASS: Model file is readable ($(ls -lh experiments/artifacts/0/50/nmf-l0-r50.pkl | awk '{print $5}'))"
else
  echo "  ✗ FAIL: Model file not found or not readable"
  ((errors++))
fi
echo ""

# Check 4: Training data readable
echo "[4/8] Checking training data..."
if [ -r data/final_dataset_20_concepts.json ]; then
  lines=$(wc -l < data/final_dataset_20_concepts.json)
  echo "  ✓ PASS: Data file readable ($lines lines)"
else
  echo "  ✗ FAIL: Data file not found or not readable"
  ((errors++))
fi
echo ""

# Check 5: API key configured
echo "[5/8] Checking OPENAI_API_KEY setup..."
if [ -f "$HOME/.openai.env" ]; then
  source "$HOME/.openai.env"
  if [ -n "$OPENAI_API_KEY" ]; then
    echo "  ✓ PASS: API key is configured (${#OPENAI_API_KEY} chars)"
  else
    echo "  ✗ FAIL: ~/.openai.env exists but OPENAI_API_KEY not set"
    ((errors++))
  fi
else
  echo "  ✗ FAIL: ~/.openai.env does not exist"
  ((errors++))
fi
echo ""

# Check 6: Python environment
echo "[6/8] Checking Python environment..."
if command -v python &> /dev/null; then
  echo "  ✓ PASS: Python is available"
else
  echo "  ⚠ WARNING: Python not in PATH (OK if running via Slurm with conda)"
  ((warnings++))
fi
echo ""

# Check 7: Script syntax
echo "[7/8] Checking script syntax..."
if bash -n experiments/run_snmf_steering.sh 2>/dev/null; then
  echo "  ✓ PASS: Script syntax is valid"
else
  echo "  ✗ FAIL: Script has syntax errors"
  ((errors++))
fi
echo ""

# Check 8: Path consistency
echo "[8/8] Checking path configuration..."
reads_from_artifacts=$(grep -c "experiments/artifacts" experiments/run_snmf_steering.sh || true)
writes_to_results=$(grep -c "experiments/results" experiments/run_snmf_steering.sh || true)
echo "  • Reads from artifacts: $reads_from_artifacts locations"
echo "  • Writes to results: $writes_to_results locations"
if [ "$writes_to_results" -ge 5 ]; then
  echo "  ✓ PASS: Outputs configured for results directory"
else
  echo "  ⚠ WARNING: Expected at least 5 output paths to results"
  ((warnings++))
fi
echo ""

# Summary
echo "========================================="
echo "Summary:"
echo "  Errors: $errors"
echo "  Warnings: $warnings"
echo "========================================="
echo ""

if [ $errors -eq 0 ]; then
  echo "✓ ALL CHECKS PASSED - Safe to run sbatch!"
  exit 0
else
  echo "✗ $errors CRITICAL ISSUE(S) FOUND - DO NOT RUN"
  exit 1
fi
