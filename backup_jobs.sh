#!/bin/bash

set -euo pipefail


TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

echo "$TIMESTAMP Starting backup jobs..."

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="$2"
      shift 2
      ;;
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --raw)
      RAW=1
      shift 1
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${DEST:-}" ]]; then
  echo "Error: --dest argument is required." >&2
  exit 1
fi

if [[ -z "${SOURCE:-}" ]]; then
  echo "Error: --source argument is required." >&2
  exit 1
fi


full_path=$(realpath $0)
echo "$TIMESTAMP Script location is $full_path"
script_folder=$(dirname "$full_path")
echo "$TIMESTAMP cd to script folder $script_folder"
cd "$script_folder"

if [[ -d ".venv" ]]; then
  echo "$TIMESTAMP Loading env from .venv"
  source .venv/bin/activate
fi

echo "$TIMESTAMP Running script"
script_cmd="PYTHONPATH=. python download_and_parse_gemini_batches.py --submitted-jobs-file $SOURCE --backup-folder $DEST"
if [[ "${RAW:-0}" -eq 1 ]]; then
  echo "$TIMESTAMP Running in raw mode, will not parse the downloaded batches"
  script_cmd="$script_cmd --raw"
fi

echo "$TIMESTAMP Running command: $script_cmd"
eval "$script_cmd"

