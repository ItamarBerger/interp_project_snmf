import asyncio
import json
import os
import re
import argparse
from typing import List, Any
import time
# from openai import AsyncOpenAI
# from openai.types.chat import ChatCompletion
import google.generativeai as genai

from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv

# ---------------- Prompts ---------------- #
CONCEPT_PROMPT = """
You are given a set of tokens, their surrounding context (words before and after the token), and an importance score.
Your task is to identify a brief, direct concept label that describes the connection between these tokens.

### Instructions:

1. **Focus on High-Importance Samples:**
   Examine only the token-context pairs with the highest importance scores. If a significant drop is observed beyond a threshold, ignore the lower-scoring pairs.

2. **Prioritize Tokens First:**
   - First look for patterns in the tokens themselves (identical tokens, related words, grammatical features, suffixes/prefixes, semantic patterns, etc.)
   - Only examine contexts if the tokens themselves show no clear relationship

3. **Output a Concise Label:**
   - Provide ONLY a brief concept label or short phrase
   - Do NOT start with phrases like "The tokens represent..." or "The concept is..."
   - If one specific token dominates, name it directly in quotes
   - Keep it as short as possible while being meaningful

### Input:

Token-Context Pairs:
```{token_context_str}```

Output only the concept label:
"""
# ---------------- Helpers ---------------- #
def extract_results_section(text: str) -> str | None:
    # Since we no longer have a "Results:" section, just return the text directly
    return text.strip() if text else None
def load_data(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def save_data(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _parse_int_list(csv: str) -> List[int]:
    if not csv:
        return []
    return [int(x.strip()) for x in csv.split(",") if x.strip()]

def _to_float_activation(x) -> float:
    """Handle values like 0.123, string 0.123, or tensor(0.123) safely."""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    s = s.replace("tensor(", "").replace(")", "")
    try:
        return float(s)
    except ValueError:
        return float("nan")

# ---------------- Argparse ---------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(description="Generate concept descriptions from token-context activations.")
    # I/O
    p.add_argument("--input-json", default="rebuttal/init_methods/svd/concept_contexts.json",
                   help="Path to input JSON containing top_activations per (K, layer, level, h_row).")
    p.add_argument("--output-json", default="rebuttal/init_methods/svd/input_descriptions.json",
                   help="Path to write the output descriptions JSON.")
    # Gemini
    p.add_argument("--model", default="gemini-1.5-flash", help="Gemini model name.")
    p.add_argument("--env-var", default="GEMINI_API_KEY",
                   help="Environment variable holding the API key (loaded via python-dotenv if present).")
    # Filtering / selection
    p.add_argument("--layers", default="0,6,12,18,25,31",
                   help="Comma-separated list of layer indices to include.")
    p.add_argument("--k-values", default="100",
                   help="Comma-separated list of K (rank) values to include.")
    # Generation controls
    p.add_argument("--top-m", type=int, default=10, help="Number of top activations to consider.")
    p.add_argument("--max-tokens", type=int, default=200, help="max_tokens for each completion.")
    p.add_argument("--concurrency", type=int, default=25, help="Semaphore limit for concurrent calls.")
    p.add_argument("--retries", type=int, default=5, help="Tenacity stop_after_attempt.")
    return p

# ---------------- Async workers ---------------- #
def make_generate_concept(retries: int, model, max_tokens: int, semaphore: asyncio.Semaphore):
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(retries))
    async def _inner(entry, top_m: int):
        # sort and pick top-M by activation
        top = sorted(
            entry["top_activations"],
            key=lambda x: _to_float_activation(x["activation"]),
            reverse=True
        )[:top_m]

        token_context_str = "\n".join(
            f"Token: `{act['token']}`, Context: `{act['context']}` | Score: `{_to_float_activation(act['activation'])}`"
            for act in top
        )
        prompt = CONCEPT_PROMPT.format(token_context_str=token_context_str)

        async with semaphore:
            resp = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": 0.2}
            )
        return extract_results_section(resp.text.strip())
    return _inner

def make_process_entry(generate_concept):
    async def _inner(entry, top_m: int):
        concept_desc = await generate_concept(entry, top_m)
        print(".", end="", flush=True)
        return {
            "K": entry["K"],
            "layer": entry["layer"],
            "level": entry["level"], # hierarchical level
            "h_row": entry["h_row"],
            "description": concept_desc,
        }
    return _inner

# ---------------- Main ---------------- #
async def run(args):
    data = load_data(args.input_json)
    layers = set(_parse_int_list(args.layers))
    k_values = set(_parse_int_list(args.k_values))
    # each k corresponds to a specific level. Hence number of levels is len(k_values)
    n_levels = len(k_values)
    levels = [i for i in range(n_levels)]

    # Load API key
    load_dotenv()
    api_key = os.getenv(args.env_var)
    if not api_key:
        raise RuntimeError(f"API key not found in environment variable '{args.env_var}'.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    semaphore = asyncio.Semaphore(args.concurrency)
    # client = AsyncOpenAI(api_key=api_key)

    generate_concept = make_generate_concept(
        retries=args.retries, model=model, max_tokens=args.max_tokens, semaphore=semaphore
    )
    process_entry = make_process_entry(generate_concept)

    tasks = [
        process_entry(e, args.top_m)
        for e in data
        if (int(e["layer"]) in layers and int(e["K"]) in k_values and int(e["level"]) in levels)
    ]

    print(f"Running over {len(tasks)} tasks …")
    results = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)

    save_data(args.output_json, results)
    print(f"\nWrote {len(results)} descriptions to {args.output_json}")

def main():
    args = build_arg_parser().parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
