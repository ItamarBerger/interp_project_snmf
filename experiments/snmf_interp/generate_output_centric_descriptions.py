import os
import json
import re
import asyncio
import argparse
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from experiments.evaluation.json_handler import JsonHandler
import logging
from experiments.utils import retry_with_attempts, RateLimiter, batched

from utils import setup_logging

logger = logging.getLogger(__name__)

# ── utils ─────────────────────────────────────────────────────────────────────
def parse_int_list(spec: Optional[str]) -> Optional[List[int]]:
    """
    Parse '0,1,2' or '0-3' or '0,2,5-7' into a list of ints.
    Returns None if spec is None or empty.
    """
    if not spec:
        return None
    out: List[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))
        else:
            out.append(int(part))
    return out

def extract_results_section(text: str) -> Optional[str]:
    m = re.search(r"Results:\s*(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else None

def load_data(path: str) -> list[dict]:
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path: str, data) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

# ── core ──────────────────────────────────────────────────────────────────────
async def run(args):
    # .env
    load_dotenv()  # discovers OPENAI_API_KEY and optional defaults

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment (.env).")

    input_json  = args.input  
    output_json = args.output
    model       = args.model  or 'gemini-2.0-flash'
    top_m       = args.top_m
    concurrency = args.concurrency 
    max_tokens  = args.max_tokens

    layers = parse_int_list(args.layers) 
    ranks  = parse_int_list(args.ranks)

    # client & semaphore
    # client = AsyncOpenAI(api_key=api_key)
    client = genai.Client(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    rate_limiter = RateLimiter(max_requests=1900, window_seconds=60)

    CONNECTION_PROMPT = """
You are given a set of tokens and their importance score.
Your task is to determine what is the connection between all the tokens.

### Instructions:

1. **Focus on High-Importance Samples:**  
   Examine only the token pairs-score with the highest importance scores. If a significant drop is observed beyond a threshold, ignore the lower-scoring tokens.

2. Always choose the simplest and most obvious underlying connection. 
   
3. Ignore noisy tokens. Some tokens may be unrelated to the concept, only consider tokens that share the most obvious connection.

4. If there is no concept or underlying connection between the tokens output under the Results section: TRASH

### Output Format:

Analysis:
<reason about what the underlying connection is.>

Results:
- Connection: <Single sentence description of the single most obvious connection between the tokens>

### Token-Score Pairs:
```{token_context_str}```

Do not add any markdown to the examples.
Make sure you output a very precise and detailed description of the concept that fully captures it.
""".strip()

    @retry_with_attempts(args.retries)
    async def _call_gemini(prompt: str,  max_tokens: int):
        await rate_limiter.acquire()
        async with semaphore:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=prompt,
                config=GenerateContentConfig(max_output_tokens=max_tokens, temperature=0.2)
            )
        return resp

    async def generate_concept(entry: dict) -> dict:
        # pair tokens & scores, sort, take top M
        pairs = list(zip(entry['top_shifted_tokens'], entry['top_logit_values']))
        top = sorted(pairs, key=lambda x: float(x[1]), reverse=True)[:top_m]
        token_context_str = "\n".join(
            f"Token: {tok[0][0]} | Importance Score: {scr}" for tok, scr in top
        )

        prompt = CONNECTION_PROMPT.format(token_context_str=token_context_str)
        logger.info(f"[→] Generating for K={entry.get('K', 'SAE')} layer={entry['layer']} row={entry['h_row']} level={entry.get('level', 0)}…")

        resp = await _call_gemini(prompt, max_tokens)
        content = resp.text.strip()
        result  = extract_results_section(content) or "ERROR: no Results section"
        logger.info(f"[✔] Done K={entry.get('K', 'SAE')} layer={entry['layer']} level = {entry.get('level', 0)}")
        return {
            'description': result,
            'layer': entry['layer'],
            'level' : entry.get('level', 0),
            'K': entry.get('K', 'SAE'),
            'h_row': entry['h_row'],
            'sign': entry.get('intervention_sign')
        }

    # filter data
    levels = [i for i in range(len(ranks))] if ranks else [0]
    data = load_data(input_json)
    filtered = [
        e for e in data
        if int(e['layer']) in layers and ('K' not in e or not ranks or int(e['K']) in ranks) and int(e['level']) in levels
    ]


    logger.info(f"Processing {len(filtered)} entries…")
    tasks = [generate_concept(e) for e in filtered]

    results = []
    batches = batched(tasks, args.batch_size)

    for batch_idx, batch in enumerate(batches):
        logger.info(f"Starting batch {batch_idx + 1}/{(len(tasks) + args.batch_size - 1) // args.batch_size} with {len(batch)} tasks.")
        for coro in asyncio.as_completed(batch):
            try:
                res = await coro
                results.append(res)
                logger.info(f"  → Completed {len(results)}/{len(filtered)}")
            except Exception as err:
                logger.info(f"  ⚠ Sample exception: {err}")


    json_handler = JsonHandler(
        ["description", "layer", "level", "h_row", "K", "sign"],
        output_json,
        auto_write=False
    )
    for row in results:
        json_handler.add_row(**row)
    json_handler.write()

    bad  = sum(('TRASH' in (r['description'] or "")) for r in results)
    good = len(results) - bad
    logger.info(f"Good: {good}\tBad: {bad}")

# ── entrypoint ────────────────────────────────────────────────────────────────
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate concept descriptions from token-score pairs using OpenAI.")
    # IO & model
    p.add_argument("--input", "-i", type=str, help="Path to input JSON (overrides INPUT_JSON env).")
    p.add_argument("--output", "-o", type=str, help="Path to output JSON (overrides OUTPUT_JSON env).")
    p.add_argument("--model", "-m", type=str, help="Model name (overrides MODEL env).")
    # Selection / behavior
    p.add_argument("--layers", type=str, help="Layers filter like '23,31' or '0-3,6'. Default env/23,31.")
    p.add_argument("--ranks",  type=str, help="Ranks (K) filter like '50,100' or '50-200'. If omitted, no K filter.")
    p.add_argument("--top-m", type=int, help="Top-M tokens to consider per entry (overrides TOP_M env).")
    p.add_argument("--concurrency", type=int, help="Max concurrent API calls (overrides CONCURRENCY env).", default=25)
    p.add_argument("--max-tokens", type=int, default=5000, help="max_tokens for the completion (default 5000).")
    p.add_argument("--retries", "-r", type=int, default=5, help="Number of retries for API calls (default 5).")
    p.add_argument("--batch-size", type=int, default=20, help="Number of tasks to batch together (default 20).")
    return p

def main():
    setup_logging()
    args = build_argparser().parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
