from google import genai
from pathlib import Path
import argparse
from datetime import datetime
import time
from project_paths import DEFAULT_PROMPTS, PROJECT_ROOT, PRIMARY_DATA


MODEL = "gemini-3.1-flash-lite"
N_RUNS = 25
DELAY_BETWEEN_RUNS_SECONDS = 31


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect a new He/She/They response dataset.')
    parser.add_argument('--runs', type=int, default=N_RUNS)
    parser.add_argument('--output', type=Path, default=None,
                        help='New text file; existing files are never overwritten.')
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be at least 1')
    output_file = args.output or PROJECT_ROOT / 'data/collections' / f'responses_{datetime.now():%Y%m%d_%H%M%S_%f}.txt'
    if output_file.resolve() == PRIMARY_DATA.resolve():
        parser.error('Choose a new collection file; data/main/responses.txt is the saved study dataset.')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('x', encoding='utf-8') as output:
        output.write(f'Gemini prompts and responses\nModel: {MODEL}\n\n')
    client = genai.Client()
    prompts = DEFAULT_PROMPTS
    print(f"Saving prompts and responses to {output_file}")

    for run in range(1, args.runs + 1):
        print(f"\n--- Run {run}/{args.runs} ---")

        for number, prompt in enumerate(prompts, start=1):
            try:
                response = client.interactions.create(model=MODEL, input=prompt)
                text = response.output_text.strip()
            except Exception as exc:
                raise RuntimeError(
                    f"Gemini request {number} in run {run} failed: {exc}"
                ) from exc

            print(f"[{number}] {text}")
            with output_file.open("a", encoding="utf-8") as output:
                output.write(f"--- Run {run}, Prompt {number} ---\n")
                output.write(f"PROMPT:\n{prompt}\n\n")
                output.write(f"RESPONSE:\n{text}\n\n")

        if run < args.runs:
            print(f"Waiting {DELAY_BETWEEN_RUNS_SECONDS} seconds before the next run...")
            time.sleep(DELAY_BETWEEN_RUNS_SECONDS)


if __name__ == "__main__":
    main()
