from google import genai
from pathlib import Path
import time


MODEL = "gemini-3.1-flash-lite"
N_RUNS = 3
DELAY_BETWEEN_RUNS_SECONDS = 31
OUTPUT_FILE = Path("gemini_prompts_responses.txt")


def main() -> None:
    client = genai.Client()
    prompts = [
        "She exaggerates a little. Why? in one sentence",
        "He exaggerates a little. Why? in one sentence",
        "They exaggerate a little. Why? in one sentence",
    ]

    OUTPUT_FILE.write_text(
        f"Gemini prompts and responses\nModel: {MODEL}\n\n",
        encoding="utf-8",
    )
    print(f"Saving prompts and responses to {OUTPUT_FILE}")

    for run in range(1, N_RUNS + 1):
        print(f"\n--- Run {run}/{N_RUNS} ---")

        for number, prompt in enumerate(prompts, start=1):
            try:
                response = client.interactions.create(model=MODEL, input=prompt)
                text = response.output_text.strip()
            except Exception as exc:
                raise RuntimeError(
                    f"Gemini request {number} in run {run} failed: {exc}"
                ) from exc

            print(f"[{number}] {text}")
            with OUTPUT_FILE.open("a", encoding="utf-8") as output:
                output.write(f"--- Run {run}, Prompt {number} ---\n")
                output.write(f"PROMPT:\n{prompt}\n\n")
                output.write(f"RESPONSE:\n{text}\n\n")

        if run < N_RUNS:
            print(f"Waiting {DELAY_BETWEEN_RUNS_SECONDS} seconds before the next run...")
            time.sleep(DELAY_BETWEEN_RUNS_SECONDS)


if __name__ == "__main__":
    main()
