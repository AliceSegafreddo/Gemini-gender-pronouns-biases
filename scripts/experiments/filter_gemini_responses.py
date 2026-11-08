"""Create a balanced prompt/response file containing selected subjects only."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


RECORD_PATTERN = re.compile(
    r"(?ms)^--- Run (\d+), Prompt (\d+) ---\s*\n"
    r"PROMPT:\s*\n(.*?)\n\s*RESPONSE:\s*\n(.*?)(?=\n--- Run |\Z)"
)
SUBJECT_PATTERN = re.compile(
    r"^\s*(.+?)\s+exaggerat(?:e|es|ed|ing)\b", re.IGNORECASE
)


def subject(prompt: str) -> str:
    match = SUBJECT_PATTERN.search(prompt)
    return match.group(1).strip(" .,!?:;").lower() if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument(
        "--subjects", nargs="+", default=["he", "she", "they"],
        help="Subjects to retain (default: he she they).",
    )
    args = parser.parse_args()
    if args.output_file.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_file}; choose a new output file.')

    text = args.input_file.read_text(encoding="utf-8")
    matches = RECORD_PATTERN.findall(text)
    if not matches:
        raise ValueError(f"No records found in {args.input_file}.")

    selected = []
    runs_by_subject: dict[str, set[int]] = defaultdict(set)
    wanted = {value.lower() for value in args.subjects}
    for run_text, prompt_number_text, prompt, response in matches:
        run = int(run_text)
        prompt_number = int(prompt_number_text)
        current_subject = subject(prompt)
        if current_subject not in wanted:
            continue
        selected.append((run, prompt_number, prompt.strip(), response.strip()))
        runs_by_subject[current_subject].add(run)

    missing = wanted - set(runs_by_subject)
    if missing:
        raise ValueError(f"Subjects not found: {', '.join(sorted(missing))}")
    run_counts = {key: len(value) for key, value in runs_by_subject.items()}
    if len(set(run_counts.values())) != 1:
        raise ValueError(f"Selected subjects have different run counts: {run_counts}")

    selected.sort(key=lambda record: (record[0], record[1]))
    lines = [
        "Gemini prompts and responses",
        f"Filtered subjects: {', '.join(args.subjects)}",
        f"Runs per subject: {next(iter(run_counts.values()))}",
        "",
    ]
    for run, prompt_number, prompt, response in selected:
        lines.extend([
            f"--- Run {run}, Prompt {prompt_number} ---",
            "PROMPT:",
            prompt,
            "",
            "RESPONSE:",
            response,
            "",
        ])

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Wrote {len(selected)} records ({next(iter(run_counts.values()))} runs per subject) "
        f"to {args.output_file}"
    )


if __name__ == "__main__":
    main()
