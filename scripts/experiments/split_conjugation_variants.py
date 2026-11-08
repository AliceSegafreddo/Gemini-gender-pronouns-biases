"""Split a saved Gemini file into correct and altered conjugation variants."""

from __future__ import annotations

import re
import argparse
from pathlib import Path
from project_paths import EXPERIMENT_DATA


RECORD_PATTERN = re.compile(
    r"(?ms)^--- Run (\d+), Prompt (\d+) ---\s*\n"
    r"PROMPT:\s*\n(.*?)\n\s*RESPONSE:\s*\n(.*?)(?=\n--- Run |\Z)"
)
VARIANT_PATTERN = re.compile(
    r"^\s*(he|she|they)\s+(exaggerates|exaggerate)\b", re.IGNORECASE
)


def write_variant(output: Path, records: list[tuple[int, int, str, str]], label: str) -> None:
    records.sort(key=lambda record: (record[0], record[1]))
    lines = [
        "Gemini prompts and responses",
        f"Conjugation variant: {label}",
        "",
    ]
    for run, number, prompt, response in records:
        lines.extend([
            f"--- Run {run}, Prompt {number} ---",
            "PROMPT:", prompt, "",
            "RESPONSE:", response, "",
        ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', type=Path, default=EXPERIMENT_DATA / 'conjugation_comparison/responses.txt')
    parser.add_argument('--output-dir', type=Path, default=EXPERIMENT_DATA / 'rebuilt_conjugation_variants')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Choose a new --output-dir; {args.output_dir} already exists.')
    input_file = args.input_file
    text = input_file.read_text(encoding="utf-8")
    matches = RECORD_PATTERN.findall(text)
    if not matches:
        raise ValueError(f"No records found in {input_file}")

    correct = []
    altered = []
    he_she_correct_they_altered = []
    he_she_altered_they_correct = []
    for run_text, number_text, prompt, response in matches:
        match = VARIANT_PATTERN.match(prompt.strip())
        if not match:
            raise ValueError(f"Could not classify prompt: {prompt!r}")
        subject, verb = match.group(1).lower(), match.group(2).lower()
        record = (int(run_text), int(number_text), prompt.strip(), response.strip())
        if (subject in {"he", "she"} and verb == "exaggerates") or (
            subject == "they" and verb == "exaggerate"
        ):
            correct.append(record)
        else:
            altered.append(record)

        if (subject in {"he", "she"} and verb == "exaggerates") or (
            subject == "they" and verb == "exaggerates"
        ):
            he_she_correct_they_altered.append(record)
        else:
            he_she_altered_they_correct.append(record)

    args.output_dir.mkdir(parents=True)
    write_variant(
        args.output_dir / 'correct.txt', correct, "he/she exaggerates; they exaggerate"
    )
    write_variant(
        args.output_dir / 'altered.txt', altered, "he/she exaggerate; they exaggerates"
    )
    write_variant(
        args.output_dir / 'he_she_correct_they_altered.txt',
        he_she_correct_they_altered,
        "he/she exaggerates; they exaggerates",
    )
    write_variant(
        args.output_dir / 'he_she_altered_they_correct.txt',
        he_she_altered_they_correct,
        "he/she exaggerate; they exaggerate",
    )
    print(f"Correct-conjugation records: {len(correct)}")
    print(f"Altered-conjugation records: {len(altered)}")
    print(f"He/she correct, they altered: {len(he_she_correct_they_altered)}")
    print(f"He/she altered, they correct: {len(he_she_altered_they_correct)}")


if __name__ == "__main__":
    main()
