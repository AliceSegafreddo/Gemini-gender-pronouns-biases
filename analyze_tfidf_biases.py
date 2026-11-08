"""TF-IDF analysis of Gemini answers grouped by prompt subject/pronoun.

Each subject/pronoun is treated as one document: all of its answers across
replicate runs are concatenated. Prompts themselves are never included.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from nltk.stem.snowball import SnowballStemmer
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from project_paths import PRIMARY_DATA, PRIMARY_RESULTS, EXPERIMENT_RESULTS


RECORD_PATTERN = re.compile(
    r"--- Run (\d+), Prompt (\d+) ---\s*"
    r"PROMPT:\s*(.*?)\s*"
    r"RESPONSE:\s*(.*?)(?=\n--- Run |\Z)",
    re.DOTALL,
)
# Match the verb in the prompt regardless of subject agreement or tense,
# e.g. "he exaggerates" and "they exaggerate".
SUBJECT_PATTERN = re.compile(
    r"^\s*(.+?)\s+exaggerat(?:e|es|ed|ing)\b", re.IGNORECASE
)
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
STEMMER = SnowballStemmer("english")

# This list is available as an optional preprocessing choice. Words are kept
# by default because quantifiers and pronouns can be meaningful bias signals.
STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and",
    "are", "as", "at", "be", "because", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "of", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
}
PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
    "ourselves", "you", "your", "yours", "yourself", "yourselves", "he",
    "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "who", "whom",
    "whose", "which", "that", "this", "these", "those",
}
# Grammatical words are excluded by default. Quantifiers such as "more",
# "most", "some", and "all" are intentionally not included here.
ALWAYS_IGNORED_WORDS = {
    # Articles
    "a", "an", "the",
    # Prepositions
    "about", "above", "across", "after", "against", "along", "among",
    "around", "at", "before", "behind", "below", "beneath", "beside",
    "between", "beyond", "by", "despite", "down", "during", "except",
    "for", "from", "in", "inside", "into", "near", "of", "off", "on",
    "onto", "out", "over", "past", "since", "through", "throughout",
    "till", "to", "toward", "towards", "under", "until", "up", "upon",
    "with", "within", "without",
    # Coordinating and subordinating conjunctions
    "and", "nor", "or", "but", "yet", "so", "as", "although", "because",
    "before", "if", "once", "since", "than", "though", "unless", "when",
    "whenever", "whereas", "wherever", "whether", "while",
}


def extract_subject(prompt: str, prompt_number: int) -> str:
    match = SUBJECT_PATTERN.search(prompt)
    if match:
        return match.group(1).strip(" .,!?:;").lower()
    return f"prompt_{prompt_number}"


def read_records(input_file: Path) -> list[tuple[int, int, str, str]]:
    text = input_file.read_text(encoding="utf-8")
    matches = RECORD_PATTERN.findall(text)
    if not matches:
        raise ValueError(f"No prompt/response records found in {input_file}.")

    return [
        (int(run), int(number), prompt.strip(), response.strip())
        for run, number, prompt, response in matches
    ]


def normalized_tokens(
    text: str, keep_stopwords: bool, keep_pronouns: bool
) -> tuple[list[str], dict[str, set[str]]]:
    tokens = []
    surface_forms: dict[str, set[str]] = defaultdict(set)
    for surface in TOKEN_PATTERN.findall(text.lower()):
        if surface in ALWAYS_IGNORED_WORDS:
            continue
        if not keep_pronouns and surface in PRONOUNS:
            continue
        if not keep_stopwords and surface in STOPWORDS:
            continue
        normalized = STEMMER.stem(surface)
        tokens.append(normalized)
        surface_forms[normalized].add(surface)
    return tokens, surface_forms


def make_ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]


def analyze(
    records,
    expected_runs: int | None,
    keep_stopwords: bool,
    keep_pronouns: bool,
    top_k: int,
):
    documents: dict[str, list[str]] = defaultdict(list)
    run_numbers: dict[str, set[int]] = defaultdict(set)
    surface_forms: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for run, prompt_number, prompt, response in records:
        subject = extract_subject(prompt, prompt_number)
        tokens, surfaces = normalized_tokens(response, keep_stopwords, keep_pronouns)
        documents[subject].extend(tokens)
        run_numbers[subject].add(run)
        for token, forms in surfaces.items():
            surface_forms[subject][token].update(forms)

    run_counts = {subject: len(runs) for subject, runs in run_numbers.items()}
    counts = ", ".join(
        f"{subject}={count}" for subject, count in sorted(run_counts.items())
    )
    if len(set(run_counts.values())) != 1:
        raise ValueError(
            f"Subjects do not have the same number of runs: {counts}. "
            "Complete the input file before analyzing it."
        )
    inferred_runs = next(iter(run_counts.values()))
    if expected_runs is not None and inferred_runs != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} runs per subject, found {counts}."
        )

    subjects = sorted(documents)
    n_documents = len(subjects)
    rows = []
    for n, ngram_type in ((1, "unigram"), (2, "bigram"), (3, "trigram")):
        counts_by_subject = {
            subject: Counter(make_ngrams(documents[subject], n))
            for subject in subjects
        }
        document_frequency = Counter(
            ngram
            for counts in counts_by_subject.values()
            for ngram in counts
        )

        for subject in subjects:
            counts = counts_by_subject[subject]
            total_ngrams = sum(counts.values())
            for ngram, count in counts.items():
                tf = count / total_ngrams if total_ngrams else 0.0
                idf = math.log(n_documents / document_frequency[ngram])
                rows.append({
                    "subject": subject,
                    "ngram_type": ngram_type,
                    "ngram": ngram,
                    "surface_forms": ", ".join(
                        sorted(set().union(*(surface_forms[subject].get(part, set())
                                             for part in ngram.split())))
                    ),
                    "occurrences": count,
                    "document_ngrams": total_ngrams,
                    "documents_containing_ngram": document_frequency[ngram],
                    "documents": n_documents,
                    "runs_combined": len(run_numbers[subject]),
                    "tf": tf,
                    "idf": idf,
                    "tf_idf": tf * idf,
                })

    type_order = {"unigram": 0, "bigram": 1, "trigram": 2}
    rows.sort(key=lambda row: (
        row["subject"], type_order[row["ngram_type"]], -row["tf_idf"], row["ngram"]
    ))

    # Keep the highest-scoring terms separately for each subject and n-gram size.
    selected_rows = []
    selected_counts = defaultdict(int)
    for row in rows:
        group = (row["subject"], row["ngram_type"])
        if selected_counts[group] < top_k:
            selected_rows.append(row)
            selected_counts[group] += 1
    return selected_rows


def write_excel(rows: list[dict], output_file: Path) -> None:
    """Write a readable workbook with summary and one sheet per n-gram type."""
    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"
    summary.append(["TF-IDF bias analysis"])
    summary.append(["Each subject is one document made from all replicate answers."])
    summary.append(["The CSV contains the same rows as the Top Terms sheet."])
    summary.append([])
    summary.append(["Subject", "N-gram type", "Rows", "Highest TF-IDF"])

    type_order = {"unigram": 0, "bigram": 1, "trigram": 2}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["subject"], row["ngram_type"])].append(row)
    for subject, ngram_type in sorted(groups, key=lambda key: (key[0], type_order[key[1]])):
        group_rows = groups[(subject, ngram_type)]
        summary.append([
            subject,
            ngram_type,
            len(group_rows),
            max(row["tf_idf"] for row in group_rows),
        ])

    headers = list(rows[0]) if rows else []
    for ngram_type in ("unigram", "bigram", "trigram"):
        sheet = workbook.create_sheet(ngram_type.title() + "s")
        sheet.append(headers)
        for row in rows:
            if row["ngram_type"] == ngram_type:
                sheet.append([row[header] for header in headers])
        format_sheet(sheet, headers)

    all_terms = workbook.create_sheet("Top Terms")
    all_terms.append(headers)
    for row in rows:
        all_terms.append([row[header] for header in headers])
    format_sheet(all_terms, headers)

    summary.freeze_panes = "A6"
    summary["A1"].font = Font(bold=True, size=16)
    summary["A2"].font = Font(italic=True, color="666666")
    for cell in summary[5]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    summary.auto_filter.ref = f"A5:D{summary.max_row}"
    summary.column_dimensions["A"].width = 20
    summary.column_dimensions["B"].width = 16
    summary.column_dimensions["C"].width = 10
    summary.column_dimensions["D"].width = 18
    for row in summary.iter_rows(min_row=6, min_col=4, max_col=4):
        row[0].number_format = "0.000000"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_file)


def format_sheet(sheet, headers: list[str]) -> None:
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    for index, header in enumerate(headers, start=1):
        width = max(12, min(32, len(header) + 2))
        if header in {"ngram", "surface_forms"}:
            width = 30
        sheet.column_dimensions[get_column_letter(index)].width = width
    if "tf_idf" in headers and sheet.max_row > 1:
        tfidf_column = get_column_letter(headers.index("tf_idf") + 1)
        tfidf_range = f"{tfidf_column}2:{tfidf_column}{sheet.max_row}"
        sheet.conditional_formatting.add(
            tfidf_range,
            ColorScaleRule(
                start_type="min", start_color="E2F0D9",
                mid_type="percentile", mid_value=50, mid_color="FFF2CC",
                end_type="max", end_color="F4CCCC",
            ),
        )
    for header in ("tf", "idf", "tf_idf"):
        if header in headers:
            column = headers.index(header) + 1
            for column_cells in sheet.iter_cols(
                min_col=column, max_col=column, min_row=2
            ):
                for cell in column_cells:
                    cell.number_format = "0.000000"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", type=Path, nargs="?", default=PRIMARY_DATA,
        help="Completed response file (default: data/main/responses.txt).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output CSV; defaults to the main or experiment-specific results folder.",
    )
    parser.add_argument(
        "--excel-output", type=Path, default=None,
        help="Excel output path (default: same name as CSV, with .xlsx).",
    )
    parser.add_argument(
        "--expected-runs", type=int, default=None,
        help="Optional run-count check; otherwise the count is inferred from the file.",
    )
    parser.add_argument(
        "--remove-stopwords", action="store_true",
        help="Also remove a broader list of common words; articles, prepositions, and conjunctions are always removed.",
    )
    parser.add_argument(
        "--keep-pronouns", action="store_true",
        help="Keep pronouns in the answer text; they are removed by default.",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of highest-scoring terms per subject and n-gram type (default: 10).",
    )
    args = parser.parse_args()
    if args.output is None:
        result_dir = (PRIMARY_RESULTS if args.input_file.resolve() == PRIMARY_DATA.resolve()
                      else EXPERIMENT_RESULTS / args.input_file.parent.name / args.input_file.stem)
        args.output = result_dir / 'tfidf/terms.csv'

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    records = read_records(args.input_file)
    rows = analyze(
        records,
        args.expected_runs,
        not args.remove_stopwords,
        args.keep_pronouns,
        args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    excel_output = args.excel_output or args.output.with_suffix(".xlsx")
    write_excel(rows, excel_output)
    print(f"Analyzed {len(records)} answers across {len(set(row['subject'] for row in rows))} subjects.")
    print(f"Wrote {len(rows)} TF-IDF rows to {args.output}")
    print(f"Wrote formatted workbook to {excel_output}")


if __name__ == "__main__":
    main()
