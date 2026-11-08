"""Distance analysis after averaging correct/altered responses by pronoun."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from sentence_transformers import SentenceTransformer

import analyze_gemini_responses as base
from project_paths import PRIMARY_DATA, EXPERIMENT_DATA, EXPERIMENT_RESULTS


def normalize(vector):
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correct-file", type=Path,
        default=PRIMARY_DATA,
    )
    parser.add_argument(
        "--altered-file", type=Path,
        default=EXPERIMENT_DATA / 'altered_conjugation/responses.txt',
    )
    parser.add_argument("--image-root", type=Path, default=EXPERIMENT_RESULTS / 'combined_conjugations/embeddings')
    parser.add_argument("--model", default="all-MiniLM-L12-v2", choices=base.MODEL_BY_SLUG)
    args = parser.parse_args()

    correct_records, correct_prompts = base.read_records(args.correct_file)
    altered_records, altered_prompts = base.read_records(args.altered_file)
    if len(correct_prompts) != len(altered_prompts):
        raise ValueError("Correct and altered files do not contain the same number of prompts.")

    correct_prompt_numbers = sorted(correct_records[min(correct_records)])
    altered_prompt_numbers = sorted(altered_records[min(altered_records)])
    if len(correct_prompt_numbers) != len(altered_prompt_numbers):
        raise ValueError("Correct and altered files do not have matching prompt sets.")

    model_name, model_slug = base.MODEL_BY_SLUG[args.model]
    print(f"Embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    correct_labels = [
        base.extract_subject(prompt, index) for index, prompt in enumerate(correct_prompts)
    ]
    altered_labels = [
        base.extract_subject(prompt, index) for index, prompt in enumerate(altered_prompts)
    ]
    altered_index_by_label = {label: index for index, label in enumerate(altered_labels)}
    if set(correct_labels) != set(altered_labels):
        raise ValueError("Correct and altered files do not contain the same subjects.")
    altered_order = [altered_index_by_label[label] for label in correct_labels]

    # Average the prompt vectors for matching pronouns, not file positions.
    correct_prompt_embeddings = embedder.encode(correct_prompts, normalize_embeddings=True)
    altered_prompt_embeddings = embedder.encode(altered_prompts, normalize_embeddings=True)[altered_order]
    prompt_embeddings = normalize((correct_prompt_embeddings + altered_prompt_embeddings) / 2)

    runs = sorted(correct_records)
    if runs != sorted(altered_records):
        raise ValueError("Correct and altered files do not contain the same runs.")

    pair_indices = list(combinations(range(len(correct_prompts)), 2))
    pair_scores = {pair: [] for pair in pair_indices}
    run_embeddings = []
    labels = correct_labels
    altered_number_by_label = {
        label: altered_prompt_numbers[index]
        for label, index in altered_index_by_label.items()
    }

    for run in runs:
        correct_run = correct_records[run]
        altered_run = altered_records[run]
        if sorted(correct_run) != correct_prompt_numbers or sorted(altered_run) != altered_prompt_numbers:
            raise ValueError(f"Run {run} does not contain every expected prompt.")

        correct_answers = [correct_run[number][1] for number in correct_prompt_numbers]
        altered_answers = [altered_run[altered_number_by_label[label]][1] for label in labels]
        correct_embeddings = embedder.encode(correct_answers, normalize_embeddings=True)
        altered_embeddings = embedder.encode(altered_answers, normalize_embeddings=True)
        averaged_embeddings = normalize((correct_embeddings + altered_embeddings) / 2)
        run_embeddings.append(averaged_embeddings)
        for first, second in pair_indices:
            pair_scores[(first, second)].append(
                float(averaged_embeddings[first] @ averaged_embeddings[second])
            )

    if len(run_embeddings) < 2:
        raise ValueError("At least two runs are required for sample standard deviation.")

    prompt_pair_scores = {
        pair: [float(prompt_embeddings[pair[0]] @ prompt_embeddings[pair[1]])]
        for pair in pair_indices
    }
    average_embeddings = normalize(np.mean(np.stack(run_embeddings), axis=0))
    output_directory = args.image_root / model_slug
    output_directory.mkdir(parents=True, exist_ok=True)
    flagged_pairs, rank_correlation = base.compare_similarity_rankings(
        prompt_pair_scores,
        pair_scores,
        labels,
        output_directory / "prompt_answer_ranking_divergence.txt",
    )
    base.save_2d_plot(
        average_embeddings, correct_prompts, labels, pair_scores,
        f"2D PCA of averaged correct+altered answer embeddings\n{model_name}",
        output_directory / "answer_embeddings_2d.png", flagged_pairs,
    )
    base.save_2d_plot(
        prompt_embeddings, correct_prompts, labels, prompt_pair_scores,
        f"2D PCA of averaged correct+altered prompt embeddings\n{model_name}",
        output_directory / "prompt_embeddings_2d.png", flagged_pairs,
    )
    base.save_similarity_heatmap(
        pair_scores, labels, f"Averaged correct+altered answer cosine similarities\n{model_name}",
        output_directory / "answer_similarity_heatmap.png",
    )
    base.save_similarity_heatmap(
        prompt_pair_scores, labels, f"Averaged correct+altered prompt cosine similarities\n{model_name}",
        output_directory / "prompt_similarity_heatmap.png",
    )

    print(f"Combined runs: {len(runs)}; subjects: {', '.join(labels)}")
    print("Combined answer similarity summary:")
    for first, second in pair_indices:
        scores = pair_scores[(first, second)]
        print(
            f"{labels[first]}–{labels[second]}: mean={mean(scores):.4f}, "
            f"sample std={stdev(scores):.4f}"
        )
    print("Combined prompt similarity summary:")
    for first, second in pair_indices:
        print(f"{labels[first]}–{labels[second]}: cosine={prompt_pair_scores[(first, second)][0]:.4f}")
    print(f"Figures saved to {output_directory}")
    print(f"Flagged pairs: {len(flagged_pairs)}; rank correlation: {rank_correlation:.3f}")


if __name__ == "__main__":
    main()
