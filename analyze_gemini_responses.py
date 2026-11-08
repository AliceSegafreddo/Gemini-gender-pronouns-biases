from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
import argparse
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from project_paths import PRIMARY_DATA, PRIMARY_RESULTS, EXPERIMENT_RESULTS


INPUT_FILE = PRIMARY_DATA
IMAGE_ROOT = PRIMARY_RESULTS / 'embeddings'
EMBEDDING_MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
    ("sentence-transformers/all-MiniLM-L12-v2", "all-MiniLM-L12-v2"),
    ("sentence-transformers/paraphrase-MiniLM-L3-v2", "paraphrase-MiniLM-L3-v2"),
    ("BAAI/bge-small-en-v1.5", "bge-small-en-v1.5"),
]
PRIMARY_MODEL_SLUG = "all-MiniLM-L12-v2"
MODEL_BY_SLUG = {slug: (name, slug) for name, slug in EMBEDDING_MODELS}
RECORD_PATTERN = re.compile(
    r"--- Run (\d+), Prompt (\d+) ---\n"
    r"PROMPT:\n(.*?)\n\n"
    r"RESPONSE:\n(.*?)(?=\n--- Run |\Z)",
    re.DOTALL,
)
SUBJECT_PATTERN = re.compile(
    r"^\s*(.+?)\s+exaggerat(?:e|es|ed|ing)\b", re.IGNORECASE
)


def extract_subject(prompt: str, index: int) -> str:
    """Extract the complete subject before the prompt's main verb."""
    match = SUBJECT_PATTERN.search(prompt)
    if match:
        return match.group(1).strip(" .,!?:;\t")

    # Keep a useful fallback if a future prompt uses a different wording.
    return prompt.split(maxsplit=1)[0].strip(".,!?;:") or f"Prompt {index + 1}"


def read_records(input_file: Path = INPUT_FILE) -> tuple[dict[int, dict[int, tuple[str, str]]], list[str]]:
    if not input_file.exists():
        raise FileNotFoundError(
            f"{input_file} does not exist. Run collect_gemini_responses.py first."
        )

    matches = RECORD_PATTERN.findall(input_file.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"No prompt/response records found in {input_file}.")

    records: dict[int, dict[int, tuple[str, str]]] = {}
    prompts_by_number: dict[int, str] = {}
    for run_text, prompt_number_text, prompt, response in matches:
        run = int(run_text)
        prompt_number = int(prompt_number_text)
        prompt = prompt.strip()
        response = response.strip()
        records.setdefault(run, {})[prompt_number] = (prompt, response)

        previous_prompt = prompts_by_number.setdefault(prompt_number, prompt)
        if previous_prompt != prompt:
            raise ValueError(
                f"Prompt {prompt_number} changes between runs: "
                f"{previous_prompt!r} vs {prompt!r}"
            )

    prompts = [prompts_by_number[number] for number in sorted(prompts_by_number)]
    return records, prompts


def save_2d_plot(
    embeddings,
    prompts,
    prompt_labels,
    pair_scores,
    title,
    output_file,
    flagged_pairs=None,
) -> None:
    coordinates = PCA(n_components=2).fit_transform(embeddings)
    n_prompts = len(prompts)
    colors = [plt.get_cmap("tab10")(index % 10) for index in range(n_prompts)]
    fig, ax = plt.subplots(figsize=(13, max(8, 5 + 0.25 * len(pair_scores))))

    for index in range(n_prompts):
        ax.scatter(
            coordinates[index, 0],
            coordinates[index, 1],
            s=220,
            color=colors[index],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            prompt_labels[index],
            (coordinates[index, 0], coordinates[index, 1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

    for edge_number, ((first, second), scores) in enumerate(pair_scores.items()):
        # Curved edges keep overlapping pair connections distinguishable.
        side = -1 if edge_number % 2 else 1
        radius = side * (0.10 + 0.035 * (edge_number // 2))
        edge = FancyArrowPatch(
            coordinates[first],
            coordinates[second],
            arrowstyle="-",
            connectionstyle=f"arc3,rad={radius}",
            color="#D62728" if flagged_pairs and (first, second) in flagged_pairs else "#777777",
            linewidth=2.2 if flagged_pairs and (first, second) in flagged_pairs else 1.2,
            alpha=0.7,
            zorder=1,
        )
        ax.add_patch(edge)

        if n_prompts <= 5:
            midpoint = (coordinates[first] + coordinates[second]) / 2
            difference = coordinates[second] - coordinates[first]
            length = np.linalg.norm(difference)
            perpendicular = np.array([-difference[1], difference[0]]) / max(length, 1e-12)
            label_position = midpoint + perpendicular * radius * length * 0.7
            ax.annotate(
                f"mean cosine = {mean(scores):.3f}",
                label_position,
                fontsize=9,
                color="#555555",
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[index],
            markersize=9,
            label=f"{prompt_labels[index]}: {textwrap.fill(prompt, width=42)}",
        )
        for index, prompt in enumerate(prompts)
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.70, 0.86),
        frameon=False,
        fontsize=9,
    )

    pair_lines = [
        f"{'! ' if flagged_pairs and (first, second) in flagged_pairs else '  '}"
        f"{prompt_labels[first]} ↔ {prompt_labels[second]} = {mean(scores):.3f}"
        for (first, second), scores in pair_scores.items()
    ]
    fig.text(
        0.70,
        0.04,
        "Pairwise mean cosine similarities\n" + "\n".join(pair_lines),
        va="bottom",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9, "pad": 5},
    )

    ax.set_title(title)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.grid(alpha=0.25)
    ax.margins(x=0.35, y=0.35)
    fig.subplots_adjust(right=0.67, bottom=0.08)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_similarity_heatmap(pair_scores, prompt_labels, title, output_file) -> None:
    n_prompts = len(prompt_labels)
    matrix = np.eye(n_prompts)
    for (first, second), scores in pair_scores.items():
        similarity = mean(scores)
        matrix[first, second] = similarity
        matrix[second, first] = similarity

    fig_size = max(6, min(16, 2.0 + 1.0 * n_prompts))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Prompt")
    ax.set_xticks(range(n_prompts), labels=prompt_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_prompts), labels=prompt_labels)

    for row in range(n_prompts):
        for column in range(n_prompts):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < 0.65 else "black",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compare_similarity_rankings(
    prompt_pair_scores,
    answer_pair_scores,
    prompt_labels,
    output_file,
):
    prompt_values = {pair: mean(scores) for pair, scores in prompt_pair_scores.items()}
    answer_values = {pair: mean(scores) for pair, scores in answer_pair_scores.items()}
    prompt_order = sorted(prompt_values, key=prompt_values.get, reverse=True)
    answer_order = sorted(answer_values, key=answer_values.get, reverse=True)
    prompt_ranks = {pair: rank for rank, pair in enumerate(prompt_order, start=1)}
    answer_ranks = {pair: rank for rank, pair in enumerate(answer_order, start=1)}

    pairs = list(prompt_values)
    rank_shift_threshold = max(1, len(pairs) // 6)
    flagged_pairs = {
        pair
        for pair in pairs
        if abs(answer_ranks[pair] - prompt_ranks[pair]) >= rank_shift_threshold
    }

    prompt_rank_vector = np.array([prompt_ranks[pair] for pair in pairs])
    answer_rank_vector = np.array([answer_ranks[pair] for pair in pairs])
    rank_correlation = float(np.corrcoef(prompt_rank_vector, answer_rank_vector)[0, 1])

    ranked_comparison = sorted(
        pairs,
        key=lambda pair: abs(answer_ranks[pair] - prompt_ranks[pair]),
        reverse=True,
    )
    lines = [
        "Prompt-vs-answer similarity ranking divergence",
        "================================================",
        f"Rank correlation (Spearman-style): {rank_correlation:.3f}",
        f"Large-shift threshold: {rank_shift_threshold} ranking positions",
        "",
        "FLAGGED PAIRS",
    ]

    if not flagged_pairs:
        lines.append("None")
    else:
        for pair in ranked_comparison:
            if pair not in flagged_pairs:
                continue
            first, second = pair
            shift = answer_ranks[pair] - prompt_ranks[pair]
            lines.append(
                f"! {prompt_labels[first]} ↔ {prompt_labels[second]} | "
                f"prompt rank {prompt_ranks[pair]} ({prompt_values[pair]:.3f}) -> "
                f"answer rank {answer_ranks[pair]} ({answer_values[pair]:.3f}) | "
                f"rank shift {shift:+d} | cosine change "
                f"{answer_values[pair] - prompt_values[pair]:+.3f}"
            )

    lines.extend(["", "ALL PAIRS (largest rank shifts first)"])
    for pair in ranked_comparison:
        first, second = pair
        shift = answer_ranks[pair] - prompt_ranks[pair]
        lines.append(
            f"{prompt_labels[first]} ↔ {prompt_labels[second]} | "
            f"prompt rank {prompt_ranks[pair]} -> answer rank {answer_ranks[pair]} | "
            f"prompt {prompt_values[pair]:.3f} | answer {answer_values[pair]:.3f} | "
            f"shift {shift:+d}"
        )

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return flagged_pairs, rank_correlation


def analyze_model(model_name, model_slug, records, prompts, pairs, image_root=IMAGE_ROOT):
    print(f"\n=== Embedding model: {model_name} ===")
    embedder = SentenceTransformer(model_name)
    prompt_labels = [extract_subject(prompt, index) for index, prompt in enumerate(prompts)]
    output_directory = image_root / model_slug
    output_directory.mkdir(parents=True, exist_ok=True)
    answer_plot_file = output_directory / "answer_embeddings_2d.png"
    prompt_plot_file = output_directory / "prompt_embeddings_2d.png"
    answer_heatmap_file = output_directory / "answer_similarity_heatmap.png"
    prompt_heatmap_file = output_directory / "prompt_similarity_heatmap.png"
    divergence_report_file = output_directory / "prompt_answer_ranking_divergence.txt"

    prompt_embeddings = embedder.encode(prompts, normalize_embeddings=True)
    prompt_pair_scores = {
        pair: [float(prompt_embeddings[pair[0]] @ prompt_embeddings[pair[1]])]
        for pair in pairs
    }
    pair_scores = {pair: [] for pair in pairs}
    run_embeddings = []
    prompt_numbers = sorted(records[min(records)])
    for run in sorted(records):
        run_records = records[run]
        if sorted(run_records) != prompt_numbers:
            raise ValueError(f"Run {run} does not contain exactly all prompts.")

        responses = [run_records[number][1] for number in prompt_numbers]
        embeddings = embedder.encode(responses, normalize_embeddings=True)
        run_embeddings.append(embeddings)

        for first, second in pairs:
            similarity = float(embeddings[first] @ embeddings[second])
            pair_scores[(first, second)].append(similarity)

    if len(run_embeddings) < 2:
        raise ValueError("At least two runs are required to calculate sample standard deviation.")

    print("Answer similarity summary:")
    for first, second in pairs:
        scores = pair_scores[(first, second)]
        print(
            f"Prompts {first + 1} & {second + 1}: "
            f"scores={', '.join(f'{score:.4f}' for score in scores)}, "
            f"mean={mean(scores):.4f}, sample std={stdev(scores):.4f}"
        )

    average_embeddings = np.mean(np.stack(run_embeddings), axis=0)
    average_embeddings /= np.linalg.norm(average_embeddings, axis=1, keepdims=True)
    flagged_pairs, rank_correlation = compare_similarity_rankings(
        prompt_pair_scores,
        pair_scores,
        prompt_labels,
        divergence_report_file,
    )
    save_2d_plot(
        average_embeddings,
        prompts,
        prompt_labels,
        pair_scores,
        f"2D PCA of average answer embeddings\n{model_name}",
        answer_plot_file,
        flagged_pairs,
    )
    save_2d_plot(
        prompt_embeddings,
        prompts,
        prompt_labels,
        prompt_pair_scores,
        f"2D PCA of prompt embeddings\n{model_name}",
        prompt_plot_file,
        flagged_pairs,
    )
    save_similarity_heatmap(
        pair_scores,
        prompt_labels,
        f"Answer cosine similarities\n{model_name}",
        answer_heatmap_file,
    )
    save_similarity_heatmap(
        prompt_pair_scores,
        prompt_labels,
        f"Prompt cosine similarities\n{model_name}",
        prompt_heatmap_file,
    )

    print("\n--- Prompt similarity summary ---")
    for first, second in pairs:
        similarity = prompt_pair_scores[(first, second)][0]
        print(f"Prompts {first + 1} & {second + 1}: cosine={similarity:.4f}")

    print(f"Answer visualization saved to {answer_plot_file}")
    print(f"Prompt visualization saved to {prompt_plot_file}")
    print(f"Answer heatmap saved to {answer_heatmap_file}")
    print(f"Prompt heatmap saved to {prompt_heatmap_file}")
    print(f"Ranking divergence report saved to {divergence_report_file}")
    print(f"Flagged pairs: {len(flagged_pairs)}; rank correlation: {rank_correlation:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze saved Gemini responses with local embedding models."
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run the comparison for every configured embedding model.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_BY_SLUG,
        help=f"Run one configured model (default: {PRIMARY_MODEL_SLUG}).",
    )
    parser.add_argument(
        "--input-file", type=Path, default=INPUT_FILE,
        help=f"Saved prompt/response file (default: {INPUT_FILE}).",
    )
    parser.add_argument(
        "--image-root", type=Path, default=None,
        help="Figure folder; defaults to the main or experiment-specific results folder.",
    )
    args = parser.parse_args()
    if args.image_root is None:
        args.image_root = (IMAGE_ROOT if args.input_file.resolve() == PRIMARY_DATA.resolve()
                           else EXPERIMENT_RESULTS / args.input_file.parent.name / args.input_file.stem / 'embeddings')

    records, prompts = read_records(args.input_file)
    pairs = list(combinations(range(len(prompts)), 2))

    if args.all_models and args.model:
        parser.error("Use either --all-models or --model, not both.")

    if args.all_models:
        selected_models = EMBEDDING_MODELS
    elif args.model:
        selected_models = [MODEL_BY_SLUG[args.model]]
    else:
        selected_models = [MODEL_BY_SLUG[PRIMARY_MODEL_SLUG]]

    print("Embedding models: " + ", ".join(slug for _, slug in selected_models))
    for model_name, model_slug in selected_models:
        analyze_model(model_name, model_slug, records, prompts, pairs, args.image_root)


if __name__ == "__main__":
    main()
