from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INPUT_FILE = Path("gemini_prompts_responses.txt")
PLOT_FILE = Path("prompt_embeddings_2d.png")
RECORD_PATTERN = re.compile(
    r"--- Run (\d+), Prompt (\d+) ---\n"
    r"PROMPT:\n(.*?)\n\n"
    r"RESPONSE:\n(.*?)\n\n",
    re.DOTALL,
)


def read_records() -> tuple[dict[int, dict[int, tuple[str, str]]], list[str]]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} does not exist. Run collect_gemini_responses.py first."
        )

    matches = RECORD_PATTERN.findall(INPUT_FILE.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"No prompt/response records found in {INPUT_FILE}.")

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


def save_2d_plot(average_embeddings, prompts, pair_scores) -> None:
    coordinates = PCA(n_components=2).fit_transform(average_embeddings)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=180, color="#4472C4", zorder=3)

    for index, prompt in enumerate(prompts):
        label = f"P{index + 1}: {textwrap.fill(prompt, width=28)}"
        ax.annotate(
            label,
            (coordinates[index, 0], coordinates[index, 1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
        )

    for (first, second), scores in pair_scores.items():
        ax.plot(
            [coordinates[first, 0], coordinates[second, 0]],
            [coordinates[first, 1], coordinates[second, 1]],
            color="#999999",
            linewidth=1,
            zorder=1,
        )
        midpoint = (coordinates[first] + coordinates[second]) / 2
        ax.annotate(
            f"mean cosine = {mean(scores):.3f}",
            midpoint,
            fontsize=9,
            color="#555555",
            ha="center",
            va="bottom",
        )

    ax.set_title("2D PCA of average prompt embeddings")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.grid(alpha=0.25)
    ax.margins(x=0.35, y=0.35)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    records, prompts = read_records()
    prompt_numbers = list(range(1, len(prompts) + 1))
    pairs = list(combinations(range(len(prompts)), 2))

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    pair_scores = {pair: [] for pair in pairs}
    run_embeddings = []

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

    print("--- Similarity summary ---")
    for first, second in pairs:
        scores = pair_scores[(first, second)]
        print(
            f"Prompts {first + 1} & {second + 1}: "
            f"scores={', '.join(f'{score:.4f}' for score in scores)}, "
            f"mean={mean(scores):.4f}, sample std={stdev(scores):.4f}"
        )

    average_embeddings = np.mean(np.stack(run_embeddings), axis=0)
    average_embeddings /= np.linalg.norm(average_embeddings, axis=1, keepdims=True)
    save_2d_plot(average_embeddings, prompts, pair_scores)
    print(f"2D visualization saved to {PLOT_FILE}")


if __name__ == "__main__":
    main()
