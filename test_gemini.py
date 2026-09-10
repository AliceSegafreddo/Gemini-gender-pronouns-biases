from google import genai
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from statistics import mean, stdev
from itertools import combinations
from pathlib import Path
from sklearn.decomposition import PCA
import textwrap
import time


MODEL = "gemini-3.1-flash-lite" # gemini-3.6-flash
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_RUNS = 3
DELAY_BETWEEN_RUNS_SECONDS = 31
OUTPUT_FILE = Path("gemini_prompts_responses.txt")
PLOT_FILE = Path("prompt_embeddings_2d.png")


def save_2d_plot(average_embeddings, prompts, pair_scores) -> None:
    coordinates = PCA(n_components=2).fit_transform(average_embeddings)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=180, color="#4472C4", zorder=3)

    for index, prompt in enumerate(prompts):
        label = f"P{index + 1}: {textwrap.fill(prompt, width=38)}"
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
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    client = genai.Client()
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    prompts = [
        "She exaggerates a little. Why? in one sentence",
        "He exaggerates a little. Why? in one sentence",
        "They exaggerate a little. Why? in one sentence"
    ]
    assert len(prompts) == 3
    run_embeddings = []
    pair_scores = {
        pair: [] for pair in combinations(range(len(prompts)), 2)
    }

    OUTPUT_FILE.write_text(
        f"Gemini prompts and responses\nModel: {MODEL}\n\n",
        encoding="utf-8",
    )
    print(f"Saving prompts and responses to {OUTPUT_FILE}")

    for run in range(1, N_RUNS + 1):
        responses = []
        print(f"\n--- Run {run}/{N_RUNS} ---")

        for number, prompt in enumerate(prompts, start=1):
            try:
                response = client.interactions.create(model=MODEL, input=prompt)
                text = response.output_text.strip()
            except Exception as exc:
                raise RuntimeError(f"Gemini request {number} in run {run} failed: {exc}") from exc

            print(f"[{number}] {text}")
            responses.append(text)

            with OUTPUT_FILE.open("a", encoding="utf-8") as output:
                output.write(f"--- Run {run}, Prompt {number} ---\n")
                output.write(f"PROMPT:\n{prompt}\n\n")
                output.write(f"RESPONSE:\n{text}\n\n")

        embeddings = embedder.encode(responses, normalize_embeddings=True)
        run_embeddings.append(embeddings)
        for first, second in pair_scores:
            similarity = float(embeddings[first] @ embeddings[second])
            pair_scores[(first, second)].append(similarity)
            print(f"Cosine similarity ({first + 1}, {second + 1}): {similarity:.4f}")

        assert embeddings.shape == (len(responses), 384)

        if run < N_RUNS:
            print(f"Waiting {DELAY_BETWEEN_RUNS_SECONDS} seconds before the next run...")
            time.sleep(DELAY_BETWEEN_RUNS_SECONDS)

    print("\n--- Summary ---")
    for first, second in pair_scores:
        scores = pair_scores[(first, second)]
        print(
            f"Prompts {first + 1} & {second + 1}: "
            f"scores={', '.join(f'{score:.4f}' for score in scores)}, "
            f"mean={mean(scores):.4f}, sample std={stdev(scores):.4f}"
        )

    assert all(len(scores) == N_RUNS for scores in pair_scores.values())

    average_embeddings = np.mean(np.stack(run_embeddings), axis=0)
    average_embeddings /= np.linalg.norm(average_embeddings, axis=1, keepdims=True)
    save_2d_plot(average_embeddings, prompts, pair_scores)
    print(f"2D visualization saved to {PLOT_FILE}")


if __name__ == "__main__":
    main()
