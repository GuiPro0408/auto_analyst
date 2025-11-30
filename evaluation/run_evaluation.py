"""Run evaluation metrics across a dataset."""

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # Ensure local package imports work when run as a script
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import evaluate_all


def load_dataset(path: Path):
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of items.")
    return data


def main(dataset_path: str, model_name: str):
    data = load_dataset(Path(dataset_path))
    scores = []
    for item in data:
        question = item["question"]
        reference = item["reference_answer"]
        answer = item["answer"]
        contexts: List[str] = item.get("contexts", [])
        score = evaluate_all(
            question=question,
            answer=answer,
            reference=reference,
            contexts=contexts,
            model_name=model_name,
        )
        scores.append(score)
    # Aggregate averages
    agg = (
        {k: sum(s[k] for s in scores) / len(scores) for k in scores[0].keys()}
        if scores
        else {}
    )
    print("Aggregate scores")
    for metric, value in agg.items():
        print(f"{metric}: {value:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Auto-Analyst outputs.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSON list with question, reference_answer, answer, contexts.",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="Embedding model used for scoring."
    )
    args = parser.parse_args()
    main(args.dataset, args.model)
