"""Evaluate a model on GSM8K and MATH with simple latency metrics."""

import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset
from dotenv import load_dotenv
import matplotlib.pyplot as plt


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "25"))
MATH_DATASET = os.getenv("MATH_DATASET", "HuggingFaceH4/MATH-500")


def load_benchmarks() -> list[dict[str, str]]:
    """Load a small, reproducible test slice from both benchmark datasets."""
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    math = load_dataset(MATH_DATASET, split="test")
    examples: list[dict[str, str]] = []
    for row in gsm8k.select(range(min(MAX_SAMPLES, len(gsm8k)))):
        examples.append({"dataset": "gsm8k", "question": row["question"], "answer": row["answer"]})
    for row in math.select(range(min(MAX_SAMPLES, len(math)))):
        examples.append({"dataset": "math", "question": row["problem"], "answer": row["answer"]})
    return examples


def answer_text(value: Any) -> str:
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("generated_text", value.get("text", value))
    return str(value)


def expected_answer(reference: str) -> str:
    """Return a normalized answer suitable for a starter exact-match metric."""
    if "####" in reference:
        reference = reference.rsplit("####", 1)[1]
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", reference)
    return (boxed[-1] if boxed else reference).strip().replace(",", "")


def predicted_answer(response: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", response)
    if boxed:
        return boxed[-1].strip().replace(",", "")
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", response.replace(",", ""))
    return numbers[-1] if numbers else response.strip()


def evaluate(generate: Callable[[str], str]) -> None:
    examples = load_benchmarks()
    correct = 0
    total_latency = 0.0
    by_dataset: dict[str, list[float | int]] = {}
    for example in examples:
        start = time.perf_counter()
        response = generate(example["question"])
        latency = time.perf_counter() - start
        total_latency += latency
        is_correct = predicted_answer(answer_text(response)) == expected_answer(example["answer"])
        correct += int(is_correct)
        stats = by_dataset.setdefault(example["dataset"], [0, 0.0, 0])
        stats[0] += int(is_correct)
        stats[1] += latency
        stats[2] += 1
    print(f"overall_accuracy={correct / len(examples):.3f}")
    print(f"mean_latency_seconds={total_latency / len(examples):.3f}")
    dataset_names = list(by_dataset)
    accuracies = [by_dataset[name][0] / by_dataset[name][2] for name in dataset_names]
    plt.figure(figsize=(6, 4))
    plt.bar(dataset_names, accuracies, color=["#4C78A8", "#F58518"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Dataset")
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "accuracy.jpeg", format="jpeg", dpi=150)
    plt.close()
    for name, (hits, latency, count) in by_dataset.items():
        print(f"{name}_accuracy={hits / count:.3f} mean_latency_seconds={latency / count:.3f}")


from google import genai


client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL_ID = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")


def generate(question: str) -> str:
    prompt = f"Solve this math problem. Give the final answer clearly.\n\nQuestion: {{question}}\nAnswer:"
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    return response.text


if __name__ == "__main__":
    evaluate(generate)
