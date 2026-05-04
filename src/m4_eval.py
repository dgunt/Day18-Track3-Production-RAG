"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def empty_result() -> dict:
        return {
            **{name: 0.0 for name in metric_names},
            "per_question": [
                EvalResult(q, a, c, gt, 0.0, 0.0, 0.0, 0.0)
                for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
            ],
        }

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from ragas.llms.base import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        return empty_result()

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0),
        bypass_n=True,
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
    except Exception:
        return empty_result()

    df = result.to_pandas()

    def as_float(value) -> float:
        try:
            value = float(value)
            return value if value == value else 0.0
        except (TypeError, ValueError):
            return 0.0

    def aggregate_score(name: str) -> float:
        try:
            return as_float(result[name])
        except (KeyError, TypeError):
            return as_float(df[name].mean()) if name in df else 0.0

    per_question = [
        EvalResult(
            question=questions[i],
            answer=answers[i],
            contexts=contexts[i],
            ground_truth=ground_truths[i],
            faithfulness=as_float(getattr(row, "faithfulness", 0.0)),
            answer_relevancy=as_float(getattr(row, "answer_relevancy", 0.0)),
            context_precision=as_float(getattr(row, "context_precision", 0.0)),
            context_recall=as_float(getattr(row, "context_recall", 0.0)),
        )
        for i, row in enumerate(df.itertuples(index=False))
    ]

    return {
        **{name: aggregate_score(name) for name in metric_names},
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    from heapq import nsmallest

    if bottom_n <= 0:
        return []
    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    diagnosis_map = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    def avg_score(result: EvalResult) -> float:
        return sum(float(getattr(result, metric, 0.0)) for metric in metrics) / len(metrics)

    failures = []
    for result in nsmallest(bottom_n, eval_results, key=avg_score):
        worst_metric = min(metrics, key=lambda metric: float(getattr(result, metric, 0.0)))
        diagnosis, fix = diagnosis_map[worst_metric]
        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": float(getattr(result, worst_metric, 0.0)),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
