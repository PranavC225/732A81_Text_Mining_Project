import json, os, sys
sys.path.insert(0, "/workspace")

from agent.rag_agent import load_chain
import evaluate as hf_evaluate
import pandas as pd

DATASET_PATH = "eval/eval_dataset.json"
RESULTS_PATH = "eval/results.json"

print("Loading evaluation metrics...")
rouge     = hf_evaluate.load("rouge")
bertscore = hf_evaluate.load("bertscore")

print("Loading RAG chain...")
chain = load_chain(top_k=5)

with open(DATASET_PATH) as f:
    dataset = json.load(f)

print(f"\nRunning agent on {len(dataset)} questions...\n")
predictions, references, details = [], [], []

for item in dataset:
    pred = chain.invoke(item["question"])
    predictions.append(pred)
    references.append(item["gold_answer"])
    details.append({
        "id":           item["id"],
        "category":     item.get("category", ""),
        "question":     item["question"],
        "gold_answer":  item["gold_answer"],
        "predicted":    pred,
    })
    print(f"  [{item['id']:02d}] {item['question'][:60]}...")

print("\nComputing ROUGE scores...")
rouge_scores = rouge.compute(predictions=predictions, references=references)

print("Computing BERTScore...")
bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")

for i, d in enumerate(details):
    d["bertscore_f1"] = round(bert_scores["f1"][i], 4)

avg_bertscore = sum(bert_scores["f1"]) / len(bert_scores["f1"])

# Per-category breakdown
df = pd.DataFrame(details)
cat_summary = (
    df.groupby("category")["bertscore_f1"]
    .agg(["mean", "count"])
    .round(4)
    .rename(columns={"mean": "avg_bertscore_f1", "count": "n"})
    .to_dict(orient="index")
)

summary = {
    "n_questions":        len(dataset),
    "rouge1":             round(rouge_scores["rouge1"], 4),
    "rouge2":             round(rouge_scores["rouge2"], 4),
    "rougeL":             round(rouge_scores["rougeL"], 4),
    "bertscore_f1_avg":   round(avg_bertscore, 4),
    "per_category":       cat_summary,
}

print("\n=== Results ===")
print(f"ROUGE-1:          {summary['rouge1']}")
print(f"ROUGE-2:          {summary['rouge2']}")
print(f"ROUGE-L:          {summary['rougeL']}")
print(f"BERTScore F1 avg: {summary['bertscore_f1_avg']}")
print("\nPer-category BERTScore F1:")
for cat, vals in cat_summary.items():
    print(f"  {cat:<25} {vals['avg_bertscore_f1']}  (n={vals['n']})")

with open(RESULTS_PATH, "w") as f:
    json.dump({"summary": summary, "detail": details}, f, indent=2)

print(f"\nFull results saved to {RESULTS_PATH}")