import json, os, sys
sys.path.insert(0, "/workspace")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import evaluate as hf_evaluate

CLEAN_DIR   = "data/clean"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DATASET_PATH = "eval/eval_dataset.json"

PROMPT_TEMPLATE = """You are a helpful assistant for international students at Linköping University.
Answer the question using only the context provided below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

CONFIGS = [
    {"chunk_size": 500, "chunk_overlap": 80,  "top_k": 5},  # baseline
    {"chunk_size": 250, "chunk_overlap": 40,  "top_k": 5},  # smaller chunks
    {"chunk_size": 500, "chunk_overlap": 80,  "top_k": 8},  # more context
    {"chunk_size": 250, "chunk_overlap": 40,  "top_k": 8},  # both
]

rouge     = hf_evaluate.load("rouge")
bertscore = hf_evaluate.load("bertscore")

with open(DATASET_PATH) as f:
    dataset = json.load(f)

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
llm = Ollama(base_url=OLLAMA_HOST, model="llama3.2:3b", temperature=0.1)
prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

ablation_results = []

for cfg in CONFIGS:
    label = f"chunk={cfg['chunk_size']} overlap={cfg['chunk_overlap']} top_k={cfg['top_k']}"
    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"{'='*60}")

    # Build fresh in-memory vectorstore for this config
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " "],
    )

    all_chunks, all_metas = [], []
    for fname in sorted(os.listdir(CLEAN_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(CLEAN_DIR, fname), encoding="utf-8") as f:
            doc = json.load(f)
        chunks = splitter.split_text(doc["text"])
        metas  = [{"source": doc["name"], "url": doc["url"]} for _ in chunks]
        all_chunks.extend(chunks)
        all_metas.extend(metas)
    print(f"Total chunks: {len(all_chunks)}")

    # In-memory store (no persistence — keeps configs isolated)
    vectorstore = Chroma(
        collection_name=f"ablation_{cfg['chunk_size']}_{cfg['top_k']}",
        embedding_function=embedder,
    )
    vectorstore.add_texts(texts=all_chunks, metadatas=all_metas)
    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg["top_k"]})

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    predictions, references = [], []
    for item in dataset:
        pred = chain.invoke(item["question"])
        predictions.append(pred)
        references.append(item["gold_answer"])
        print(f"  [{item['id']:02d}] done")

    r = rouge.compute(predictions=predictions, references=references)
    b = bertscore.compute(predictions=predictions, references=references, lang="en")
    avg_b = round(sum(b["f1"]) / len(b["f1"]), 4)

    result = {
        "config": label,
        "chunk_size":    cfg["chunk_size"],
        "chunk_overlap": cfg["chunk_overlap"],
        "top_k":         cfg["top_k"],
        "n_chunks":      len(all_chunks),
        "rouge1":        round(r["rouge1"], 4),
        "rouge2":        round(r["rouge2"], 4),
        "rougeL":        round(r["rougeL"], 4),
        "bertscore_f1":  avg_b,
    }
    ablation_results.append(result)
    print(f"\nROUGE-L: {result['rougeL']}  BERTScore: {result['bertscore_f1']}")

print("\n\n=== Ablation Summary ===")
print(f"{'Config':<45} {'Chunks':>6} {'ROUGE-1':>8} {'ROUGE-L':>8} {'BERTScore':>10}")
print("-" * 82)
for r in ablation_results:
    print(f"{r['config']:<45} {r['n_chunks']:>6} {r['rouge1']:>8} {r['rougeL']:>8} {r['bertscore_f1']:>10}")

with open("eval/ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=2)
print("\nSaved to eval/ablation_results.json")
