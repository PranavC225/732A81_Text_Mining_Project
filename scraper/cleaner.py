import json, os, re

RAW_DIR   = "data/raw"
CLEAN_DIR = "data/clean"
os.makedirs(CLEAN_DIR, exist_ok=True)

MIN_LINE_LEN = 30

SKIP_PATTERNS = [
    re.compile(r"^(share on|follow us|tags|quick links|edit|svenska|menu)", re.I),
    re.compile(r"^(facebook|linkedin|instagram|email|cookie)", re.I),
    re.compile(r"^\*"),
    re.compile(r"^photo credit", re.I),
    re.compile(r"^photographer", re.I),
]

def clean(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) < MIN_LINE_LEN:
            continue
        if any(p.match(line) for p in SKIP_PATTERNS):
            continue
        cleaned.append(line)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()

for fname in os.listdir(RAW_DIR):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(RAW_DIR, fname), encoding="utf-8") as f:
        doc = json.load(f)

    original_len = len(doc["text"])
    doc["text"] = clean(doc["text"])
    cleaned_len = len(doc["text"])

    out_path = os.path.join(CLEAN_DIR, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"  {fname}: {original_len} → {cleaned_len} chars")