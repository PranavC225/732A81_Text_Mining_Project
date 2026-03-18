import requests
from bs4 import BeautifulSoup
import json, os

SOURCES = [
    {"name": "liu_accommodation",   "url": "https://liu.se/en/article/accommodation"},
    {"name": "liu_student_faq",     "url": "https://liu.se/en/article/liu-student-accommodation"},
    {"name": "studentbostader",     "url": "https://www.studentbostader.se/en/find-apartments"},
    {"name": "kombo_goodtoknow",    "url": "https://bostad.karservice.se/en/pages/visit/100097"},
    {"name": "kombo_emergency",     "url": "https://bostad.karservice.se/en/pages/visit/100098"},
    {"name": "kombo_scam",          "url": "https://bostad.karservice.se/en/pages/visit/100099"},
    {"name": "hyresbostader_intl",  "url": "https://www.hyresbostader.se/artikel/international-students"},
]

HEADERS = {"User-Agent": "Mozilla/5.0 (research project, LiU course 732A81)"}
OUT_DIR = "data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

def scrape(source):
    r = requests.get(source["url"], headers=HEADERS, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    for tag in soup(["nav", "footer", "script", "style", "header", "form"]):
        tag.decompose()

    main = (soup.find("main") or
            soup.find("article") or
            soup.find("div", class_=lambda c: c and "content" in c.lower()) or
            soup.body)

    text = main.get_text(separator="\n", strip=True)
    out = {"name": source["name"], "url": source["url"], "text": text}

    path = os.path.join(OUT_DIR, f"{source['name']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  Saved {path} ({len(text)} chars)")

if __name__ == "__main__":
    for s in SOURCES:
        try:
            print(f"Scraping {s['name']}...")
            scrape(s)
        except Exception as e:
            print(f"  FAILED {s['name']}: {e}")