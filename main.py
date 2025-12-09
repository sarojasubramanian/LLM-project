# -*- coding: utf-8 -*-
import json
from src.scrape_events import scrape_paginated, to_rag_docs

# --- EDIT THESE ---
START_URL   = "https://calendar.northeastern.edu/calendar/upcoming?experience=&order=date"
MAX_PAGES   = 6
FOLLOW_LINKS = False          # set False to skip opening each event detail page
OUT_EVENTS  = "events.json"
OUT_RAG     = "rag_docs.json"
# ------------------

def run():
    events = scrape_paginated(START_URL, max_pages=MAX_PAGES, follow_links=FOLLOW_LINKS)
    with open(OUT_EVENTS, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(events)} events → {OUT_EVENTS}")

    docs = to_rag_docs(events)
    with open(OUT_RAG, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(docs)} RAG docs → {OUT_RAG}")

if __name__ == "__main__":
    run()