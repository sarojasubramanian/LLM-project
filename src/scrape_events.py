# -*- coding: utf-8 -*-
import hashlib, json, re, time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

UA = "EventScraper/1.0 (+contact@example.com)"
# Separate connect/read timeouts to avoid long hangs
TIMEOUT = (5, 20)  # 5s to connect, 20s to read
SLEEP_PAGE = 1.2  # delay between listing pages
SLEEP_DETAIL = 1.2 # delay between detail pages

# a single session with retries/backoff
_session = None
def _get_session():
    global _session
    if _session is None:
        s = requests.Session()
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        s.headers.update({
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        })
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.mount("http://", HTTPAdapter(max_retries=retries))
        _session = s
    return _session

def fetch(url: str, accept: str = "text/html") -> str:
    s = _get_session()
    headers = {"Accept": accept, "Referer": url}
    r = s.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def absolute(base: str, href: Optional[str]) -> Optional[str]:
    return urljoin(base, href.strip()) if href else None

def text_or_none(el) -> Optional[str]:
    if not el: return None
    t = (el.get_text(" ", strip=True) or "").strip()
    return t or None

def parse_date(val: Optional[str]) -> Optional[str]:
    if not val: return None
    s = val.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):  # already ISO-ish
        return s
    try:
        return dtparser.parse(s, fuzzy=True).isoformat()
    except Exception:
        return s

# ----------------------------- JSON-LD (@type: Event) -----------------------------
def events_from_jsonld(soup: BeautifulSoup, page_url: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = s.string or s.get_text() or ""
        if not raw.strip(): 
            continue
        for blob in _split_json_candidates(raw):
            try:
                data = json.loads(blob)
            except Exception:
                continue
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("@graph"), list):
                    for g in item["@graph"]:
                        out.extend(_event_from_ld(g, page_url))
                else:
                    out.extend(_event_from_ld(item, page_url))
    return out

def _split_json_candidates(raw: str) -> List[str]:
    try:
        json.loads(raw); return [raw]
    except Exception:
        pass
    chunks = re.findall(r"\{.*?\}(?=\s*(?:\{|\Z))", raw, flags=re.S)
    return chunks or [raw]

def _first_string(v):
    """Return a human string for v (str | dict | list | None)."""
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, dict):
        # Prefer common text-like fields
        return _first_string(v.get("name") or v.get("url") or v.get("@id") or v.get("text"))
    if isinstance(v, list):
        for it in v:
            s = _first_string(it)
            if s:
                return s
    return None

def _format_address(addr):
    """Format address that might be dict | str | list."""
    if addr is None:
        return None
    if isinstance(addr, str):
        return addr.strip() or None
    if isinstance(addr, dict):
        parts = [
            addr.get("streetAddress"),
            addr.get("addressLocality"),
            addr.get("addressRegion"),
            addr.get("postalCode"),
            addr.get("addressCountry"),
        ]
        parts = [str(p).strip() for p in parts if p]
        return ", ".join(parts) if parts else None
    if isinstance(addr, list):
        # return the first renderable address
        for a in addr:
            s = _format_address(a)
            if s:
                return s
    return None

def _first_urlish(v):
    """Extract a usable URL from str | dict | list."""
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, dict):
        return _first_urlish(v.get("url") or v.get("@id") or v.get("contentUrl"))
    if isinstance(v, list):
        for it in v:
            u = _first_urlish(it)
            if u:
                return u
    return None

def _event_from_ld(obj: Dict[str, Any], page_url: str) -> List[Dict[str, Any]]:
    def is_event(o: Dict[str, Any]) -> bool:
        t = o.get("@type")
        if isinstance(t, list):
            return any(str(x).lower() == "event" for x in t)
        return str(t).lower() == "event"

    if not isinstance(obj, dict) or not is_event(obj):
        return []

    # Core fields (accept str | dict | list gracefully)
    title = (obj.get("name") or "") if isinstance(obj.get("name"), str) else _first_string(obj.get("name"))
    title = (title or "").strip()

    desc = _first_string(obj.get("description"))
    url  = _first_urlish(obj.get("url")) or page_url
    img  = _first_urlish(obj.get("image"))

    # Dates (may be strings or arrays)
    start = obj.get("startDate")
    end   = obj.get("endDate")
    if isinstance(start, list): start = start[0] if start else None
    if isinstance(end, list):   end   = end[0]   if end   else None
    start = parse_date(start)
    end   = parse_date(end)

    # Location & address (can be str | dict | list)
    loc = obj.get("location")
    if isinstance(loc, list) and loc:
        loc = loc[0]
    venue_name = _first_string(loc) if not isinstance(loc, dict) else _first_string(loc.get("name")) or _first_string(loc.get("url"))
    address = None
    if isinstance(loc, dict):
        address = _format_address(loc.get("address"))
    venue = venue_name or address

    return [make_event_record(
        title=title,
        url=url,
        source_url=page_url,
        image=img,
        start=start,
        end=end,
        venue=venue,
        description=desc,
    )]

# ----------------------------- Heuristic HTML fallback -----------------------------
def events_from_html(soup: BeautifulSoup, page_url: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    cards = soup.select(
        '[itemtype*="Event" i], .event, .events, .event-card, .event-item, li.event, article.event'
    ) or soup.select("article, li, .card, .tile, .list-item")

    for c in cards:
        title = text_or_none(c.select_one("h1, h2, h3, .title, .event-title, a[title]")) or text_or_none(c.select_one("a"))
        t = c.select_one("time[datetime]") or c.select_one("time")
        start = parse_date(t.get("datetime") if t and t.has_attr("datetime") else text_or_none(t)) \
                or parse_date(text_or_none(c.select_one(".date, .event-date, .when, .start")))
        venue = text_or_none(c.select_one(".location, .venue, address"))
        a = c.select_one("a[href]")
        url = absolute(page_url, a.get("href")) if a else page_url
        img_el = c.select_one("img[src], img[data-src], img[data-original]")
        img_src = (img_el.get("src") or img_el.get("data-src") or img_el.get("data-original")) if img_el else None
        img = absolute(page_url, img_src) if img_src else None

        if title or start:
            events.append(make_event_record(title, url, page_url, img, start, None, venue, None))
    return dedupe(events)

# ----------------------------- Utils -----------------------------
def make_event_record(title: Optional[str], url: Optional[str], source_url: str,
                      image: Optional[str], start: Optional[str], end: Optional[str],
                      venue: Optional[str], description: Optional[str]) -> Dict[str, Any]:
    title = (title or "").strip()
    url = url or source_url
    key = f"{title}|{start}|{url}"
    eid = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return {
        "id": eid,
        "title": title or None,
        "start": start,
        "end": end,
        "url": url,
        "image": image,
        "venue": venue or None,
        "description": description or None,
        "source_url": source_url,
        "last_seen": datetime.utcnow().isoformat() + "Z",
    }

def dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for r in rows:
        k = (r.get("title") or "", r.get("start") or "", r.get("url") or "")
        if k in seen: continue
        seen.add(k); out.append(r)
    return out

def enrich_descriptions(rows: List[Dict[str, Any]], max_follow: int = 20) -> List[Dict[str, Any]]:
    enriched = []
    for i, r in enumerate(rows):
        if i >= max_follow:
            enriched.append(r); continue
        try:
            html = fetch(r["url"])
            soup = BeautifulSoup(html, "lxml")
            body = soup.select_one("article, .event-body, .description, .event-description, .content, main")
            desc = text_or_none(body) or r.get("description")
            enriched.append({**r, "description": desc})
            time.sleep(SLEEP_DETAIL)
        except Exception:
            enriched.append(r)
    return enriched

# ----------------------------- Pagination helpers -----------------------------
def next_page_url(current_url: str, soup: BeautifulSoup) -> Optional[str]:
    rel_next = soup.select_one('a[rel="next"], link[rel="next"]')
    if rel_next and rel_next.get("href"):
        return urljoin(current_url, rel_next["href"])

    cand = (
        soup.select_one('a.next, a.next-page, a[aria-label="Next"], a[title*="Next" i]') or
        soup.find("a", string=lambda t: t and t.strip().lower().startswith("next"))
    )
    if cand and cand.get("href"):
        return urljoin(current_url, cand["href"])

    parsed = urlparse(current_url)
    m = re.match(r"^(.*?/upcoming)(?:/(\d+))?$", parsed.path)
    if m:
        base_path, page_str = m.group(1), m.group(2)
        next_page_num = 2 if page_str is None else int(page_str) + 1
        next_path = f"{base_path}/{next_page_num}"
        return urlunparse(parsed._replace(path=next_path))

    qs = parse_qs(parsed.query)
    if "page" in qs:
        try:
            n = int(qs["page"][0]) + 1
            qs["page"] = [str(n)]
            return urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))
        except Exception:
            pass

    return None

# ----------------------------- Public API -----------------------------
def scrape_listing_page(url: str, follow_links: bool = False) -> List[Dict[str, Any]]:
    html = fetch(url)
    soup = BeautifulSoup(html, "lxml")
    events = events_from_jsonld(soup, url) or events_from_html(soup, url)
    if follow_links and events:
        events = enrich_descriptions(events, max_follow=50)
    return dedupe(events)

def scrape_paginated(url: str,
                     max_pages: int = 30,
                     follow_links: bool = False,
                     max_total_events: int = 800,
                     hard_timeout_seconds: int = 300) -> List[Dict[str, Any]]:
    """
    Crawl listing pages via Next links until none found, or caps/timeouts hit.
    """
    start_ts = time.time()
    all_events: List[Dict[str, Any]] = []
    seen_pages = set()
    page_idx = 0

    while url and page_idx < max_pages:
        if time.time() - start_ts > hard_timeout_seconds:
            print(f"[stop] hit hard timeout ({hard_timeout_seconds}s)")
            break
        if url in seen_pages:
            print(f"[stop] already saw {url} (cycle detected)")
            break
        seen_pages.add(url)
        page_idx += 1

        print(f"[page {page_idx}] GET {url}")
        html = fetch(url)
        soup = BeautifulSoup(html, "lxml")

        # Extract events (JSON-LD first)
        page_events = events_from_jsonld(soup, url) or events_from_html(soup, url)
        print(f"[page {page_idx}] found {len(page_events)} events")

        if follow_links and page_events:
            # Only enrich a few per page to keep things snappy
            page_events = enrich_descriptions(page_events, max_follow=20)

        all_events.extend(page_events)
        if len(all_events) >= max_total_events:
            print(f"[stop] reached max_total_events={max_total_events}")
            break

        nxt = next_page_url(url, soup)
        if not nxt:
            print("[done] no next page")
            break
        if nxt == url:
            print("[stop] next equals current (bad pagination); stopping to avoid loop")
            break

        url = nxt
        time.sleep(SLEEP_PAGE)

    all_events = dedupe(all_events)
    print(f"[total] {len(all_events)} events after de-dup")
    return all_events

def to_rag_docs(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []
    for e in events:
        parts = [
            e.get("title") or "",
            f"Start: {e.get('start')}" if e.get("start") else "",
            f"End: {e.get('end')}" if e.get("end") else "",
            f"Venue: {e.get('venue')}" if e.get("venue") else "",
            e.get("description") or ""
        ]
        docs.append({
            "id": e["id"],
            "text": "\n".join([p for p in parts if p]).strip(),
            "metadata": {
                "url": e.get("url"), "image": e.get("image"),
                "source_url": e.get("source_url"), "start": e.get("start"),
                "end": e.get("end"), "venue": e.get("venue"),
                "title": e.get("title"), "last_seen": e.get("last_seen")
            }
        })
    return docs