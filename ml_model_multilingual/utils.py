# ml_model_multilingual/utils.py
import re
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from typing import List, Dict

def clean_text(query: str) -> str:
    # Keep basic alphanumerics and spaces for site search
    return re.sub(r"[^a-zA-Z0-9\s]", " ", query).strip()

def site_search_google(query: str, site: str, num: int = 2) -> List[str]:
    """
    Use google site-specific search (via googlesearch-python).
    Returns a list of URLs (may be empty).
    """
    q = f"{query} site:{site}"
    try:
        results = [u for u in search(q, num_results=num, lang="en")]
        return results
    except Exception:
        return []

def fetch_page_title_snippet(url: str) -> Dict[str, str]:
    """Fetch title and a short snippet from a URL (best-effort)."""
    try:
        r = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string.strip() if soup.title else url
        # snippet: first <p> text
        p = soup.find("p")
        snippet = p.get_text().strip() if p else ""
        return {"url": url, "title": title, "snippet": snippet}
    except Exception:
        return {"url": url, "title": url, "snippet": ""}

def find_fact_checks(query: str, sites: List[str]) -> List[Dict]:
    """
    For each trusted site, try a site-specific google search and then fetch basic metadata.
    Returns a list of dicts: {name, url, title, snippet}
    """
    q = clean_text(query)
    found = []
    for site in sites:
        try:
            urls = site_search_google(q, site, num=2)
            for u in urls:
                meta = fetch_page_title_snippet(u)
                found.append({"name": site, "url": meta["url"], "title": meta["title"], "snippet": meta["snippet"]})
            if found:
                # if we found for this site, move on to next site
                continue
        except Exception:
            continue
    return found
