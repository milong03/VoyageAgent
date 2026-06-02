"""
Multi-Hop RAG Pipeline
======================
Hop 1 — Query Wikipedia API for real, live city information (attractions, culture, history).
Hop 2 — Extract sub-entities from Hop 1 results and query Wikipedia again for each one,
         giving deep, layered context beyond a simple search.

Falls back to local corpus (attractions.json + curated travel blogs) when offline or rate-limited.
"""

import os
import re
import json
import urllib.request
import urllib.parse
import urllib.error

# ── Curated local fallback blogs (used when Wikipedia is unavailable) ──────────
TRAVEL_BLOGS = {
    "tokyo_sushi_guide": (
        "Local Guide: Tokyo's best sushi is often found at standing bars (Tachigui) "
        "in Tsukiji Outer Market or Ginza. Try Otoro (fatty tuna) and Uni (sea urchin). "
        "Always dip the fish-side into soy sauce, never the rice, and eat in one bite."
    ),
    "tokyo_pet_guide": (
        "Pet Owner Blog: Tokyo is increasingly pet-friendly in Yoyogi and Shibuya. "
        "Dogs on trains must be in an enclosed carrier under 10kg with a paid luggage ticket (~290 JPY)."
    ),
    "nakamise_dori_food": (
        "Travel Blog: At Nakamise-dori near Senso-ji, try Ningyo-yaki and Kibi-dango. "
        "Note: Eating while walking is considered rude in Japan — eat directly in front of the stall."
    ),
    "paris_cafe_etiquette": (
        "Parisian Guide: Outdoor terrace seating costs slightly more than bar seating. "
        "Well-behaved dogs are welcome on terraces. Order 'un cafe' for espresso like a local."
    ),
    "sg_hawker_rules": (
        "Singapore Foodie: At hawker centres, 'chope' your seat with a tissue packet before ordering. "
        "Must-try: Hainanese Chicken Rice, Char Kway Teow, Laksa. Pets not allowed inside hawker halls."
    ),
    "sg_tanjong_beach_tips": (
        "Sentosa Guide: Tanjong Beach Club is the top dog-friendly beach spot on weekends. "
        "Public washing stations near restrooms let you rinse your dog's paws after the beach."
    ),
}


class WikipediaFetcher:
    """Fetches real live content from the Wikipedia REST API. No API key required."""

    SEARCH_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
    TIMEOUT = 6

    def search(self, query: str, num_results: int = 3) -> list[dict]:
        """Search Wikipedia for a query, return list of {title, snippet}."""
        params = urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "utf8": 1,
            "format": "json"
        })
        url = f"{self.SEARCH_URL}?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VoyageAgent/1.0"})
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("query", {}).get("search", [])
                return [
                    {
                        "title": r["title"],
                        "snippet": re.sub(r"<[^>]+>", "", r.get("snippet", ""))
                    }
                    for r in results
                ]
        except Exception:
            return []

    def get_summary(self, title: str) -> str:
        """Fetch the full plain-text summary for a Wikipedia article."""
        encoded = urllib.parse.quote(title.replace(" ", "_"))
        url = self.SUMMARY_URL.format(encoded)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VoyageAgent/1.0"})
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
                return data.get("extract", "")[:1200]   # cap at 1200 chars per article
        except Exception:
            return ""


class MultiHopRAG:
    """
    Two-hop Retrieval-Augmented Generation pipeline.

    Hop 1 — Live Wikipedia search for the city + user interests.
    Hop 2 — Extract named entities (landmarks, cuisines, districts) from Hop 1 text
             and query Wikipedia for each, adding deep layered context.

    Falls back to the local corpus when Wikipedia is unreachable.
    """

    def __init__(self, db_path: str = "data/attractions.json"):
        self.db_path = db_path
        self.wiki = WikipediaFetcher()
        self._load_local_corpus()

    # ── Local corpus (fallback) ────────────────────────────────────────────────
    def _load_local_corpus(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.db = json.load(f)
        else:
            self.db = {"cities": {}}

        self.corpus = {}
        for city_key, city_data in self.db.get("cities", {}).items():
            city_name = city_data.get("name", city_key)
            for attr in city_data.get("attractions", []):
                doc_id = attr.get("id")
                self.corpus[doc_id] = {
                    "title": f"{city_name} — {attr.get('name')}",
                    "text": attr.get("rag_details", ""),
                    "tags": attr.get("tags", []),
                    "city": city_key
                }

        for blog_id, blog_text in TRAVEL_BLOGS.items():
            city = "unknown"
            if "tokyo" in blog_id or "nakamise" in blog_id:
                city = "tokyo"
            elif "paris" in blog_id:
                city = "paris"
            elif "sg_" in blog_id:
                city = "singapore"
            self.corpus[blog_id] = {
                "title": blog_id.replace("_", " ").title(),
                "text": blog_text,
                "tags": ["blog", "local-tip"],
                "city": city
            }

    def _search_local(self, query: str, city_filter: str = None) -> list:
        """Term-frequency search against the local corpus."""
        terms = set(re.findall(r"\b\w+\b", query.lower()))
        results = []
        for doc_id, doc in self.corpus.items():
            if city_filter and doc.get("city") != city_filter:
                continue
            text = (doc["text"] + " " + doc["title"]).lower()
            score = sum(text.count(t) for t in terms if len(t) >= 3)
            if score > 0:
                results.append({"id": doc_id, "title": doc["title"], "text": doc["text"], "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ── Entity extractor (drives Hop 2) ───────────────────────────────────────
    def _extract_named_entities(self, text: str, city: str) -> list[str]:
        """
        Pull specific landmark / cuisine / district names from retrieved text
        that are worth a second-hop Wikipedia lookup.
        """
        entities = []

        # Generic proper-noun pattern — Title Case sequences 2-4 words
        proper_nouns = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text)

        # Filter out the city itself and very common words
        skip = {city.lower(), "day", "the", "this", "that", "here", "there",
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                "january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december"}

        seen = set()
        for noun in proper_nouns:
            key = noun.lower()
            if key not in skip and key not in seen and len(noun) > 4:
                seen.add(key)
                entities.append(noun)
                if len(entities) >= 4:   # cap at 4 sub-entities for latency
                    break

        return entities

    # ── Main RAG execution ─────────────────────────────────────────────────────
    def execute_rag(self, query: str, city: str = None) -> dict:
        """
        Execute the full Multi-Hop RAG pipeline.

        Returns:
            context  (str)  : Full text block ready to inject into the LLM prompt
            hops_log (list) : Step-by-step retrieval trace for the UI
            retrieved_docs (list) : Source titles
        """
        city_key = city.lower().strip() if city else None
        log = []
        log.append(f"Starting Multi-Hop RAG | Query: '{query}' | City: {city or 'Any'}")

        hop1_docs = []
        hop2_docs = []

        # ── HOP 1 — Live Wikipedia search ────────────────────────────────────
        hop1_query = f"{city} {query}" if city else query
        log.append(f"Hop 1: Querying Wikipedia API for '{hop1_query}'...")

        wiki_results = self.wiki.search(hop1_query, num_results=3)

        if wiki_results:
            log.append(f"Hop 1: Wikipedia returned {len(wiki_results)} articles: "
                       + ", ".join(r["title"] for r in wiki_results))

            for result in wiki_results[:3]:
                full_text = self.wiki.get_summary(result["title"])
                if full_text:
                    hop1_docs.append({
                        "title": result["title"],
                        "text": full_text,
                        "source": "wikipedia"
                    })
                    log.append(f"Hop 1: Fetched full summary for '{result['title']}' ({len(full_text)} chars)")
        else:
            log.append("Hop 1: Wikipedia unreachable or no results. Falling back to local corpus.")
            local_results = self._search_local(query, city_filter=city_key)
            if not local_results:
                local_results = self._search_local(query)
            for doc in local_results[:3]:
                hop1_docs.append({"title": doc["title"], "text": doc["text"], "source": "local"})
            log.append(f"Hop 1 (local fallback): Retrieved {len(hop1_docs)} documents.")

        # ── HOP 2 — Named-entity expansion ───────────────────────────────────
        if hop1_docs:
            combined_hop1_text = " ".join(d["text"] for d in hop1_docs)
            entities = self._extract_named_entities(combined_hop1_text, city or "")
            log.append(f"Hop 2: Extracted named entities from Hop 1: {entities}")

            seen_titles = {d["title"].lower() for d in hop1_docs}

            for entity in entities:
                # Search Wikipedia specifically for this sub-entity + city
                entity_query = f"{entity} {city}" if city else entity
                log.append(f"Hop 2: Querying Wikipedia for sub-entity '{entity_query}'...")
                sub_results = self.wiki.search(entity_query, num_results=1)

                if sub_results:
                    top = sub_results[0]
                    if top["title"].lower() not in seen_titles:
                        sub_text = self.wiki.get_summary(top["title"])
                        if sub_text:
                            hop2_docs.append({
                                "title": top["title"],
                                "text": sub_text,
                                "source": "wikipedia"
                            })
                            seen_titles.add(top["title"].lower())
                            log.append(f"Hop 2: Fetched '{top['title']}' ({len(sub_text)} chars)")

            # Also pull matching local blog tips (pet rules, hawker etiquette, etc.)
            local_tips = self._search_local(query, city_filter=city_key)
            for tip in local_tips[:2]:
                if tip["title"].lower() not in seen_titles:
                    hop2_docs.append({"title": tip["title"], "text": tip["text"], "source": "local-blog"})
                    seen_titles.add(tip["title"].lower())
                    log.append(f"Hop 2: Added local blog tip: '{tip['title']}'")
        else:
            log.append("Hop 2: Skipping — no Hop 1 results available.")

        # ── Compile final context block ────────────────────────────────────────
        all_docs = hop1_docs + hop2_docs
        context_blocks = []
        for doc in all_docs:
            source_label = f"[Source: {doc['source']}]"
            context_blocks.append(f"### {doc['title']} {source_label}\n{doc['text']}")

        context_str = "\n\n".join(context_blocks)

        if not context_str.strip():
            context_str = (
                f"No specific RAG context was retrieved for '{city or query}'. "
                "Please rely on general travel knowledge and the tool data provided."
            )

        log.append(f"RAG complete. Total documents: {len(all_docs)} "
                   f"(Hop1: {len(hop1_docs)}, Hop2: {len(hop2_docs)})")

        return {
            "query": query,
            "context": context_str,
            "retrieved_docs": [d["title"] for d in all_docs],
            "hops_log": log
        }
