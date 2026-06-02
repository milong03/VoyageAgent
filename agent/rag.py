import os
import json
import re

# Additional deep travel blog snippets to support multi-hop retrieval
TRAVEL_BLOGS = {
    "tokyo_sushi_guide": "Local Guide: Tokyo's best sushi isn't always in luxury Michelin restaurants. In Tsukiji Outer Market or Ginza, look for standing sushi bars (Tachigui) where you get premium cuts like Otoro (fatty tuna) and Uni (sea urchin) for a fraction of the cost. Always dip the fish-side, never the rice, into soy sauce, and eat it in a single bite.",
    "tokyo_pet_guide": "Pet Owner Blog: Tokyo is becoming highly pet-friendly, especially in neighborhoods like Yoyogi and Shibuya. For dining, cafes like 'Dumbo Doughnuts' or outdoor terraces in 'Girafe' welcome dogs. When taking dogs on trains, they must be in an enclosed carrier bag (pet buggy or bag) weighing under 10kg, and you must purchase a personal luggage ticket (around 290 JPY).",
    "nakamise_dori_food": "Travel Blog: When walking through Nakamise-dori at Senso-ji, you must try 'Ningyo-yaki' (sponge cakes shaped like lanterns and bells stuffed with sweet red bean paste). Another must-try is 'Kibi-dango' (sweet millet dumplings coated in toasted soybean flour). Important: Eating while walking is considered rude in Japan. Eat your snacks directly in front of the stall where you bought them!",
    "paris_cafe_etiquette": "Parisian Guide: In Paris, if you sit at a outdoor table, you will pay a slightly higher price (terrace price) than if you stand at the bar. If your dog is well-behaved, they are almost always allowed to sit next to your chair on the outdoor terrace. To order like a local, ask for 'un cafe' (espresso) or 'un cafe au lait' (coffee with milk, usually only drunk at breakfast).",
    "sg_hawker_rules": "Singapore Foodie: When visiting hawker centres like Chinatown Complex or Maxwell, remember the golden rule: 'Chope' your seat. Locals place a packet of tissue paper on a table to reserve it before ordering food. Order the legendary Hainanese Chicken Rice, Char Kway Teow, and Laksa. Most hawker tables are outdoor/semi-covered but pets are generally not allowed inside hawker halls due to hygiene regulations.",
    "sg_tanjong_beach_tips": "Sentosa Guide: Tanjong Beach Club is the ultimate dog hangout on weekends. You can reserve a daybed, splash in the pool, and let your dog run on the sand. Bring dog shampoo because there are public washing stations near the restrooms where you can rinse the salt and sand off your dog's paws."
}

class MultiHopRAG:
    """Implements Multi-Hop Retrieval-Augmented Generation to fetch deep, multi-layered information."""
    
    def __init__(self, db_path: str = "data/attractions.json"):
        self.db_path = db_path
        self._load_data()

    def _load_data(self):
        # Load main attractions
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.db = json.load(f)
        else:
            self.db = {"cities": {}}
            
        # Compile all available RAG documents
        self.corpus = {}
        
        # Add attraction RAG details
        for city_key, city_data in self.db.get("cities", {}).items():
            city_name = city_data.get("name")
            for attr in city_data.get("attractions", []):
                doc_id = attr.get("id")
                self.corpus[doc_id] = {
                    "title": f"{city_name} - {attr.get('name')}",
                    "text": attr.get("rag_details"),
                    "tags": attr.get("tags", []),
                    "city": city_key
                }
                
        # Add travel blog snippets
        for blog_id, blog_text in TRAVEL_BLOGS.items():
            # Associate blog with cities based on keywords
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

    def _search_corpus(self, query: str, city_filter: str = None) -> list:
        """Searches the local corpus for matching snippets using a simple term-frequency matcher."""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        if not query_terms:
            return []
            
        results = []
        for doc_id, doc in self.corpus.items():
            # If city filter is specified and doesn't match, skip
            if city_filter and doc.get("city") != city_filter:
                continue
                
            doc_text = doc["text"].lower()
            doc_title = doc["title"].lower()
            
            # Simple scoring: count term matches
            score = 0
            for term in query_terms:
                if len(term) < 3: # Skip very short terms
                    continue
                # Give higher weight to matches in the title
                score += doc_title.count(term) * 5
                score += doc_text.count(term)
                
            if score > 0:
                results.append({
                    "id": doc_id,
                    "title": doc["title"],
                    "text": doc["text"],
                    "score": score
                })
                
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def execute_rag(self, query: str, city: str = None) -> dict:
        """Performs a Multi-Hop Retrieval process.
        
        Hop 1: Searches the corpus for the primary terms in the user query.
        Hop 2: Identifies related entities/terms in the Hop 1 results (e.g. 'Hachiko', 'Nakamise-dori', 'sushi') 
               and performs a secondary query to fetch deep local blogs or tip sheets.
        """
        city_key = city.lower().strip() if city else None
        
        log = []
        log.append(f"Starting Multi-Hop RAG for query: '{query}' (City: {city or 'Any'})")
        
        # --- HOP 1: Primary Retrieval ---
        hop1_results = self._search_corpus(query, city_filter=city_key)
        
        if not hop1_results:
            log.append("Hop 1: No direct matches found. Attempting broader search.")
            # Try searching without city filter
            hop1_results = self._search_corpus(query)
            
        top_hop1 = hop1_results[:2] # Take top 2 primary docs
        log.append(f"Hop 1 retrieved {len(hop1_results)} documents. Top matches: " + 
                   ", ".join([d["title"] for d in top_hop1]) if top_hop1 else "None")
        
        # --- HOP 2: Secondary Retrieval (Follow-up hops) ---
        # Extract potential multi-hop keywords/entities from Hop 1 content
        secondary_queries = []
        hop2_results = []
        
        if top_hop1:
            combined_text = " ".join([d["text"] for d in top_hop1]).lower()
            
            # Look for specific sub-topics mentioned in the primary documents
            entities_to_check = {
                "hachiko": "Hachiko statue",
                "nakamise": "Nakamise-dori street snacks",
                "sushi": "Tokyo local sushi guide",
                "pet": "Tokyo pet travel guide",
                "cafe": "Paris cafe etiquette",
                "hawker": "Singapore hawker culture",
                "tanjong": "Sentosa Tanjong beach dog guidelines"
            }
            
            for key, search_phrase in entities_to_check.items():
                if key in combined_text or key in query.lower():
                    secondary_queries.append(search_phrase)
            
            # If we found secondary entities, search for them!
            if secondary_queries:
                log.append(f"Hop 2: Extracted sub-entities {secondary_queries} from primary documents. Performing second-hop queries.")
                for sec_query in secondary_queries:
                    sec_results = self._search_corpus(sec_query, city_filter=city_key)
                    for r in sec_results[:1]: # Take the best match for each secondary entity
                        if r["id"] not in [d["id"] for d in top_hop1]:
                            hop2_results.append(r)
                            log.append(f"Hop 2 match found: '{r['title']}' for query '{sec_query}'")
            else:
                log.append("Hop 2: No secondary sub-entities detected. Skipping second hop.")
        else:
            log.append("Hop 2: Skipping secondary hop due to empty primary results.")
            
        # Combine all retrieved contexts
        all_retrieved = top_hop1 + hop2_results
        
        context_blocks = []
        for doc in all_retrieved:
            context_blocks.append(f"[{doc['title']}]\n{doc['text']}")
            
        context_str = "\n\n".join(context_blocks)
        
        return {
            "query": query,
            "context": context_str,
            "retrieved_docs": [d["title"] for d in all_retrieved],
            "hops_log": log
        }
