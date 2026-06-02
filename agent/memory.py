import os
import json
import numpy as np
import faiss
import threading

# Standard list of 128 travel concepts to create a dense semantic representation
TRAVEL_CONCEPTS = [
    "budget", "cheap", "affordable", "backpack", "economy", "free", "discount", "save",
    "luxury", "expensive", "premium", "5-star", "boutique", "lavish", "classy", "upscale",
    "pet", "dog", "cat", "animal", "pets", "furry", "puppy", "canine",
    "vegan", "vegetarian", "plant-based", "halal", "kosher", "gluten-free", "allergy", "organic",
    "anime", "otaku", "manga", "gaming", "cosplay", "geek", "arcade", "nintendo",
    "art", "museum", "gallery", "exhibition", "painting", "sculpture", "design", "creative",
    "culture", "history", "historical", "temple", "shrine", "church", "monument", "palace",
    "nature", "garden", "park", "hike", "hiking", "mountain", "outdoor", "forest",
    "beach", "island", "coastal", "resort", "sea", "ocean", "swim", "sunbath",
    "shopping", "fashion", "mall", "market", "souvenir", "store", "boutique", "vintage",
    "kids", "family", "children", "child", "playground", "theme-park", "zoo", "aquarium",
    "romantic", "honeymoon", "couple", "date", "scenic", "view", "cozy", "quiet",
    "adventure", "active", "sport", "thrill", "climb", "rafting", "diving", "exotic",
    "relax", "spa", "wellness", "massage", "onsen", "pool", "sauna", "quiet",
    "nightlife", "bar", "club", "pub", "wine", "beer", "cocktail", "party", "lounge",
    "sushi", "ramen", "pastry", "croissant", "cafe", "coffee", "tea", "hawker", "foodie",
    "flight", "hotel", "travel", "trip", "vacation", "explore", "sightseeing", "guide"
]

class ShortTermMemory:
    """Manages active conversation context grouped by city/topic."""
    def __init__(self):
        self.chats = {"default": []}
        self.active_city = "default"
        self.active_parameters = {}

    def update_parameters(self, new_params: dict):
        """Merges new LLM parameters with active state so they aren't forgotten."""
        for k, v in new_params.items():
            if v is not None:
                # Never overwrite a valid parameter with None
                self.active_parameters[k] = v

    def get_parameters(self) -> dict:
        return dict(self.active_parameters)

    def set_active_city(self, city: str):
        if city:
            city_key = city.lower().strip()
            self.active_city = city_key
            if city_key not in self.chats:
                self.chats[city_key] = []

    def add_message(self, role: str, content: str, city: str = None):
        if city:
            self.set_active_city(city)
        self.chats[self.active_city].append({"role": role, "content": content})

    def get_context(self, city: str = None) -> list:
        if city:
            city_key = city.lower().strip()
            return self.chats.get(city_key, [])
        return self.chats.get(self.active_city, [])

    def clear(self):
        self.chats = {"default": []}
        self.active_city = "default"
        self.active_parameters = {}


class FAISSPreferenceMemory:
    """Manages long-term user preferences using a FAISS vector database."""
    def __init__(self, storage_dir: str = "data/memory"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.index_path = os.path.join(self.storage_dir, "preferences.index")
        self.meta_path = os.path.join(self.storage_dir, "preferences.json")
        
        self.dim = len(TRAVEL_CONCEPTS)
        self.preferences = []
        self.lock = threading.Lock()
        
        # Initialize or load FAISS Index
        with self.lock:
            if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        self.preferences = json.load(f)
                except Exception as e:
                    print(f"Error loading FAISS memory: {e}. Reinitializing...")
                    self._init_empty_index()
            else:
                self._init_empty_index()

    def _init_empty_index(self):
        # IndexFlatIP uses Inner Product, which is Cosine Similarity when vectors are normalized
        self.index = faiss.IndexFlatIP(self.dim)
        self.preferences = []

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Embeds text into a 128-dimensional dense travel concept vector."""
        text_lower = text.lower()
        vector = np.zeros(self.dim, dtype=np.float32)
        
        # Count keyword occurrences and weight them
        for i, concept in enumerate(TRAVEL_CONCEPTS):
            # Simple keyword matching with word boundary / substring checks
            count = text_lower.count(concept)
            if count > 0:
                vector[i] = count
                
        # Normalize the vector to unit length (L2 norm) for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            # Fallback to a uniform vector if no travel terms match
            vector = np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)
            
        return np.expand_dims(vector, axis=0)

    def add_preference(self, preference_text: str):
        """Adds a preference to the vector database and metadata store."""
        if not preference_text.strip():
            return
        
        with self.lock:
            # Check if already exists to avoid duplication
            if preference_text in self.preferences:
                return
                
            vector = self._text_to_vector(preference_text)
            
            # Add vector to FAISS Index
            self.index.add(vector)
            
            # Save textual metadata
            self.preferences.append(preference_text)
            self._save_to_disk()

    def query_preferences(self, query: str, top_k: int = 3, threshold: float = 0.1) -> list:
        """Queries FAISS vector database for matching user preferences."""
        with self.lock:
            if not self.preferences:
                return []
                
            query_vector = self._text_to_vector(query)
            
            # Search the FAISS index
            k = min(top_k, len(self.preferences))
            distances, indices = self.index.search(query_vector, k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(self.preferences) and dist >= threshold:
                    results.append({
                        "preference": self.preferences[idx],
                        "score": float(dist)
                    })
            return results

    def get_all_preferences(self) -> list:
        """Returns a copy of all stored preferences to prevent concurrency read mutation errors."""
        with self.lock:
            return list(self.preferences)

    def delete_preference(self, index: int):
        """Deletes a preference by index. Requires rebuilding the FAISS index."""
        with self.lock:
            if 0 <= index < len(self.preferences):
                self.preferences.pop(index)
                # Rebuild index
                self.index = faiss.IndexFlatIP(self.dim)
                for pref in self.preferences:
                    vector = self._text_to_vector(pref)
                    self.index.add(vector)
                self._save_to_disk()

    def clear(self):
        """Wipes out all long-term preference memory."""
        with self.lock:
            self._init_empty_index()
            self._save_to_disk()

    def _save_to_disk(self):
        """Persists the FAISS index and metadata to disk. Assumes self.lock is already held."""
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.preferences, f, ensure_ascii=False, indent=2)
