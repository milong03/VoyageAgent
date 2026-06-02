import os
import json

class TravelTools:
    """Implements tools that the travel agent can execute to collect real-time data."""
    
    def __init__(self, db_path: str = "data/attractions.json"):
        self.db_path = db_path
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.db = json.load(f)
        else:
            self.db = {"cities": {}}

    def get_weather(self, city: str) -> dict:
        """Gets weather forecast for a target city."""
        city_key = city.strip().lower()
        cities = self.db.get("cities", {})
        
        if city_key not in cities:
            # Fallback for unrecognized cities
            return {
                "error": f"City '{city}' not found in attraction database. Standard forecast: Mostly sunny, 22°C.",
                "city": city,
                "summary": "Mostly sunny with a pleasant breeze.",
                "avg_temp_c": 22,
                "humidity": "50%",
                "chance_of_rain": "10%"
            }
            
        city_data = cities[city_key]
        weather = city_data.get("weather_forecast", {})
        return {
            "city": city_data.get("name"),
            "summary": weather.get("summary"),
            "avg_temp_c": weather.get("avg_temp_c"),
            "humidity": weather.get("humidity"),
            "chance_of_rain": weather.get("chance_of_rain")
        }

    def search_attractions(self, city: str, budget_usd: float = None, tags: list = None, pet_friendly: bool = False) -> list:
        """Filters attractions based on location, budget, categories, and pet friendliness."""
        city_key = city.strip().lower()
        cities = self.db.get("cities", {})
        
        if city_key not in cities:
            return []
            
        city_data = cities[city_key]
        attractions = city_data.get("attractions", [])
        
        filtered = []
        for attr in attractions:
            # 1. Budget check (cost_usd <= budget_usd)
            if budget_usd is not None and attr.get("cost_usd", 0) > budget_usd:
                continue
                
            # 2. Pet-friendliness check
            attr_tags = [t.lower() for t in attr.get("tags", [])]
            if pet_friendly and "pet-friendly" not in attr_tags:
                continue
                
            # 3. Tags filtering (matches any of the requested tags)
            if tags:
                requested_tags = [t.lower() for t in tags]
                # Check if there is intersection
                if not any(t in attr_tags for t in requested_tags):
                    continue
                    
            filtered.append({
                "id": attr.get("id"),
                "name": attr.get("name"),
                "description": attr.get("description"),
                "cost_usd": attr.get("cost_usd"),
                "duration_hours": attr.get("duration_hours"),
                "tags": attr.get("tags"),
                "best_time": attr.get("best_time")
            })
            
        return filtered

    def get_accommodation(self, city: str, max_price_usd: float = None, pet_friendly: bool = False) -> dict:
        """Gets hotel options and estimated flight costs for a target city."""
        city_key = city.strip().lower()
        cities = self.db.get("cities", {})
        
        if city_key not in cities:
            return {"error": f"City '{city}' not found in accommodation database."}
            
        city_data = cities[city_key]
        hotels = city_data.get("hotels", [])
        flights = city_data.get("flights", {})
        
        filtered_hotels = []
        for hotel in hotels:
            hotel_tags = [t.lower() for t in hotel.get("tags", [])]
            
            # Budget filter
            if max_price_usd is not None and hotel.get("price_per_night_usd", 0) > max_price_usd:
                continue
                
            # Pet friendly filter
            if pet_friendly and "pet-friendly" not in hotel_tags:
                continue
                
            filtered_hotels.append(hotel)
            
        return {
            "city": city_data.get("name"),
            "hotels": filtered_hotels,
            "flights": flights
        }

    def web_search(self, query: str) -> str:
        """Simulates multi-hop search to retrieve background context for an attraction or city detail."""
        query_lower = query.lower()
        cities = self.db.get("cities", {})
        
        # Traverse attractions to find matches
        for city_name, city_data in cities.items():
            for attr in city_data.get("attractions", []):
                # If attraction name or id is in query
                if attr.get("name").lower() in query_lower or attr.get("id").lower() in query_lower:
                    return f"Source [Wiki/Blog: {attr.get('name')}]: {attr.get('rag_details')}"
                    
        # General matching
        matches = []
        for city_name, city_data in cities.items():
            if city_name in query_lower or city_data.get("name").lower() in query_lower:
                for attr in city_data.get("attractions", []):
                    matches.append(f"- {attr.get('name')}: {attr.get('description')}")
                    
        if matches:
            return f"Search Results for '{query}':\n" + "\n".join(matches)
            
        return f"Search results for '{query}' came up empty. Local travel tip: When traveling, check opening hours ahead of time and keep local currency on hand."
