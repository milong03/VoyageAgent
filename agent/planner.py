import os
import re
import json
import google.generativeai as genai
from agent.memory import FAISSPreferenceMemory, ShortTermMemory
from agent.tools import TravelTools
from agent.rag import MultiHopRAG, TRAVEL_BLOGS

class TravelAgentPlanner:
    """The central Travel Agent that coordinates memory, RAG, tool execution, and planning."""
    
    def __init__(self, gemini_api_key: str = None):
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.memory = FAISSPreferenceMemory()
        self.short_memory = ShortTermMemory()
        self.tools = TravelTools()
        self.rag = MultiHopRAG()
        
        # Configure Gemini if API Key is available
        self.llm_available = False
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Test connection / model list
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.llm_available = True
                print("Gemini API successfully configured.")
            except Exception as e:
                print(f"Gemini API initialization failed: {e}. Falling back to Local simulation.")
                self.llm_available = False

    def update_api_key(self, api_key: str):
        """Allows updating the Gemini API key at runtime."""
        self.api_key = api_key
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.llm_available = True
            except Exception as e:
                self.llm_available = False
                raise e
        else:
            self.llm_available = False

    def plan_trip(self, user_query: str, session_id: str = "default") -> dict:
        """Executes the complete Plan-and-Execute loop to build a travel itinerary."""
        plan_logs = []
        plan_logs.append("Analyzing user query and session context...")
        
        # 1. Update Short-term memory
        self.short_memory.add_message("user", user_query)
        
        # 2. Extract active parameters from user query + active context
        extracted = self._extract_parameters(user_query)
        city = extracted["city"]
        budget = extracted["budget"]
        pet_friendly = extracted["pet_friendly"]
        interests = extracted["interests"]
        
        plan_logs.append(f"Extracted parameters -> City: {city or 'Unknown'}, Budget: {budget or 'Not specified'}, " +
                        f"Pet-Friendly: {pet_friendly}, Interests: {interests}")
        
        # 3. Retrieve Long-Term Memory from FAISS
        plan_logs.append("Querying FAISS Long-Term Memory for past session preferences...")
        matching_prefs = []
        if city:
            matching_prefs = self.memory.query_preferences(f"{city} {interests}", top_k=3)
            if matching_prefs:
                plan_logs.append(f"Retrieved {len(matching_prefs)} relevant preferences from FAISS: " + 
                               ", ".join([f"'{p['preference']}' (score: {p['score']:.2f})" for p in matching_prefs]))
                
                # Check retrieved memories to override parameters
                for p in matching_prefs:
                    pref_text = p["preference"].lower()
                    if "dog" in pref_text or "pet" in pref_text or "animal" in pref_text:
                        pet_friendly = True
                        plan_logs.append("Overriding parameter: 'pet_friendly=True' based on FAISS long-term memory!")
                    if "budget" in pref_text or "cheap" in pref_text:
                        # Extract budget if mentioned in preference
                        nums = re.findall(r'\$?(\d+)', pref_text)
                        if nums and not budget:
                            budget = float(nums[0])
                            plan_logs.append(f"Overriding parameter: 'budget={budget}' based on FAISS memory!")
            else:
                plan_logs.append("No matching preferences found in FAISS vector database.")
                
        # 4. Check for ambiguous inputs (e.g. no city specified)
        if not city:
            clarification = ("I'd love to help you plan an amazing 2-day trip! Could you please let me know "
                             "which city you'd like to visit? I currently support Tokyo, Paris, and Singapore.")
            self.short_memory.add_message("assistant", clarification)
            return {
                "response": clarification,
                "planning_steps": plan_logs,
                "parameters": extracted,
                "preferences_used": [p["preference"] for p in matching_prefs],
                "hops_log": [],
                "success": False,
                "clarification_needed": True
            }

        # 5. Execute Sub-tasks (Reasoning & Tool Invocation)
        # Sub-task 1: Run Multi-Hop RAG
        plan_logs.append(f"Sub-task 1: Executing Multi-Hop RAG to retrieve local tips for {city}...")
        rag_query = f"{city} " + " ".join(interests) + (" pet rules" if pet_friendly else "")
        rag_result = self.rag.execute_rag(rag_query, city=city)
        plan_logs.append(f"RAG Hop 1 & 2 completed. Retrieved: {rag_result['retrieved_docs']}")
        
        # Sub-task 2: Fetch Weather
        plan_logs.append(f"Sub-task 2: Fetching live weather tool data for {city}...")
        weather_data = self.tools.get_weather(city)
        plan_logs.append(f"Weather tool response: {weather_data.get('summary')} ({weather_data.get('avg_temp_c')}°C)")
        
        # Sub-task 3: Fetch Flight/Hotel options
        plan_logs.append(f"Sub-task 3: Querying accommodation tool for hotels and flights...")
        # Reserve about 40% of the budget for lodging
        max_hotel_price = (budget * 0.40 / 2) if budget else None
        accommodation_data = self.tools.get_accommodation(city, max_price_usd=max_hotel_price, pet_friendly=pet_friendly)
        plan_logs.append(f"Hotel options found: {[h['name'] for h in accommodation_data.get('hotels', [])]}")
        
        # Sub-task 4: Fetch Attractions
        plan_logs.append(f"Sub-task 4: Searching attractions tool matching categories and pet-friendliness...")
        # Allocate about 30% of budget for attractions
        max_attr_budget = (budget * 0.30) if budget else None
        attractions_list = self.tools.search_attractions(city, budget_usd=max_attr_budget, tags=interests, pet_friendly=pet_friendly)
        plan_logs.append(f"Attractions tool response: {[a['name'] for a in attractions_list]}")

        # 6. Generate rich response (Synthesize plan)
        plan_logs.append("Sub-task 5: Synthesizing plan using LLM model...")
        
        if self.llm_available:
            # Package all gathered context to pass to Gemini
            prompt = self._compile_llm_prompt(
                query=user_query,
                city=city,
                budget=budget,
                pet_friendly=pet_friendly,
                interests=interests,
                weather=weather_data,
                accommodation=accommodation_data,
                attractions=attractions_list,
                rag_context=rag_result["context"],
                preferences=[p["preference"] for p in matching_prefs]
            )
            try:
                plan_logs.append("Sending prompt with tool outputs and RAG context to Gemini model...")
                response_obj = self.model.generate_content(prompt)
                agent_response = response_obj.text
                plan_logs.append("Gemini response received.")
            except Exception as e:
                plan_logs.append(f"Gemini generation failed ({e}). Falling back to Local template engine.")
                agent_response = self._synthesize_local_response(
                    city, budget, pet_friendly, interests, weather_data, accommodation_data, attractions_list, rag_result
                )
        else:
            plan_logs.append("Running in Local Simulation mode (No API Key). Compiling robust travel guide...")
            agent_response = self._synthesize_local_response(
                city, budget, pet_friendly, interests, weather_data, accommodation_data, attractions_list, rag_result
            )
            
        # 7. Update Long-Term memory with newly detected preferences
        plan_logs.append("Detecting new user preferences to index in FAISS vector database...")
        new_preferences = self._detect_new_preferences(user_query, extracted)
        for pref in new_preferences:
            if pref not in self.memory.get_all_preferences():
                self.memory.add_preference(pref)
                plan_logs.append(f"FAISS Indexed preference: '{pref}'")
                
        self.short_memory.add_message("assistant", agent_response)
        plan_logs.append("Itinerary successfully built!")
        
        return {
            "response": agent_response,
            "planning_steps": plan_logs,
            "parameters": {
                "city": city,
                "budget": budget,
                "pet_friendly": pet_friendly,
                "interests": interests
            },
            "preferences_used": [p["preference"] for p in matching_prefs],
            "hops_log": rag_result["hops_log"],
            "success": True,
            "clarification_needed": False
        }

    def _extract_parameters(self, text: str) -> dict:
        """Parses city, budget, pet-friendliness, and tags/interests from text."""
        text_lower = text.lower()
        
        # 1. City extraction
        city = None
        if "tokyo" in text_lower:
            city = "Tokyo"
        elif "paris" in text_lower:
            city = "Paris"
        elif "singapore" in text_lower:
            city = "Singapore"
            
        # 2. Budget extraction
        budget = None
        budget_match = re.search(r'\$?(\d+)\s*(?:budget|dollars|usd|max)', text_lower)
        if not budget_match:
            # Try plain number next to dollar sign
            budget_match = re.search(r'\$\s*(\d+)', text_lower)
        if budget_match:
            budget = float(budget_match.group(1))
            
        # 3. Pet friendly check
        pet_friendly = any(kw in text_lower for kw in ["pet", "dog", "cat", "animal", "puppy", "canine"])
        
        # 4. Interests extraction
        interests = []
        interest_keywords = ["anime", "sushi", "culture", "art", "museum", "shopping", "nature", "historical", "beach", "relax", "kids", "sightseeing", "romance", "food"]
        for kw in interest_keywords:
            if kw in text_lower:
                interests.append(kw)
                
        return {
            "city": city,
            "budget": budget,
            "pet_friendly": pet_friendly,
            "interests": interests
        }

    def _detect_new_preferences(self, text: str, params: dict) -> list:
        """Finds long-term preferences that should be stored in FAISS."""
        prefs = []
        
        # Pet preference
        if params["pet_friendly"]:
            if "dog" in text.lower():
                prefs.append("The user travels with a dog and requires pet-friendly locations.")
            elif "cat" in text.lower():
                prefs.append("The user travels with a cat and requires pet-friendly locations.")
            else:
                prefs.append("The user travels with pets and requires pet-friendly services.")
                
        # Budget preference
        if params["budget"]:
            prefs.append(f"The user prefers a strict budget-conscious trip capped around ${params['budget']:.0f}.")
            
        # Interests preferences
        for interest in params["interests"]:
            prefs.append(f"The user is highly interested in {interest} activities during travel.")
            
        return prefs

    def _compile_llm_prompt(self, query: str, city: str, budget: float, pet_friendly: bool, interests: list, 
                            weather: dict, accommodation: dict, attractions: list, rag_context: str, preferences: list) -> str:
        """Compiles the prompt to send to Gemini including gathered context and tools responses."""
        return f"""
You are an Intelligent Travel Planning AI Agent. You specialize in generating highly personalized, realistic 2-day travel itineraries.

User Query: "{query}"

TARGET CITY: {city}
BUDGET LIMIT: ${budget if budget else "None specified"}
PET-FRIENDLY TRIP REQUIRED: {pet_friendly}
USER INTERESTS: {interests}

--- RETRIEVED LONG-TERM MEMORY PREFERENCES ---
{json.dumps(preferences, indent=2) if preferences else "No past preferences retrieved."}

--- ACTIVE WEATHER FOR TARGET CITY ---
{json.dumps(weather, indent=2)}

--- ACCOMMODATION & FLIGHTS OPTIONS (Retrieved from Tool) ---
{json.dumps(accommodation, indent=2)}

--- TARGET ATTRACTIONS (Retrieved from Tool) ---
{json.dumps(attractions, indent=2)}

--- MULTI-HOP RAG CONTEXT (Deep Background details & Blogs) ---
{rag_context}

--- INSTRUCTIONS ---
Formulate a beautiful, engaging, and detailed 2-day travel plan (Itinerary) for this user.
Make it highly realistic, taking into account:
1. The budget: Ensure the hotel and activities costs fit within the requested budget.
2. Pet-friendliness: Recommend ONLY pet-friendly attractions if 'PET-FRIENDLY' is true. Highlight pet rules (e.g. train regulations or beach policies) in a special tip section.
3. The weather: Reference the retrieved weather and give clothing / scheduling advice.
4. Multi-hop RAG context: Inject cultural trivia, histories, and local tips (like cafe etiquette or hawker rules) smoothly into the itinerary.

Structure your response perfectly in Markdown:
1. **Overview & Weather Guidelines**: A welcoming introduction detailing the forecast.
2. **Day 1: [Theme]** & **Day 2: [Theme]**: Break down by morning, afternoon, evening. Specify timings, estimated costs, and descriptive notes.
3. **Accommodation Selection**: Highlight the hotel choice and its costs.
4. **Estimated Budget Breakdown Table**: Show itemized costs for Hotels, Attractions, Food/Misc, and how it aligns with the budget limit.
5. **Special Travel Tips (RAG Advice)**: Include pet guidelines, dining tips, or custom trivia based on RAG.
"""

    def _synthesize_local_response(self, city: str, budget: float, pet_friendly: bool, interests: list,
                                  weather: dict, accommodation: dict, attractions: list, rag_result: dict) -> str:
        """Generates a highly detailed, professional travel plan from templates if Gemini is unavailable."""
        
        # Select hotel option
        hotels = accommodation.get("hotels", [])
        selected_hotel = None
        for h in hotels:
            # Prefer pet-friendly if requested
            h_tags = [t.lower() for t in h.get("tags", [])]
            if pet_friendly and "pet-friendly" not in h_tags:
                continue
            selected_hotel = h
            break
            
        if not selected_hotel and hotels:
            selected_hotel = hotels[0]
            
        hotel_cost_2_nights = (selected_hotel.get("price_per_night_usd", 120) * 2) if selected_hotel else 0
        hotel_name = selected_hotel.get("name", "Local Premium Lodge") if selected_hotel else "Not specified"
        hotel_desc = selected_hotel.get("description", "A comfortable central lodge.") if selected_hotel else ""
        
        # Select attractions (Day 1: 2 items, Day 2: 2 items)
        selected_attrs = attractions[:4]
        while len(selected_attrs) < 4 and len(attractions) > 0:
            selected_attrs.append(attractions[0]) # duplicate/pad if too few
            
        # If no attractions returned, use a fallback
        if not selected_attrs:
            selected_attrs = [
                {"name": "Central Historic Plaza", "cost_usd": 0, "duration_hours": 2, "description": "Beautiful public plaza."},
                {"name": "City Botanical Gardens", "cost_usd": 5, "duration_hours": 2, "description": "Relaxing green space."},
                {"name": "Downtown Walking Tour", "cost_usd": 0, "duration_hours": 1.5, "description": "Iconic streets and monuments."},
                {"name": "Scenic Riverfront Promenade", "cost_usd": 0, "duration_hours": 2, "description": "Stunning sunset views."}
            ]
            
        # Allocate attractions to days
        d1_morn = selected_attrs[0]
        d1_aft = selected_attrs[1] if len(selected_attrs) > 1 else selected_attrs[0]
        d2_morn = selected_attrs[2] if len(selected_attrs) > 2 else selected_attrs[0]
        d2_aft = selected_attrs[3] if len(selected_attrs) > 3 else selected_attrs[0]
        
        # Calculate budgets
        attr_total_cost = sum(a.get("cost_usd", 0) for a in [d1_morn, d1_aft, d2_morn, d2_aft])
        food_est = 45 * 2 # $45/day food
        misc_est = 20 * 2 # $20/day transport
        
        total_estimate = hotel_cost_2_nights + attr_total_cost + food_est + misc_est
        
        budget_status = ""
        if budget:
            diff = budget - total_estimate
            if diff >= 0:
                budget_status = f"🎉 **Under Budget!** You have **${diff:.2f}** remaining from your ${budget:.0f} budget limit."
            else:
                budget_status = f"⚠️ **Over Budget by ${abs(diff):.2f}**. You might want to opt for cheaper lodging or free attractions."
                
        # Build Markdown
        md = []
        md.append(f"# Personalized 2-Day Itinerary: {city}")
        md.append(f"\n### 🌤️ Weather Forecast for {city}")
        md.append(f"> **{weather.get('summary')}**\n> - **Average Temperature**: {weather.get('avg_temp_c')}°C\n> - **Chance of Rain**: {weather.get('chance_of_rain')}\n> - **Humidity**: {weather.get('humidity')}")
        
        md.append(f"\n---\n")
        md.append(f"## 🏨 Recommended Lodging")
        md.append(f"**{hotel_name}**")
        md.append(f"- **Price**: ${selected_hotel.get('price_per_night_usd', 0)}/night (Total for 2 nights: **${hotel_cost_2_nights}**)")
        md.append(f"- **Rating**: {selected_hotel.get('rating', '4.5')}/5")
        md.append(f"- *{hotel_desc}*")
        
        md.append(f"\n---\n")
        md.append(f"## 🗺️ Day-by-Day Travel Schedule")
        
        # Day 1
        md.append(f"### 🗓️ Day 1: Exploring Core Heritage & Landmarks")
        md.append(f"- **09:00 AM - Morning Exploration | {d1_morn['name']}**")
        md.append(f"  - **Cost**: ${d1_morn.get('cost_usd', 0)} | **Duration**: {d1_morn.get('duration_hours', 2)} hours")
        md.append(f"  - *{d1_morn.get('description', '')}*")
        md.append(f"  - 💡 *Tip*: {d1_morn.get('best_time', 'Go early for the best experience.')}")
        
        md.append(f"- **12:30 PM - Lunch Break**")
        md.append(f"  - Try local specialties in the central food markets.")
        
        md.append(f"- **02:00 PM - Afternoon Sightseeing | {d1_aft['name']}**")
        md.append(f"  - **Cost**: ${d1_aft.get('cost_usd', 0)} | **Duration**: {d1_aft.get('duration_hours', 2)} hours")
        md.append(f"  - *{d1_aft.get('description', '')}*")
        md.append(f"  - 💡 *Tip*: {d1_aft.get('best_time', 'Recommended in the afternoon.')}")
        
        md.append(f"- **06:00 PM - Dinner & Evening Walk**")
        md.append(f"  - Head out for a relaxed dinner near {hotel_name}.")

        # Day 2
        md.append(f"\n### 🗓️ Day 2: Modern Culture & Immersion")
        md.append(f"- **09:00 AM - Morning Activity | {d2_morn['name']}**")
        md.append(f"  - **Cost**: ${d2_morn.get('cost_usd', 0)} | **Duration**: {d2_morn.get('duration_hours', 2)} hours")
        md.append(f"  - *{d2_morn.get('description', '')}*")
        md.append(f"  - 💡 *Tip*: {d2_morn.get('best_time', 'Early visit suggested.')}")
        
        md.append(f"- **12:00 PM - Lunch**")
        md.append(f"  - Relax at a pet-friendly cafe or local street vendor.")
        
        md.append(f"- **02:00 PM - Special Interest Visit | {d2_aft['name']}**")
        md.append(f"  - **Cost**: ${d2_aft.get('cost_usd', 0)} | **Duration**: {d2_aft.get('duration_hours', 2)} hours")
        md.append(f"  - *{d2_aft.get('description', '')}*")
        md.append(f"  - 💡 *Tip*: {d2_aft.get('best_time', 'Great spot to close out your trip.')}")
        
        md.append(f"- **06:00 PM - Farewell Sunset Walk**")
        md.append(f"  - Grab some street eats and reflect on an amazing weekend.")
        
        # Budget breakdown table
        md.append(f"\n---\n")
        md.append(f"## 📊 Estimated Budget Breakdown")
        md.append(f"| Category | Estimated Cost (USD) | Details |")
        md.append(f"| :--- | :--- | :--- |")
        md.append(f"| **Lodging** | ${hotel_cost_2_nights:.2f} | 2 Nights at {hotel_name} |")
        md.append(f"| **Attractions** | ${attr_total_cost:.2f} | Entry fees for sights |")
        md.append(f"| **Food & Dining** | ${food_est:.2f} | Estimated at $45/day |")
        md.append(f"| **Local Transport & Misc** | ${misc_est:.2f} | Subways, taxis, and small treats |")
        md.append(f"| **Total Estimated** | **${total_estimate:.2f}** | |")
        md.append(f"\n{budget_status}")
        
        # Multi-Hop RAG insights
        md.append(f"\n---\n")
        md.append(f"## 💡 Deep Local Insights & Guidelines (Multi-Hop RAG)")
        for key, snippet in TRAVEL_BLOGS.items():
            if city.lower().strip() in key or ("nakamise" in key and city.lower().strip() == "tokyo") or ("tanjong" in key and city.lower().strip() == "singapore"):
                md.append(f"\n- **{key.replace('_', ' ').title()}**:")
                md.append(f"  > *{snippet}*")
                
        return "\n".join(md)
