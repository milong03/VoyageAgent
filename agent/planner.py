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

        self.llm_available = False
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
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

    # ──────────────────────────────────────────────────────────────────────────
    # INTENT DETECTION
    # ──────────────────────────────────────────────────────────────────────────
    def _detect_intent(self, text: str) -> str:
        """
        Classifies the user message into one of:
        - 'plan_trip'       : User wants a full itinerary
        - 'budget_advice'   : User is asking what budget is needed / if theirs is enough
        - 'follow_up'       : User is asking a follow-up question about the last response
        - 'general_advice'  : Open-ended travel advice without a specific trip request
        """
        t = text.lower()

        # Follow-up signals — refers to previous conversation
        follow_up_signals = [
            "that", "it", "this", "those", "the one", "the hotel", "the plan",
            "you mentioned", "what about", "tell me more", "explain", "why",
            "how much is that", "too expensive", "cheaper option", "alternative",
            "is that good", "what do you think", "any other", "instead"
        ]
        if self.short_memory.get_context() and any(s in t for s in follow_up_signals):
            return "follow_up"

        # Budget advice signals
        budget_advice_signals = [
            "how much", "what budget", "what is a good budget", "recommend a budget",
            "suggest a budget", "what should i budget", "minimum budget", "how expensive",
            "is my budget enough", "is that enough", "can i afford", "is $", "is it cheap",
            "how much does it cost", "cost of", "price of", "how much money",
            "what would it cost", "realistic budget", "reasonable budget"
        ]
        if any(s in t for s in budget_advice_signals):
            return "budget_advice"

        # General travel advice — no specific trip request
        general_advice_signals = [
            "what should i do", "what to do", "best time to visit", "tips for",
            "advice for", "what to see", "must see", "recommend", "what is",
            "tell me about", "describe", "what's it like", "what is it like"
        ]
        if any(s in t for s in general_advice_signals):
            # Only general if no city or no planning verb
            planning_verbs = ["plan", "book", "trip", "visit", "go to", "travel", "itinerary"]
            if not any(v in t for v in planning_verbs):
                return "general_advice"

        return "plan_trip"

    # ──────────────────────────────────────────────────────────────────────────
    # BUDGET FEASIBILITY
    # ──────────────────────────────────────────────────────────────────────────
    def _check_budget_feasibility(self, city: str, budget: float, pet_friendly: bool) -> dict:
        """
        Checks whether the given budget is realistic for a 2-day trip.
        Returns a dict with feasibility info and a recommended minimum budget.
        """
        # Get all hotels without any budget filter to find absolute minimum
        all_accommodation = self.tools.get_accommodation(city, max_price_usd=None, pet_friendly=False)
        all_hotels = all_accommodation.get("hotels", [])

        if all_hotels:
            cheapest_hotel = min(all_hotels, key=lambda h: h.get("price_per_night_usd", 9999))
            min_hotel_per_night = cheapest_hotel.get("price_per_night_usd", 80)
        else:
            min_hotel_per_night = 80  # fallback estimate

        min_hotel_total = min_hotel_per_night * 2
        min_food = 30 * 2        # $30/day bare minimum
        min_transport = 15 * 2   # $15/day local transport
        min_total = min_hotel_total + min_food + min_transport

        # Recommended comfortable budget
        recommended = min_total * 1.5

        feasible = budget is None or budget >= min_total

        warning = None
        if budget is not None and budget < min_total:
            warning = (
                f"Your budget of ${budget:.0f} is below the estimated minimum of ${min_total:.0f} "
                f"for a 2-day trip to {city} (cheapest hotel is ${min_hotel_per_night}/night, "
                f"plus food and transport). "
                f"A realistic starting budget would be around ${recommended:.0f}. "
                f"I will still generate a plan showing what is possible, but lodging options may be unavailable."
            )
        elif budget is not None and budget < recommended:
            warning = (
                f"Your budget of ${budget:.0f} is workable but tight for {city}. "
                f"A more comfortable budget would be around ${recommended:.0f}. "
                f"I will prioritise the most affordable options."
            )

        return {
            "feasible": feasible,
            "min_viable_budget": round(min_total),
            "recommended_budget": round(recommended),
            "cheapest_hotel_per_night": min_hotel_per_night,
            "warning": warning
        }

    # ──────────────────────────────────────────────────────────────────────────
    # ADVISORY RESPONSE (Gemini-powered or template fallback)
    # ──────────────────────────────────────────────────────────────────────────
    def _handle_advisory_query(self, user_query: str, intent: str, city: str = None) -> str:
        """
        Handles budget advice, general advice, and follow-up questions
        using Gemini with full conversation history injected as context.
        """
        history = self.short_memory.get_context()
        history_text = "\n".join(
            [f"{m['role'].upper()}: {m['content'][:500]}" for m in history]
        ) if history else "No prior conversation."

        # Pull any feasibility data if city is known
        feasibility_note = ""
        if city:
            weather = self.tools.get_weather(city)
            acc = self.tools.get_accommodation(city, max_price_usd=None, pet_friendly=False)
            all_hotels = acc.get("hotels", [])
            if all_hotels:
                prices = [h.get("price_per_night_usd", 0) for h in all_hotels]
                feasibility_note = (
                    f"\n\nLIVE TOOL DATA for {city}:"
                    f"\n- Hotel price range: ${min(prices)}/night to ${max(prices)}/night"
                    f"\n- Weather: {weather.get('summary', 'N/A')}"
                    f"\n- Estimated minimum 2-day budget: ${min(prices)*2 + 90:.0f}"
                    f"\n- Estimated comfortable 2-day budget: ${min(prices)*2 + 200:.0f}"
                )

        gemini_error = None
        if self.llm_available:
            prompt = f"""You are VoyageAgent, an Intelligent Travel Planning AI. Answer the user's question helpfully and conversationally.

CONVERSATION HISTORY (last 6 turns):
{history_text}

CURRENT USER MESSAGE: "{user_query}"
{feasibility_note}

INSTRUCTIONS:
- If the user asks about budget, give specific dollar figures based on the live tool data above.
- If the user asks a follow-up, refer to the conversation history for context.
- If the user's budget seems insufficient, clearly say so and suggest a realistic alternative.
- Be direct, concise, and helpful. Do not generate a full itinerary unless asked.
- If you recommend a budget, explain what it covers (hotel, food, transport, activities).
- Format clearly in Markdown.
"""
            try:
                response_obj = self.model.generate_content(prompt)
                return response_obj.text
            except Exception as e:
                gemini_error = str(e)
                print(f"Advisory Gemini error: {gemini_error}")

        # Local fallback for advisory queries
        if city and intent == "budget_advice":
            feasibility = self._check_budget_feasibility(city, None, False)
            return (
                f"## Budget Guide for {city}\n\n"
                f"Here are estimated costs for a 2-day trip:\n\n"
                f"| Category | Estimate |\n"
                f"| :--- | :--- |\n"
                f"| Cheapest hotel (2 nights) | ${feasibility['cheapest_hotel_per_night'] * 2} |\n"
                f"| Food (2 days at $35/day) | $70 |\n"
                f"| Local transport (2 days) | $30 |\n"
                f"| **Minimum viable total** | **${feasibility['min_viable_budget']}** |\n"
                f"| **Comfortable recommended** | **${feasibility['recommended_budget']}** |\n\n"
                f"For a more relaxed experience with mid-range hotels and activities, "
                f"I would recommend budgeting around **${feasibility['recommended_budget']}**."
            )

        # If LLM failed, tell the user explicitly why they can't ask dynamic questions
        if gemini_error or not self.llm_available:
            return (
                f"> **🔌 Local Simulation Mode Active**\n>\n"
                f"> I cannot answer dynamic follow-up questions like *\"{user_query}\"* right now because my connection to Gemini is offline.\n"
                f"> *(Reason: {gemini_error if gemini_error else 'No API Key configured'})*\n>\n"
                f"> To enable full conversational context and recommendations, please enter a valid Gemini API Key in the Settings menu."
            )

        return (
            "I am here to help with travel planning. Could you tell me which city you are interested in "
            "so I can give you specific advice on costs, hotels, and activities?"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN PLAN-AND-EXECUTE LOOP
    # ──────────────────────────────────────────────────────────────────────────
    def plan_trip(self, user_query: str, session_id: str = "default") -> dict:
        """Executes the complete Plan-and-Execute loop to build a travel itinerary."""
        plan_logs = []
        plan_logs.append("Analyzing user query and session context...")

        # 1. Extract parameters first so we know which city context to use
        extracted = self._extract_parameters(user_query)
        city = extracted["city"]

        # 2. Update short-term memory grouped by the active city
        self.short_memory.add_message("user", user_query, city=city)

        # 3. Detect intent (uses context of active city)
        intent = self._detect_intent(user_query)
        plan_logs.append(f"Intent detected: '{intent}'")

        budget = extracted["budget"]
        pet_friendly = extracted["pet_friendly"]
        interests = extracted["interests"]

        plan_logs.append(
            f"Extracted parameters -> City: {city or 'Unknown'}, Budget: {budget or 'Not specified'}, "
            f"Pet-Friendly: {pet_friendly}, Interests: {interests}"
        )

        # 4. Retrieve Long-Term Memory from FAISS
        plan_logs.append("Querying FAISS Long-Term Memory for past session preferences...")
        matching_prefs = []
        query_city = city or self._extract_city_from_history()
        if query_city:
            matching_prefs = self.memory.query_preferences(f"{query_city} {interests}", top_k=3)
            if matching_prefs:
                plan_logs.append(
                    f"Retrieved {len(matching_prefs)} relevant preferences from FAISS: " +
                    ", ".join([f"'{p['preference']}' (score: {p['score']:.2f})" for p in matching_prefs])
                )
                for p in matching_prefs:
                    pref_text = p["preference"].lower()
                    if "dog" in pref_text or "pet" in pref_text or "animal" in pref_text:
                        pet_friendly = True
                        plan_logs.append("Overriding parameter: 'pet_friendly=True' based on FAISS long-term memory!")
                    if "budget" in pref_text or "cheap" in pref_text:
                        nums = re.findall(r'\$?(\d+)', pref_text)
                        if nums and not budget:
                            budget = float(nums[0])
                            plan_logs.append(f"Overriding parameter: 'budget={budget}' based on FAISS memory!")
            else:
                plan_logs.append("No matching preferences found in FAISS vector database.")

        # 5. Handle non-planning intents (advisory / follow-up)
        if intent in ("budget_advice", "follow_up", "general_advice"):
            plan_logs.append(f"Routing to advisory handler for intent: '{intent}'")
            advisory_city = city or self._extract_city_from_history()
            response = self._handle_advisory_query(user_query, intent, city=advisory_city)
            self.short_memory.add_message("assistant", response)
            return {
                "response": response,
                "planning_steps": plan_logs,
                "parameters": extracted,
                "preferences_used": [p["preference"] for p in matching_prefs],
                "hops_log": [],
                "success": True,
                "clarification_needed": False
            }

        # 6. No city — ask for clarification
        if not city:
            country = extracted.get("country")
            if country:
                clarification = (
                    f"I see you want to visit {country.title()}! Since I plan highly detailed local itineraries, "
                    f"could you tell me which specific city in {country.title()} you'd like to travel to?"
                )
                self.short_memory.add_message("assistant", clarification, city="default")
                return {
                    "response": clarification,
                    "planning_steps": plan_logs,
                    "parameters": extracted,
                    "preferences_used": [],
                    "hops_log": [],
                    "success": False,
                    "clarification_needed": True
                }

            # Try to pull city from recent conversation history
            city = self._extract_city_from_history()
            if city:
                plan_logs.append(f"Resolved city '{city}' from conversation history.")
            else:
                clarification = (
                    "I would love to help you plan an amazing 2-day trip! "
                    "Which city would you like to visit? I can plan for any city in the world."
                )
                self.short_memory.add_message("assistant", clarification, city="default")
                return {
                    "response": clarification,
                    "planning_steps": plan_logs,
                    "parameters": extracted,
                    "preferences_used": [],
                    "hops_log": [],
                    "success": False,
                    "clarification_needed": True
                }

        # 7. Budget feasibility check
        plan_logs.append(f"Sub-task 0: Checking budget feasibility for {city}...")
        feasibility = self._check_budget_feasibility(city, budget, pet_friendly)
        if feasibility["warning"]:
            plan_logs.append(f"Budget advisor: {feasibility['warning']}")

        # 8. Sub-task 1: Multi-Hop RAG
        plan_logs.append(f"Sub-task 1: Executing Multi-Hop RAG to retrieve local tips for {city}...")
        rag_query = f"{city} " + " ".join(interests) + (" pet rules" if pet_friendly else "")
        rag_result = self.rag.execute_rag(rag_query, city=city)
        plan_logs.append(f"RAG Hop 1 & 2 completed. Retrieved: {rag_result['retrieved_docs']}")

        # 9. Sub-task 2: Weather
        plan_logs.append(f"Sub-task 2: Fetching live weather tool data for {city}...")
        weather_data = self.tools.get_weather(city)
        plan_logs.append(f"Weather tool response: {weather_data.get('summary')} ({weather_data.get('avg_temp_c')}C)")

        # 10. Sub-task 3: Accommodation
        plan_logs.append("Sub-task 3: Querying accommodation tool for hotels and flights...")
        max_hotel_price = (budget * 0.40 / 2) if budget and budget > 0 else None
        accommodation_data = self.tools.get_accommodation(city, max_price_usd=max_hotel_price, pet_friendly=pet_friendly)
        plan_logs.append(f"Hotel options found: {[h['name'] for h in accommodation_data.get('hotels', [])]}")

        # 11. Sub-task 4: Attractions
        plan_logs.append("Sub-task 4: Searching attractions tool matching categories and pet-friendliness...")
        max_attr_budget = (budget * 0.30) if budget and budget > 0 else None
        attractions_list = self.tools.search_attractions(city, budget_usd=max_attr_budget, tags=interests, pet_friendly=pet_friendly)
        plan_logs.append(f"Attractions tool response: {[a['name'] for a in attractions_list]}")

        # 12. Sub-task 5: Synthesize plan
        plan_logs.append("Sub-task 5: Synthesizing plan using LLM model...")
        history_context = self._format_history_for_prompt()

        if self.llm_available:
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
                preferences=[p["preference"] for p in matching_prefs],
                feasibility=feasibility,
                history_context=history_context
            )
            try:
                plan_logs.append("Sending prompt with tool outputs and RAG context to Gemini model...")
                response_obj = self.model.generate_content(prompt)
                agent_response = response_obj.text
                plan_logs.append("Gemini response received.")
            except Exception as e:
                plan_logs.append(f"Gemini generation failed ({e}). Falling back to Local template engine.")
                agent_response = self._synthesize_local_response(
                    city, budget, pet_friendly, interests, weather_data,
                    accommodation_data, attractions_list, rag_result, feasibility
                )
        else:
            plan_logs.append("Running in Local Simulation mode (No API Key). Compiling robust travel guide...")
            agent_response = self._synthesize_local_response(
                city, budget, pet_friendly, interests, weather_data,
                accommodation_data, attractions_list, rag_result, feasibility
            )

        # 13. Auto-persist new preferences to FAISS
        plan_logs.append("Detecting new user preferences to index in FAISS vector database...")
        new_preferences = self._detect_new_preferences(user_query, {
            "city": city, "budget": budget, "pet_friendly": pet_friendly, "interests": interests
        })
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

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────
    def _extract_city_from_history(self) -> str:
        """Scans recent conversation history to find a city mentioned earlier."""
        history = self.short_memory.get_context()
        for msg in reversed(history):
            params = self._extract_parameters(msg["content"])
            if params["city"]:
                return params["city"]
        return None

    def _format_history_for_prompt(self) -> str:
        """Formats the entire conversation history for this city for injection into the Gemini prompt."""
        history = self.short_memory.get_context()
        if not history or len(history) <= 1:
            return "No prior conversation for this destination."
        # Exclude the most recent user message (already in query)
        recent = history[:-1]
        return "\n".join([f"{m['role'].upper()}: {m['content'][:600]}" for m in recent])

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
        else:
            match = re.search(
                r'\b(?:trip to|visit|go to|weekend in|travel to|traveling to)\s+([a-z]+(?:\s+[a-z]+){0,2})',
                text_lower
            )
            if match:
                extracted_city = match.group(1).strip()
                stop_words = ["my", "with", "a", "an", "the", "on", "for", "and", "some", "budget", "in"]
                city_words = []
                for w in extracted_city.split():
                    if w in stop_words:
                        break
                    city_words.append(w.capitalize())
                if city_words:
                    city = " ".join(city_words)

            if not city:
                words = [w.strip(",.!?") for w in text.split()]
                non_stop = [w for w in words if w.lower() not in [
                    "i", "want", "to", "plan", "a", "trip", "visit", "go", "weekend",
                    "in", "with", "my", "dog", "cat", "pet", "budget", "and", "the", "an"
                ]]
                if len(non_stop) == 1:
                    city = non_stop[0].capitalize()

        # Filter out country names
        countries = {
            "japan", "france", "italy", "germany", "spain", "uk", "usa", "united states",
            "united kingdom", "canada", "australia", "china", "india", "brazil", "mexico",
            "korea", "malaysia", "thailand", "vietnam", "indonesia", "philippines", "greece",
            "egypt", "turkey", "switzerland", "sweden", "norway", "denmark", "finland",
            "netherlands", "belgium", "austria", "portugal", "russia", "new zealand",
            "south africa", "argentina", "colombia", "peru", "chile", "ireland", "poland"
        }
        country = None
        if city and city.lower() in countries:
            country = city
            city = None

        # Expand common city abbreviations
        abbreviations = {
            "kl": "Kuala Lumpur",
            "nyc": "New York City",
            "la": "Los Angeles",
            "sf": "San Francisco",
            "hk": "Hong Kong",
            "dc": "Washington",
            "bkk": "Bangkok",
            "dxb": "Dubai"
        }
        if city and city.lower() in abbreviations:
            city = abbreviations[city.lower()]

        # 2. Budget extraction
        budget = None
        budget_match = re.search(r'(-?\d+)\s*(?:budget|dollars|usd|max)', text_lower)
        if not budget_match:
            budget_match = re.search(
                r'(?:budget\s*of|budget\s*:\s*|budget|max\s*of|max\s*:\s*|max)\s*\$?(-?\d+)',
                text_lower
            )
        if not budget_match:
            budget_match = re.search(r'\$\s*(-?\d+)', text_lower)
        if budget_match:
            budget = float(budget_match.group(1))

        # 3. Pet friendly
        pet_friendly = any(kw in text_lower for kw in ["pet", "dog", "cat", "animal", "puppy", "canine"])

        # 4. Interests
        interests = []
        interest_keywords = [
            "anime", "sushi", "culture", "art", "museum", "shopping", "nature",
            "historical", "beach", "relax", "kids", "sightseeing", "romance", "food"
        ]
        for kw in interest_keywords:
            if kw in text_lower:
                interests.append(kw)

        return {"city": city, "country": country, "budget": budget, "pet_friendly": pet_friendly, "interests": interests}

    def _detect_new_preferences(self, text: str, params: dict) -> list:
        """Finds long-term preferences that should be stored in FAISS."""
        prefs = []
        if params["pet_friendly"]:
            if "dog" in text.lower():
                prefs.append("The user travels with a dog and requires pet-friendly locations.")
            elif "cat" in text.lower():
                prefs.append("The user travels with a cat and requires pet-friendly locations.")
            else:
                prefs.append("The user travels with pets and requires pet-friendly services.")
        if params["budget"]:
            prefs.append(f"The user prefers a strict budget-conscious trip capped around ${params['budget']:.0f}.")
        for interest in params["interests"]:
            prefs.append(f"The user is highly interested in {interest} activities during travel.")
        return prefs

    def _compile_llm_prompt(self, query, city, budget, pet_friendly, interests,
                            weather, accommodation, attractions, rag_context,
                            preferences, feasibility, history_context) -> str:
        """Compiles the full context-aware prompt to send to Gemini."""

        budget_warning_block = ""
        if feasibility and feasibility.get("warning"):
            budget_warning_block = f"""
--- BUDGET FEASIBILITY ALERT ---
{feasibility['warning']}
Minimum viable budget for {city}: ${feasibility['min_viable_budget']}
Recommended comfortable budget: ${feasibility['recommended_budget']}
"""

        return f"""
You are VoyageAgent — an intelligent, context-aware travel planning AI agent built on a
Plan-and-Execute architecture with real-time tool use and multi-hop RAG from Wikipedia.

=== CORE BEHAVIOURAL RULES (ALWAYS FOLLOW) ===

1. CONTEXT MAINTENANCE
   - The conversation history below is your primary context. Use it to understand pronouns,
     follow-up references ("that hotel", "the plan you mentioned", "is that expensive?"),
     and implicit intent. NEVER treat each message as isolated.
   - If the user refers to something from a previous turn, answer using that context.

2. CLARIFICATION BEFORE PLANNING
   - If a critical parameter is ambiguous or missing, ASK before generating a full plan.
   - Missing city → ask which city.
   - No budget mentioned and city is expensive → ask for their approximate budget range.
   - Contradictory constraints (e.g., pet-friendly but chosen hotel doesn't allow pets) → flag it.
   - Do NOT invent parameters. Ask.

3. BUDGET REALISM
   - If a budget feasibility alert is present: ADDRESS IT FIRST and PROMINENTLY.
   - Provide exact figures: cheapest hotel/night, estimated food/day, transport/day.
   - Recommend a realistic minimum AND a comfortable budget.
   - Then still generate the best possible plan given the actual constraint, clearly noting
     what is unavailable (e.g., "No hotels available at this budget — cheapest is $X/night").
   - NEVER silently output "Lodging: Not specified" for a budget issue. Explain it.

4. RAG-DRIVEN DEPTH
   - The Multi-Hop RAG context below contains real Wikipedia data retrieved live for this city.
   - USE IT. Reference specific landmarks, historical facts, local customs, and cuisine details
     pulled from those sources. This makes the plan feel locally authoritative, not generic.
   - Cite specific place names, streets, dishes, and cultural notes from the RAG context.

5. ADAPTIVE SUGGESTIONS
   - If something the user wants is not feasible (budget, pet policy, closed attraction),
     proactively suggest a concrete alternative rather than just saying "not available".

=== CONVERSATION HISTORY (last 5 turns) ===
{history_context}

=== CURRENT USER MESSAGE ===
"{query}"

=== EXTRACTED PARAMETERS ===
City: {city}
Budget: ${budget if budget else "Not specified"}
Pet-Friendly: {pet_friendly}
Interests: {interests if interests else "Not specified"}
{budget_warning_block}
=== LONG-TERM FAISS MEMORY (past preferences) ===
{json.dumps(preferences, indent=2) if preferences else "No past preferences on file."}

=== LIVE WEATHER DATA ===
{json.dumps(weather, indent=2)}

=== ACCOMMODATION OPTIONS FROM TOOL ===
{json.dumps(accommodation, indent=2)}

=== ATTRACTIONS FROM TOOL ===
{json.dumps(attractions, indent=2)}

=== MULTI-HOP RAG CONTEXT (Wikipedia + Local Blogs) ===
{rag_context}

=== OUTPUT FORMAT ===
Respond in clean Markdown. Structure:
1. Budget Advisory (ONLY if budget alert exists — make it prominent and specific)
2. Trip Overview (city highlights, weather advice)
3. Day 1 Itinerary (morning / afternoon / evening with times, costs, RAG-sourced tips)
4. Day 2 Itinerary (same structure)
5. Accommodation (selected option with price, or honest explanation + minimum budget to unlock options)
6. Estimated Budget Breakdown (table: hotel, food, transport, attractions, total)
7. Local Insights & Travel Tips (drawn from RAG Wikipedia data — cite specific places/facts)
"""

    def _synthesize_local_response(self, city, budget, pet_friendly, interests,
                                   weather, accommodation, attractions, rag_result,
                                   feasibility=None) -> str:
        """Generates a detailed professional travel plan from templates (no API key required)."""

        md = []

        # Budget warning — prominent at top
        if feasibility and feasibility.get("warning"):
            md.append("## Budget Advisory\n")
            md.append(f"> **Notice:** {feasibility['warning']}\n")
            md.append(f"> - Minimum viable budget for {city}: **${feasibility['min_viable_budget']}**")
            md.append(f"> - Recommended comfortable budget: **${feasibility['recommended_budget']}**\n")
            md.append("---\n")

        # Hotel selection
        hotels = accommodation.get("hotels", [])
        selected_hotel = None
        for h in hotels:
            h_tags = [t.lower() for t in h.get("tags", [])]
            if pet_friendly and "pet-friendly" not in h_tags:
                continue
            selected_hotel = h
            break
        if not selected_hotel and hotels:
            selected_hotel = hotels[0]

        hotel_cost_2_nights = (selected_hotel.get("price_per_night_usd", 120) * 2) if selected_hotel else 0
        hotel_name = selected_hotel.get("name", "Not specified") if selected_hotel else "Not specified"
        hotel_desc = selected_hotel.get("description", "") if selected_hotel else ""

        # Attractions
        selected_attrs = attractions[:4]
        while len(selected_attrs) < 4 and attractions:
            selected_attrs.append(attractions[0])
        if not selected_attrs:
            selected_attrs = [
                {"name": "Central Historic Plaza", "cost_usd": 0, "duration_hours": 2, "description": "Beautiful public plaza.", "best_time": "Morning"},
                {"name": "City Botanical Gardens", "cost_usd": 5, "duration_hours": 2, "description": "Relaxing green space.", "best_time": "Afternoon"},
                {"name": "Downtown Walking Tour", "cost_usd": 0, "duration_hours": 1.5, "description": "Iconic streets.", "best_time": "Morning"},
                {"name": "Scenic Riverfront Promenade", "cost_usd": 0, "duration_hours": 2, "description": "Stunning sunset views.", "best_time": "Evening"},
            ]

        d1_morn, d1_aft = selected_attrs[0], selected_attrs[1] if len(selected_attrs) > 1 else selected_attrs[0]
        d2_morn, d2_aft = (selected_attrs[2] if len(selected_attrs) > 2 else selected_attrs[0],
                           selected_attrs[3] if len(selected_attrs) > 3 else selected_attrs[0])

        attr_total_cost = sum(a.get("cost_usd", 0) for a in [d1_morn, d1_aft, d2_morn, d2_aft])
        food_est = 45 * 2
        misc_est = 20 * 2
        total_estimate = hotel_cost_2_nights + attr_total_cost + food_est + misc_est

        budget_status = ""
        if budget:
            diff = budget - total_estimate
            if diff >= 0:
                budget_status = f"**Under Budget:** You have **${diff:.2f}** remaining from your ${budget:.0f} limit."
            else:
                budget_status = (
                    f"**Over Budget by ${abs(diff):.2f}.** "
                    f"A realistic budget for this trip is around **${total_estimate:.0f}**. "
                    f"Consider cheaper lodging or free attractions to reduce costs."
                )

        # Build itinerary
        md.append(f"# Personalized 2-Day Itinerary: {city}\n")
        md.append(f"### Weather Forecast for {city}")
        md.append(f"> **{weather.get('summary')}**")
        md.append(f"> - **Average Temperature**: {weather.get('avg_temp_c')}C")
        md.append(f"> - **Chance of Rain**: {weather.get('chance_of_rain')}")
        md.append(f"> - **Humidity**: {weather.get('humidity')}\n")
        md.append("---\n")

        md.append("## Recommended Lodging")
        md.append(f"**{hotel_name}**")
        if selected_hotel:
            md.append(f"- **Price**: ${selected_hotel.get('price_per_night_usd', 0)}/night (Total 2 nights: **${hotel_cost_2_nights}**)")
            md.append(f"- **Rating**: {selected_hotel.get('rating', '4.5')}/5")
            md.append(f"- *{hotel_desc}*")
        else:
            md.append(f"- No lodging options are available within your budget of ${budget:.0f}.")
            md.append(f"- The cheapest available hotel in {city} starts at **${feasibility['cheapest_hotel_per_night'] if feasibility else 80}/night**.")
            md.append(f"- To unlock lodging options, a minimum budget of **${feasibility['min_viable_budget'] if feasibility else 'N/A'}** is needed.")
        md.append("\n---\n")

        md.append("## Day-by-Day Travel Schedule\n")
        md.append(f"### Day 1: Exploring Core Heritage & Landmarks")
        md.append(f"- **09:00 AM - Morning Exploration | {d1_morn['name']}**")
        md.append(f"  - **Cost**: ${d1_morn.get('cost_usd', 0)} | **Duration**: {d1_morn.get('duration_hours', 2)} hours")
        md.append(f"  - *{d1_morn.get('description', '')}*")
        md.append(f"  - *Tip*: {d1_morn.get('best_time', 'Go early for the best experience.')}")
        md.append(f"- **12:30 PM - Lunch Break**")
        md.append(f"  - Try local specialties in the central food markets.")
        md.append(f"- **02:00 PM - Afternoon Sightseeing | {d1_aft['name']}**")
        md.append(f"  - **Cost**: ${d1_aft.get('cost_usd', 0)} | **Duration**: {d1_aft.get('duration_hours', 2)} hours")
        md.append(f"  - *{d1_aft.get('description', '')}*")
        md.append(f"  - *Tip*: {d1_aft.get('best_time', 'Recommended in the afternoon.')}")
        md.append(f"- **06:00 PM - Dinner & Evening Walk**")
        md.append(f"  - Head out for a relaxed dinner near {hotel_name}.\n")

        md.append(f"### Day 2: Modern Culture & Immersion")
        md.append(f"- **09:00 AM - Morning Activity | {d2_morn['name']}**")
        md.append(f"  - **Cost**: ${d2_morn.get('cost_usd', 0)} | **Duration**: {d2_morn.get('duration_hours', 2)} hours")
        md.append(f"  - *{d2_morn.get('description', '')}*")
        md.append(f"  - *Tip*: {d2_morn.get('best_time', 'Early visit suggested.')}")
        md.append(f"- **12:00 PM - Lunch**")
        md.append(f"  - Relax at a local street vendor or cafe.")
        md.append(f"- **02:00 PM - Special Interest Visit | {d2_aft['name']}**")
        md.append(f"  - **Cost**: ${d2_aft.get('cost_usd', 0)} | **Duration**: {d2_aft.get('duration_hours', 2)} hours")
        md.append(f"  - *{d2_aft.get('description', '')}*")
        md.append(f"  - *Tip*: {d2_aft.get('best_time', 'Great spot to close out your trip.')}")
        md.append(f"- **06:00 PM - Farewell Sunset Walk**")
        md.append(f"  - Grab some street eats and reflect on an amazing weekend.\n")

        md.append("---\n")
        md.append("## Estimated Budget Breakdown")
        md.append("| Category | Estimated Cost (USD) | Details |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| **Lodging** | ${hotel_cost_2_nights:.2f} | 2 Nights at {hotel_name} |")
        md.append(f"| **Attractions** | ${attr_total_cost:.2f} | Entry fees for sights |")
        md.append(f"| **Food & Dining** | ${food_est:.2f} | Estimated at $45/day |")
        md.append(f"| **Local Transport & Misc** | ${misc_est:.2f} | Subways, taxis, and small treats |")
        md.append(f"| **Total Estimated** | **${total_estimate:.2f}** | |")
        if budget_status:
            md.append(f"\n{budget_status}")

        # RAG insights
        md.append("\n---\n")
        md.append("## Deep Local Insights & Guidelines (Multi-Hop RAG)")
        for key, snippet in TRAVEL_BLOGS.items():
            if city.lower().strip() in key or \
               ("nakamise" in key and city.lower().strip() == "tokyo") or \
               ("tanjong" in key and city.lower().strip() == "singapore"):
                md.append(f"\n- **{key.replace('_', ' ').title()}**:")
                md.append(f"  > *{snippet}*")

        return "\n".join(md)
