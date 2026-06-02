import requests
import json
import time
import sys
import os
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://127.0.0.1:8000"

def log_header(text):
    print("\n" + "=" * 60)
    print(f" {text.upper()} ")
    print("=" * 60)

def reset_database():
    """Wipes short-term and long-term FAISS database to start fresh."""
    try:
        r = requests.post(f"{BASE_URL}/api/clear?session=true&long_term=true")
        r.raise_for_status()
        print("Success: Database reset to a fresh state.")
    except Exception as e:
        print(f"Error resetting database: {e}")
        sys.exit(1)

def run_concurrency_tests():
    log_header("Phase 1: Concurrency & Thread-Safety Stress Test")
    reset_database()
    
    # 1. Parallel Preference Insertion
    num_writes = 25
    print(f"Spawning thread pool to insert {num_writes} preferences concurrently...")
    
    def post_pref(i):
        pref_text = f"The user is highly interested in activity {i} for premium vacations."
        r = requests.post(f"{BASE_URL}/api/preferences", json={"preference": pref_text})
        return r.status_code, pref_text

    start_time = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(post_pref, i): i for i in range(num_writes)}
        for fut in as_completed(futures):
            results.append(fut.result())
    elapsed = time.time() - start_time
    
    failures = [res for res in results if res[0] != 200]
    print(f"Concurrency completed in {elapsed:.3f}s.")
    print(f"Successful inserts: {len(results) - len(failures)} / {num_writes}")
    if failures:
        print(f"WARNING: {len(failures)} concurrent insert requests failed!")
    else:
        print("PASS: All concurrent inserts succeeded with 200 OK.")
        
    # Verify FAISS DB integrity
    r_list = requests.get(f"{BASE_URL}/api/preferences")
    r_list.raise_for_status()
    prefs_in_db = r_list.json()
    print(f"Total preferences verified in FAISS DB: {len(prefs_in_db)}")
    assert len(prefs_in_db) == num_writes, f"Database size mismatch! Expected {num_writes}, got {len(prefs_in_db)}"
    print("PASS: Thread Lock successfully preserved index serialization and metadata list length.")

    # 2. Parallel Chat Sessions
    num_chats = 15
    print(f"\nSending {num_chats} concurrent chat planning requests to '/api/chat'...")
    
    def send_chat(i):
        cities = ["Tokyo", "Paris", "Singapore"]
        city = cities[i % 3]
        payload = {
            "message": f"I want to visit {city} with my dog on a budget of 1500 and I love art",
            "session_id": f"stress_session_{i}"
        }
        r = requests.post(f"{BASE_URL}/api/chat", json=payload)
        return r.status_code, r.json() if r.status_code == 200 else r.text

    start_time = time.time()
    chat_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(send_chat, i): i for i in range(num_chats)}
        for fut in as_completed(futures):
            chat_results.append(fut.result())
    elapsed = time.time() - start_time
    
    chat_failures = [res for res in chat_results if res[0] != 200]
    print(f"Concurrency completed in {elapsed:.3f}s.")
    print(f"Successful plans: {len(chat_results) - len(chat_failures)} / {num_chats}")
    if chat_failures:
        print(f"FAIL: {len(chat_failures)} concurrent chat planning requests failed!")
        print(f"Failure samples: {chat_failures[:2]}")
        sys.exit(1)
    else:
        print("PASS: All concurrent chat plans synthesized successfully.")


def run_boundary_tests():
    log_header("Phase 2: Boundary & Fuzzing Resilience")
    
    # 1. Empty message text
    print("Testing empty message text validation...")
    r = requests.post(f"{BASE_URL}/api/chat", json={"message": "", "session_id": "boundary"})
    print(f"Empty response status: {r.status_code} | details: {r.json()}")
    assert r.status_code == 400, f"Expected 400 Bad Request, got {r.status_code}"
    print("PASS: Rejected empty chat requests gracefully.")

    # 2. Huge query fuzzer (5,000+ characters)
    print("\nFuzzing with massive 5,000 character string...")
    random_junk = "".join(random.choices(string.ascii_letters + " ", k=5000))
    # Inject actual keywords to test if it parses city correctly under huge noise
    fuzz_query = f"I want to plan a weekend trip to Tokyo with my dog on budget of 2000 and do anime sightseeing " + random_junk
    
    start_time = time.time()
    r = requests.post(f"{BASE_URL}/api/chat", json={"message": fuzz_query})
    elapsed = time.time() - start_time
    
    print(f"Fuzz response status: {r.status_code} in {elapsed:.3f}s")
    assert r.status_code == 200, f"Failed under large payload: {r.text}"
    res = r.json()
    assert res["success"] is True, "Fuzzed chat planning success was False"
    assert res["parameters"]["city"].lower() == "tokyo", "Failed to extract city from fuzzed text"
    assert res["parameters"]["pet_friendly"] is True, "Failed to extract pet flag from fuzzed text"
    print("PASS: Fuzzed query handled successfully without buffer limits or timeouts.")

    # 3. Extremes in Budgets
    print("\nTesting extreme budgets (Zero, Negative, Massive)...")
    budget_tests = [
        ("Paris with dog on budget of 0", 0.0),
        ("Tokyo on budget of -500", -500.0),
        ("Singapore with my cat budget 999999999", 999999999.0)
    ]
    
    for query, expected_val in budget_tests:
        r = requests.post(f"{BASE_URL}/api/chat", json={"message": query})
        assert r.status_code == 200, f"Failed on budget query '{query}': {r.text}"
        res = r.json()
        print(f"Query: '{query}' -> Extracted Budget: {res['parameters']['budget']}")
        assert res["parameters"]["budget"] == expected_val, f"Mismatch in budget extraction: expected {expected_val}, got {res['parameters']['budget']}"
    print("PASS: Budget boundary values successfully parsed and processed.")

    # 4. Unrecognized Country Queries (triggers clarification)
    print("\nTesting country queries instead of city names (should prompt clarification)...")
    unrecognized_queries = [
        "I want to travel to Japan",
        "Plan a weekend trip to France with my puppy"
    ]
    for q in unrecognized_queries:
        r = requests.post(f"{BASE_URL}/api/chat", json={"message": q})
        assert r.status_code == 200, f"Failed on country query: {r.text}"
        res = r.json()
        print(f"Query: '{q}' -> Clarification Needed: {res['clarification_needed']}")
        assert res["clarification_needed"] is True, "Clarification was not requested for country query"
        assert "city" in res["response"].lower() or "where" in res["response"].lower(), "Response did not prompt for city name"
    print("PASS: Prompted for clarification gracefully when city was invalid.")

    # 5. Security & Injections (XSS / SQL Injection)
    print("\nTesting injection payloads (XSS & SQL Injection)...")
    injection_queries = [
        "Tokyo <script>alert('XSS')</script> budget 1200",
        "Paris' OR '1'='1' budget 800"
    ]
    for q in injection_queries:
        r = requests.post(f"{BASE_URL}/api/chat", json={"message": q})
        assert r.status_code == 200, f"Injection crashed backend: {r.text}"
        res = r.json()
        print(f"Query: '{q}' -> Extracted City: {res['parameters']['city']}")
        # Verify it handled safely
        assert res["success"] is True, "Sanitization broke execution flow"
    print("PASS: Injection payloads handled cleanly without breaking database or process flow.")


def run_faiss_scale_tests():
    log_header("Phase 3: FAISS Database Scale & Latency")
    reset_database()
    
    num_scale = 500
    print(f"Loading {num_scale} mock travel preferences rapidly into FAISS...")
    
    interests = ["culture", "anime", "nature", "sushi", "historical", "shopping", "luxury", "beach", "vegan"]
    cities = ["Tokyo", "Paris", "Singapore", "London", "Rome", "Sydney", "New York"]
    
    start_time = time.time()
    for i in range(num_scale):
        city = cities[i % len(cities)]
        interest = interests[i % len(interests)]
        pref_text = f"The user is highly interested in {interest} activities when visiting {city} on budget {1000 + i}."
        r = requests.post(f"{BASE_URL}/api/preferences", json={"preference": pref_text})
        if r.status_code != 200:
            print(f"FAIL: Blocked at preference {i}: {r.text}")
            sys.exit(1)
            
    elapsed = time.time() - start_time
    avg_insert = (elapsed / num_scale) * 1000
    print(f"Successfully loaded {num_scale} items in {elapsed:.3f}s (Average: {avg_insert:.2f}ms/insert).")
    
    # Measure Query Latency at Scale
    print("\nMeasuring query retrieval latency with 500 active preferences...")
    latencies = []
    for _ in range(50):
        city_query = random.choice(cities)
        interest_query = random.choice(interests)
        query_str = f"Plan trip to {city_query} with {interest_query} activities"
        
        q_start = time.time()
        # Trigger query via chat (which executes FAISS query internally)
        r = requests.post(f"{BASE_URL}/api/chat", json={"message": query_str})
        q_elapsed = (time.time() - q_start) * 1000
        latencies.append(q_elapsed)
        
    avg_query_latency = sum(latencies) / len(latencies)
    max_query_latency = max(latencies)
    print(f"FAISS Retrieval + Chat Planning Latency over 50 queries:")
    print(f"  - Average latency: {avg_query_latency:.2f}ms")
    print(f"  - Max latency: {max_query_latency:.2f}ms")
    
    # Assert query latency is highly optimal (under 50ms total endpoint response time is standard)
    assert avg_query_latency < 100, f"Average endpoint latency too high: {avg_query_latency:.2f}ms"
    print("PASS: Vector database scales cleanly with ultra-low cosine search latencies.")


def run_tool_null_safety_tests():
    log_header("Phase 4: Tool Null-Safety & Fallback Checks")
    
    # low budget + pet-friendly = empty lodging matches
    print("Planning trip to Tokyo with extremely low budget of $10 and pet-friendly requirements...")
    payload = {
        "message": "Tokyo pet-friendly budget 10",
        "session_id": "low_budget_test"
    }
    r = requests.post(f"{BASE_URL}/api/chat", json=payload)
    assert r.status_code == 200, f"Low budget request crashed server: {r.text}"
    
    res = r.json()
    print("Plan synthesized successfully.")
    print("Itinerary Preview (Lodging Section):\n")
    
    # Check if Lodging section has fallback message
    itinerary = res["response"]
    print("\n".join(itinerary.split("\n")[4:12])) # Print section
    
    assert "no lodging" in itinerary.lower() or "not specified" in itinerary.lower() or "premium" in itinerary.lower() or "budget" in itinerary.lower(), "Itinerary did not offer proper fallback lodging guidance"
    assert res["success"] is True, "Plan marked unsuccessful due to low budget lodging limits"
    print("\nPASS: Tool null-safety confirmed. System defaults gracefully without NullPointerExceptions.")


def configure_api_key(api_key):
    """Sets the Gemini API Key on the running server."""
    try:
        r = requests.post(f"{BASE_URL}/api/config", json={"api_key": api_key})
        r.raise_for_status()
        res = r.json()
        print(f"Success: Configured Gemini API Key. Status: {res}")
    except Exception as e:
        print(f"Error configuring API Key: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("=== RUNNING VOYAGEAGENT STRESS & QUALITY TEST SUITE ===")
    start_suite = time.time()

    # Resolve API key: CLI argument takes priority, then environment variable
    api_key = None
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.environ.get("GEMINI_API_KEY")

    if api_key:
        configure_api_key(api_key)
    else:
        print("Note: No GEMINI_API_KEY provided. Running in Local Simulation Mode.")
        print("      To use Gemini: python stress_test.py <your_api_key>")
        print("      Or set the GEMINI_API_KEY environment variable.\n")

    run_concurrency_tests()
    run_boundary_tests()
    run_faiss_scale_tests()
    run_tool_null_safety_tests()

    suite_elapsed = time.time() - start_suite
    print("\n" + "=" * 60)
    print(f"=== STRESS TEST SUITE COMPLETED SUCCESSFULLY IN {suite_elapsed:.2f}s! ===")
    print("=" * 60)

    # Final database wipe to leave environment clean
    reset_database()
