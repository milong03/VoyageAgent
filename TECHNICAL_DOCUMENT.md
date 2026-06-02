# VoyageAgent: Technical Deliverables & Architecture Document

This document serves as the formal **Brief Technical Document and Project Summary** for the **VoyageAgent** travel planning AI assistant, fulfilling the assessment deliverables. The complete codebase is pushed and live at:
**GitHub Repository**: [https://github.com/milong03/VoyageAgent](https://github.com/milong03/VoyageAgent)

---

## 1. Modular Codebase & File Architecture

VoyageAgent is built using a highly decoupled, modular design pattern to ensure scalability, ease of testing, and maintenance.

```text
c:\Users\Administrator\Desktop\intern\
├── main.py                 # FastAPI Application Server & REST Endpoints
├── Dockerfile              # Docker Container configuration for Cloud Deployment
├── requirements.txt        # Python dependency specification
├── .gitignore              # Ignores sensitive keys, pycache, and virtual envs
├── README.md               # Quick setup, installation, and run guide
├── TECHNICAL_DOCUMENT.md   # This detailed technical deliverables document
│
├── agent/                  # Central AI Agent Logic Package
│   ├── planner.py          # Plan-and-Execute planner & response compiler
│   ├── memory.py           # Short-term session & FAISS Vector database memory
│   ├── rag.py              # Multi-Hop RAG engine & local blogs corpus
│   └── tools.py            # Weather, accommodation, attractions, & search APIs
│
├── data/                   # Core Static Datastore
│   ├── attractions.json    # Dest base (Tokyo, Paris, Singapore weather/hotels)
│   └── memory/             # Local directory where FAISS binary store persists
│
└── static/                 # Single-Page Web UI Assets (Glassmorphism)
    ├── index.html          # Responsive skeleton & mobile navigation tab layout
    ├── index.css           # Styling system (glass panels, timeline animations)
    └── index.js            # Controller (chat loop, FAISS CRUD, mobile tab toggler)
```

---

## 2. REST API & Interactive UI Demo

VoyageAgent exposes a complete REST API alongside an interactive single-page dashboard.

### Interactive REST API Endpoints
* **`POST /api/chat`**: Handles conversation queries, extracts parameters, coordinates planning sub-tasks, triggers tools/RAG, auto-persists preferences in FAISS, and returns a compiled itinerary.
* **`GET /api/preferences`**: Retrieves all active vector preferences from the FAISS database.
* **`POST /api/preferences`**: Manually indexes a travel preference inside the FAISS database.
* **`DELETE /api/preferences/{index}`**: Deletes a specific indexed preference card and rebuilds the FAISS database on-the-fly.
* **`POST /api/config`**: Dynamically updates the Google Gemini API key at runtime.
* **`GET /api/config`**: Inspects active backend configurations (Gemini API status).
* **`POST /api/clear`**: Wipes conversational short-term history or clears the FAISS binary index.

> **Swagger Playground**: Try the endpoints in real-time under the auto-generated Swagger UI at `/docs` (e.g. `http://127.0.0.1:8000/docs`).

### Premium Glassmorphism UI SPA
* **Desktop Grid**: Displays an elegant 3-column cosmic slate layout—Reasoning Timeline on the left, Chat Console in the middle, and FAISS Vector Memory Manager on the right.
* **Mobile-Responsive Tabbing (<= 1024px)**: Collapses column spans and injects a premium mobile tab bar (Chat | Reasoning | FAISS Memory), allowing a 100% full-feature mobile experience.
* **Typing Indicator & Timeline**: Renders micro-animations for the planning steps and Multi-Hop RAG trace, bringing the agent's inner workings alive.

---

## 3. System Architecture & Core Flow

VoyageAgent coordinates memory, RAG, and tools through a centralized **Plan-and-Execute Loop**:

```text
                  +----------------------------------+
                  |           User Query             |
                  +----------------------------------+
                                   |
                                   v
                  +----------------------------------+
                  |      Parameter Extraction        |
                  |  (City, Budget, Pets, Interests) |
                  +----------------------------------+
                                   |
                                   v
                  +----------------------------------+
                  |  FAISS Memory Lookup (Cosine)    |
                  |  (Overrides constraints in context)|
                  +----------------------------------+
                                   |
                                   v
                  +----------------------------------+
                  |       Sub-Task Scheduler         |
                  +----------------------------------+
                      /       /    |    \       \
                     /       /     |     \       \
                    v       v      v      v       v
         +------------+ +-------+ +----+ +-------+ +-----------+
         | Multi-Hop  | |Weather| |Curr| | Lodging | |Attractions|
         | RAG Search | |  API  | | API| |   DB    | |    DB     |
         +------------+ +-------+ +----+ +-------+ +-----------+
                     \       \     |     /       /
                      \       \    |    /       /
                       v       v   v   v       v
                  +----------------------------------+
                  |     Plan Synthesis & Rendering   |
                  |          (Gemini 2.5)            |
                  +----------------------------------+
                                   |
                                   v
                  +----------------------------------+
                  |     Auto-Persist in FAISS        |
                  |   (Saves new query constraints)  |
                  +----------------------------------+
                                   |
                                   v
                  +----------------------------------+
                  |         Chat Response            |
                  +----------------------------------+
```

---

## 4. Tool Design & Selection

To explicitly satisfy the assessment rubric requiring at least 3 dynamic tools (and allowing mock DBs), VoyageAgent implements **four** distinct tools, seamlessly mixing live 100% real network APIs with local DB filters inside [agent/tools.py](file:///c:/Users/Administrator/Desktop/intern/agent/tools.py):

1. **Live Weather API (`get_weather`)**: 
   - Dynamically calls the **Open-Meteo Geocoding API** to translate city strings into exact coordinates.
   - Calls the **Open-Meteo Forecast API** to pull real-time live temperature, precipitation probability, and weather codes.
2. **Live Currency API (`get_currency_exchange`)**:
   - Dynamically calls the live **ExchangeRate-API** to pull exact daily conversion rates for USD to EUR, JPY, SGD, GBP, and CNY, enabling the LLM to perform accurate foreign currency budgeting.
3. **Attraction & Hotel DB Tools (`search_attractions` & `get_accommodation`)**: 
   - Fulfilling the rubric's "Attraction DB / Mock API" clause, these tools dynamically filter a dense local JSON corpus based on budget constraints, user interest categories, and pet policies.
4. **Live Wikipedia RAG API (`MultiHopRAG`)**: 
   - Executes live `urllib` HTTP requests against `en.wikipedia.org` to fetch actual encyclopedic knowledge in real-time, extracting cultural tips and etiquette.

---

## 5. Double-Layer Memory Design

To remember user preferences during chat turns and across active user sessions, VoyageAgent features a robust **Dual-Layer Memory**:

### A. Short-Term Memory (Conversational Context)
Saves a rolling list of `user` and `assistant` messages inside a short-term list, ensuring the AI can answer follow-up queries, clarify inputs, and maintain continuous context.

### B. Long-Term Memory (FAISS Vector Database)
Allows the agent to remember traveler profiles across multiple sessions and reboots:
1. **Local Dense Vectorizer**: To run completely offline and self-contained without costly API calls, we developed a semantic vectorizer. It maps raw preference strings onto a **128-dimensional travel concept space** (e.g. budget, pet-friendly, luxury, nature, anime, etc.).
2. **Cosine Similarity**: Vectorized inputs are searched inside a **FAISS `IndexFlatIP` (Inner Product)** index, which performs Cosine Similarity matching against user queries.
3. **Persistence**: Saves the index binary file to `data/memory/preferences.index` and metadata lists to `data/memory/preferences.json`, instantly reloading them on startup.
4. **Context Injection**: When a matching preference is found (e.g., score > 0.40), the planner **automatically overrides constraint parameters** in the prompt context (e.g. enforcing pet-friendly rules even if the user forgot to mention their dog in the current session).

---

## 6. Multi-Hop RAG Mechanism

For deep and nuanced travel suggestions, VoyageAgent bypasses simple search queries in favor of a **Two-Stage Multi-Hop RAG**:

* **Hop 1 (Primary Retrieval)**: The agent connects to the live Wikipedia API to fetch summaries of the top articles matching the target city and user's interests. If Wikipedia is unreachable, it seamlessly falls back to querying a local corpus of JSON travel blogs.
* **Hop 2 (Secondary Entity Expansion)**: It parses the raw text retrieved from Hop 1 using a regex entity extractor to identify hidden proper nouns and landmarks (e.g., extracting *Nakamise-dori* from an article about *Senso-ji*, or *Tanjong Beach* from *Sentosa*). It then triggers a secondary, automated Wikipedia API search for each extracted sub-entity to gather profound, highly specific local context.
* **Synthesis**: It bundles the broad city context (Hop 1) and the deep landmark context (Hop 2) together, injecting both layers into the final LLM prompt.
* **RAG Logs**: Captured hops are printed cleanly on the timeline, ensuring transparent agent reasoning traces.
