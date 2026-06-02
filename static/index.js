/* VOYAGEAGENT MAIN CONTROLLER */

document.addEventListener("DOMContentLoaded", () => {
    
    // --- INITIALIZE MARKDOWN ENGINE ---
    const md = window.markdownit({
        html: true,
        linkify: true,
        breaks: true
    });

    // --- DOM ELEMENT REFERENCES ---
    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const btnSendChat = document.getElementById("btn-send-chat");
    const btnResetSession = document.getElementById("btn-reset-session");
    const chatSuggestions = document.getElementById("chat-suggestions");
    const typingIndicator = document.getElementById("typing-indicator");
    
    const apiStatus = document.getElementById("api-status");
    const apiStatusText = document.getElementById("api-status-text");
    
    const constraintTags = document.getElementById("constraint-tags");
    const planningTimeline = document.getElementById("planning-timeline");
    const ragHopsLog = document.getElementById("rag-hops-log");
    
    const inputNewPreference = document.getElementById("input-new-preference");
    const btnAddPreference = document.getElementById("btn-add-preference");
    const preferencesList = document.getElementById("preferences-list");
    const btnWipeMemory = document.getElementById("btn-wipe-memory");
    
    const btnSettings = document.getElementById("btn-settings");
    const settingsModal = document.getElementById("settings-modal");
    const btnCloseSettings = document.getElementById("btn-close-settings");
    const geminiKeyInput = document.getElementById("gemini-key-input");
    const btnSaveSettings = document.getElementById("btn-save-settings");
    const keyValidationMsg = document.getElementById("key-validation-msg");

    let isRequestActive = false;

    // --- INITIALIZATION ACTIONS ---
    fetchConfig();
    fetchPreferences();

    // --- EVENT LISTENERS ---

    // Chat sending
    btnSendChat.addEventListener("click", handleSendMessage);
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Reset Chat
    btnResetSession.addEventListener("click", async () => {
        if (confirm("Are you sure you want to reset the active chat history? This clears short-term memory but keeps FAISS long-term preferences.")) {
            try {
                const res = await fetch("/api/clear?session=true&long_term=false", { method: "POST" });
                if (res.ok) {
                    chatMessages.innerHTML = `
                        <div class="message assistant-message message-initial">
                            <div class="avatar"><i class="fa-solid fa-robot"></i></div>
                            <div class="message-content">
                                <p>Chat session reset! Short-term memory has been cleared.</p>
                                <p>Where would you like to plan your next 2-day trip?</p>
                            </div>
                        </div>
                    `;
                    // Clear constraints and timeline
                    constraintTags.innerHTML = '<span class="empty-msg">No active trip planning constraints detected yet.</span>';
                    planningTimeline.innerHTML = '<li class="timeline-empty">Awaiting user query to formulate plan...</li>';
                    ragHopsLog.innerHTML = '<span class="empty-msg">RAG retrieval logs will appear here.</span>';
                }
            } catch (err) {
                console.error("Error resetting session:", err);
            }
        }
    });

    // Wipe FAISS Database
    btnWipeMemory.addEventListener("click", async () => {
        if (confirm("CRITICAL WARNING!\n\nThis will permanently wipe out all preferences indexed in your FAISS vector database. Are you sure?")) {
            try {
                const res = await fetch("/api/clear?session=false&long_term=true", { method: "POST" });
                if (res.ok) {
                    fetchPreferences();
                    fetchConfig();
                    alert("FAISS vector database successfully wiped out!");
                }
            } catch (err) {
                console.error("Error wiping memory:", err);
            }
        }
    });

    // Quick suggestions clicks
    chatSuggestions.addEventListener("click", (e) => {
        const btn = e.target.closest(".suggest-btn");
        if (btn) {
            chatInput.value = btn.getAttribute("data-prompt");
            chatInput.focus();
            handleSendMessage();
        }
    });

    // Manual Add Preference to FAISS
    btnAddPreference.addEventListener("click", async () => {
        const prefText = inputNewPreference.value.trim();
        if (!prefText) return;
        
        try {
            const res = await fetch("/api/preferences", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ preference: prefText })
            });
            if (res.ok) {
                inputNewPreference.value = "";
                fetchPreferences();
                fetchConfig();
            } else {
                const errData = await res.json();
                alert(`Error adding preference: ${errData.detail}`);
            }
        } catch (err) {
            console.error("Error manually adding preference:", err);
        }
    });

    // Settings Modal toggles
    btnSettings.addEventListener("click", () => {
        settingsModal.classList.remove("hidden");
        keyValidationMsg.classList.add("hidden");
    });

    btnCloseSettings.addEventListener("click", () => {
        settingsModal.classList.add("hidden");
    });

    settingsModal.addEventListener("click", (e) => {
        if (e.target === settingsModal) {
            settingsModal.classList.add("hidden");
        }
    });

    // Save Settings (API Key update)
    btnSaveSettings.addEventListener("click", async () => {
        const apiKey = geminiKeyInput.value.trim();
        
        btnSaveSettings.disabled = true;
        btnSaveSettings.innerText = "Saving & Connecting...";
        
        try {
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ api_key: apiKey })
            });
            
            const data = await res.json();
            
            keyValidationMsg.classList.remove("hidden");
            if (res.ok) {
                keyValidationMsg.className = "validation-msg msg-success";
                keyValidationMsg.innerHTML = `<i class="fa-solid fa-circle-check"></i> Agent successfully updated! Running with **Gemini 1.5 Flash**.`;
                geminiKeyInput.value = apiKey; // preserve
                
                // Update header status badge
                apiStatus.className = "api-status-badge mode-active";
                apiStatusText.innerText = "🔋 Gemini Active Mode";
                
                setTimeout(() => {
                    settingsModal.classList.add("hidden");
                }, 1500);
            } else {
                keyValidationMsg.className = "validation-msg msg-error";
                keyValidationMsg.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Error updating agent: ${data.detail}`;
            }
        } catch (err) {
            console.error("Error saving API config:", err);
            keyValidationMsg.className = "validation-msg msg-error";
            keyValidationMsg.innerText = "Network connection failure.";
        } finally {
            btnSaveSettings.disabled = false;
            btnSaveSettings.innerText = "Save Settings & Update Agent";
            fetchConfig();
        }
    });


    // --- ACTION HANDLERS & API CONNECTORS ---

    async function handleSendMessage() {
        const message = chatInput.value.trim();
        if (!message || isRequestActive) return;

        isRequestActive = true;
        chatInput.value = "";
        
        // 1. Append User Message
        appendMessage("user", message);
        
        // 2. Trigger Typing State
        typingIndicator.classList.remove("hidden");
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            // 3. POST Chat Query
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: message })
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Agent planning error");
            }

            const data = await res.json();
            
            // 4. Render Agent Response
            appendMessage("assistant", data.response);
            
            // 5. Update visual logs (Timeline, Constraints, RAG hops)
            updateConstraintsUI(data.parameters, data.preferences_used);
            updateTimelineUI(data.planning_steps);
            updateRagHopsUI(data.hops_log);
            
            // 6. Refresh long-term memory panel (since agent could auto-save new preferences)
            fetchPreferences();
            fetchConfig();

        } catch (err) {
            console.error("Error getting agent response:", err);
            appendMessage("assistant", `**System Error:** Failed to connect to Travel Agent. Details: \`${err.message}\``);
        } finally {
            typingIndicator.classList.add("hidden");
            isRequestActive = false;
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    function appendMessage(role, text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role}-message`;
        
        const avatarDiv = document.createElement("div");
        avatarDiv.className = "avatar";
        avatarDiv.innerHTML = role === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        
        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        
        // Render Markdown safely into HTML
        contentDiv.innerHTML = md.render(text);
        
        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Dynamic tags for constraints
    function updateConstraintsUI(params, prefsUsed) {
        constraintTags.innerHTML = "";
        
        const tags = [];
        
        if (params.city) {
            tags.push(`<span class="badge-tag tag-city"><i class="fa-solid fa-map-location-dot"></i> ${params.city}</span>`);
        }
        if (params.budget) {
            tags.push(`<span class="badge-tag tag-budget"><i class="fa-solid fa-wallet"></i> Max $${params.budget}</span>`);
        }
        if (params.pet_friendly) {
            tags.push(`<span class="badge-tag tag-pet"><i class="fa-solid fa-paw"></i> Pet-Friendly</span>`);
        }
        
        if (params.interests && params.interests.length > 0) {
            params.interests.forEach(interest => {
                tags.push(`<span class="badge-tag tag-interest"><i class="fa-solid fa-tag"></i> ${interest}</span>`);
            });
        }
        
        if (tags.length === 0) {
            constraintTags.innerHTML = '<span class="empty-msg">No active trip planning constraints detected yet.</span>';
        } else {
            constraintTags.innerHTML = tags.join("");
        }
    }

    // Step-by-step timeline updater
    function updateTimelineUI(steps) {
        planningTimeline.innerHTML = "";
        if (!steps || steps.length === 0) {
            planningTimeline.innerHTML = '<li class="timeline-empty">Awaiting user query to formulate plan...</li>';
            return;
        }
        
        steps.forEach((step, index) => {
            const li = document.createElement("li");
            li.style.animationDelay = `${index * 0.1}s`;
            // Capitalize or add custom icon hints based on text content
            li.innerHTML = step.replace(/Sub-task \d+:/, (match) => `<strong>${match}</strong>`);
            planningTimeline.appendChild(li);
        });
    }

    // RAG hops text loader
    function updateRagHopsUI(hops) {
        ragHopsLog.innerHTML = "";
        if (!hops || hops.length === 0) {
            ragHopsLog.innerHTML = '<span class="empty-msg">RAG retrieval logs will appear here.</span>';
            return;
        }
        
        hops.forEach(hop => {
            const div = document.createElement("div");
            div.className = "rag-hop-line";
            div.innerText = hop;
            ragHopsLog.appendChild(div);
        });
    }

    // Fetch and sync FAISS preferences list
    async function fetchPreferences() {
        try {
            const res = await fetch("/api/preferences");
            if (res.ok) {
                const prefs = await res.json();
                preferencesList.innerHTML = "";
                
                if (prefs.length === 0) {
                    preferencesList.innerHTML = '<li class="pref-empty">No preferences stored in long-term memory yet.</li>';
                    return;
                }
                
                prefs.forEach(item => {
                    const li = document.createElement("li");
                    li.className = "pref-item";
                    
                    const textSpan = document.createElement("span");
                    textSpan.className = "pref-item-text";
                    textSpan.innerText = item.preference;
                    
                    const deleteBtn = document.createElement("button");
                    deleteBtn.className = "pref-delete-btn";
                    deleteBtn.title = "Delete preference from FAISS Vector DB";
                    deleteBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>';
                    deleteBtn.addEventListener("click", () => handleDeletePreference(item.index));
                    
                    li.appendChild(textSpan);
                    li.appendChild(deleteBtn);
                    preferencesList.appendChild(li);
                });
            }
        } catch (err) {
            console.error("Error loading preferences from vector db:", err);
        }
    }

    async function handleDeletePreference(index) {
        try {
            const res = await fetch(`/api/preferences/${index}`, { method: "DELETE" });
            if (res.ok) {
                fetchPreferences();
                fetchConfig();
            }
        } catch (err) {
            console.error("Error deleting preference from FAISS:", err);
        }
    }

    // Fetch dynamic config parameters
    async function fetchConfig() {
        try {
            const res = await fetch("/api/config");
            if (res.ok) {
                const data = await res.json();
                
                if (data.llm_available) {
                    apiStatus.className = "api-status-badge mode-active";
                    apiStatusText.innerText = "🔋 Gemini Active Mode";
                } else {
                    apiStatus.className = "api-status-badge mode-simulated";
                    apiStatusText.innerText = "🔌 Local Simulation Mode";
                }
            }
        } catch (err) {
            console.error("Error fetching api settings:", err);
        }
    }

    // --- MOBILE TABS NAVIGATION CONTROLLER ---
    const mobileTabBtns = document.querySelectorAll(".mobile-tab-btn");
    const panelLeft = document.querySelector(".panel-left");
    const panelChat = document.querySelector(".panel-chat");
    const panelRight = document.querySelector(".panel-right");

    // Initialize active mobile panel
    if (panelChat) panelChat.classList.add("active-mobile");

    mobileTabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            // Remove active state from all mobile tab buttons
            mobileTabBtns.forEach(b => b.classList.remove("active"));
            // Add active state to clicked button
            btn.classList.add("active");
            
            // Get target panel class
            const target = btn.getAttribute("data-target");
            
            // Remove active-mobile from all panels
            if (panelLeft) panelLeft.classList.remove("active-mobile");
            if (panelChat) panelChat.classList.remove("active-mobile");
            if (panelRight) panelRight.classList.remove("active-mobile");
            
            // Add active-mobile to target panel
            if (target === "panel-left" && panelLeft) panelLeft.classList.add("active-mobile");
            if (target === "panel-chat" && panelChat) panelChat.classList.add("active-mobile");
            if (target === "panel-right" && panelRight) panelRight.classList.add("active-mobile");
        });
    });
});
