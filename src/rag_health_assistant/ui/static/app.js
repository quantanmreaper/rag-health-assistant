/* =========================================================
   AuraHealth AI - Frontend Controller & Interactive Tools
   Persistent chatbot sessions, profiles, and PDF export
   ========================================================= */

const chatState = {
  currentConversationId: null,
  sessionId: null,
  sessionType: "anonymous",
  messages: [],
  profile: null,
  conversations: [],
};

let localApiKey = localStorage.getItem("aurahealth_gemini_api_key") || "";

document.addEventListener("DOMContentLoaded", () => {
  initUI();
  fetchKBStats();
  setupEventListeners();
  initializeChat();

  if (localApiKey) {
    updateApiStatusBadge(true);
  }
});

function initUI() {
  if (typeof marked !== "undefined") {
    marked.setOptions({ breaks: true, gfm: true });
  }
}

function getCookie(name) {
  const match = document.cookie.match(new RegExp("(^| )" + name + "=([^;]+)"));
  return match ? decodeURIComponent(match[2]) : null;
}

function setCookie(name, value, days = 30) {
  const expires = new Date(Date.now() + days * 864e5).toUTCString();
  document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/; SameSite=Lax`;
}

function getSessionIdFromCookie() {
  return getCookie("aurahealth_session_id");
}

async function initializeChat() {
  try {
    const existing = getSessionIdFromCookie();
    const res = await fetch("/api/session/init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: existing || undefined }),
    });
    const data = await res.json();
    chatState.sessionId = data.session_id;
    chatState.sessionType = data.session_type || "anonymous";
    setCookie("aurahealth_session_id", chatState.sessionId);
    updateModeIndicator(chatState.sessionType);

    await loadConversations();
    await loadProfile();
  } catch (err) {
    console.error("initializeChat failed", err);
  }
}

function updateModeIndicator(sessionType) {
  const el = document.getElementById("modeIndicator");
  if (!el) return;
  const authenticated = sessionType === "authenticated";
  el.textContent = authenticated ? "Authenticated Mode" : "Anonymous Mode";
  el.classList.toggle("authenticated", authenticated);
}

async function loadConversations() {
  if (!chatState.sessionId) return;
  try {
    const res = await fetch(
      `/api/chat/sessions?session_id=${encodeURIComponent(chatState.sessionId)}`
    );
    const data = await res.json();
    chatState.conversations = data.sessions || [];
    renderConversationList();
  } catch (err) {
    console.error("loadConversations failed", err);
  }
}

function renderConversationList() {
  const list = document.getElementById("conversationList");
  if (!list) return;
  list.innerHTML = "";

  if (!chatState.conversations.length) {
    list.innerHTML = `<div class="conversation-empty">No conversations yet</div>`;
    return;
  }

  chatState.conversations.forEach((conv) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "conversation-item";
    item.setAttribute("role", "listitem");
    if (conv.conversation_id === chatState.currentConversationId) {
      item.classList.add("active");
    }
    const title = conv.title || "Untitled conversation";
    const updated = conv.updated_at
      ? new Date(conv.updated_at).toLocaleString()
      : "";
    item.innerHTML = `
      <span class="conversation-item-title">${escapeHtml(title)}</span>
      <span class="conversation-item-meta">${escapeHtml(updated)} · ${conv.message_count || 0} msgs</span>
    `;
    item.addEventListener("click", () => loadConversation(conv.conversation_id));
    list.appendChild(item);
  });
}

async function loadConversation(conversationId) {
  if (!chatState.sessionId || !conversationId) return;
  try {
    const res = await fetch(
      `/api/chat/history/${encodeURIComponent(conversationId)}?session_id=${encodeURIComponent(chatState.sessionId)}`
    );
    if (!res.ok) throw new Error("Failed to load conversation");
    const conversation = await res.json();
    chatState.currentConversationId = conversationId;
    chatState.messages = conversation.messages || [];
    updateConversationTitle(
      (conversation.metadata && conversation.metadata.title) ||
        conversation.title ||
        "Conversation"
    );
    renderMessages();
    renderConversationList();
  } catch (err) {
    console.error("loadConversation failed", err);
  }
}

function updateConversationTitle(title) {
  const el = document.getElementById("conversationTitle");
  if (el) el.textContent = title || "New Conversation";
}

function renderMessages() {
  const container = document.getElementById("chatMessages");
  if (!container) return;
  container.innerHTML = "";

  if (!chatState.messages.length) {
    container.innerHTML = document.getElementById("welcomeMessage")
      ? ""
      : "";
    // Re-inject a simple welcome if empty
    addMessageToUI({
      role: "assistant",
      content:
        "Welcome back. Ask about diabetes, blood pressure, diet, medications, or share your readings.",
      timestamp: new Date().toISOString(),
      _welcome: true,
    });
    return;
  }

  chatState.messages.forEach((msg) => addMessageToUI(msg, false));
  scrollToBottom();
}

async function startNewConversation() {
  if (!chatState.sessionId) await initializeChat();
  try {
    const res = await fetch("/api/chat/sessions/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: chatState.sessionId }),
    });
    const data = await res.json();
    chatState.currentConversationId = data.conversation_id;
    chatState.messages = [];
    updateConversationTitle("New Conversation");
    const container = document.getElementById("chatMessages");
    if (container) {
      container.innerHTML = "";
      addMessageToUI({
        role: "assistant",
        content:
          "New conversation started. How can I help with your diabetes or blood pressure today?",
        timestamp: new Date().toISOString(),
        _welcome: true,
      });
    }
    await loadConversations();
  } catch (err) {
    console.error("startNewConversation failed", err);
  }
}

async function sendMessage(content) {
  if (!content || !content.trim()) return;
  if (!chatState.sessionId) await initializeChat();

  const sendBtn = document.getElementById("sendBtn");
  if (sendBtn) sendBtn.disabled = true;

  addMessageToUI({
    role: "user",
    content,
    timestamp: new Date().toISOString(),
  });

  const typingId = showTypingIndicator();

  try {
    const res = await fetch("/api/chat/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: content,
        conversation_id: chatState.currentConversationId || undefined,
        session_id: chatState.sessionId,
        api_key: localApiKey || undefined,
      }),
    });
    const raw = await res.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      removeTypingIndicator(typingId);
      addMessageToUI({
        role: "assistant",
        content: `⚠️ Backend error (${res.status}): ${raw.slice(0, 200)}`,
        timestamp: new Date().toISOString(),
      });
      return;
    }
    removeTypingIndicator(typingId);

    if (!res.ok) {
      addMessageToUI({
        role: "assistant",
        content: `⚠️ ${data.detail || data.message || "Request failed"}`,
        timestamp: new Date().toISOString(),
      });
      return;
    }

    if (data.session_id) {
      chatState.sessionId = data.session_id;
      setCookie("aurahealth_session_id", chatState.sessionId);
    }
    if (data.conversation_id) {
      chatState.currentConversationId = data.conversation_id;
    }

    addMessageToUI({
      role: "assistant",
      content: data.response,
      timestamp: data.timestamp || new Date().toISOString(),
      metadata: {
        tools_used: data.tools_used,
        emergency: data.emergency,
        guideline_sources: data.guideline_sources,
      },
    });

    await loadConversations();
  } catch (error) {
    removeTypingIndicator(typingId);
    addMessageToUI({
      role: "assistant",
      content: `⚠️ Could not connect to assistant backend: ${error.message}`,
      timestamp: new Date().toISOString(),
    });
  } finally {
    if (sendBtn) sendBtn.disabled = false;
  }
}

function addMessageToUI(message, shouldScroll = true) {
  const messagesContainer = document.getElementById("chatMessages");
  if (!messagesContainer) return;

  const role = message.role === "user" ? "user" : "assistant";
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const emergency =
    message.metadata &&
    message.metadata.emergency &&
    message.metadata.emergency.is_emergency;
  if (emergency) row.classList.add("emergency-message");

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role === "user" ? "user-avatar" : "ai-avatar"}`;
  avatar.innerHTML =
    role === "user"
      ? `<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>`
      : `<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const meta = document.createElement("div");
  meta.className = "message-meta";
  const sender = document.createElement("span");
  sender.className = "message-sender";
  sender.textContent = role === "user" ? "You" : "AuraHealth";
  const time = document.createElement("span");
  time.className = "message-time";
  time.textContent = formatTimestamp(message.timestamp);
  meta.appendChild(sender);
  meta.appendChild(time);
  bubble.appendChild(meta);

  if (emergency) {
    const alertDiv = document.createElement("div");
    alertDiv.className = "emergency-alert-card";
    const flags = (message.metadata.emergency.matched_flags || []).join(", ");
    alertDiv.innerHTML = `
      <strong>🚨 EMERGENCY ALERT: ${escapeHtml(flags)}</strong>
      <p style="margin-top: 4px;">${escapeHtml(message.metadata.emergency.alert_message || "")}</p>
    `;
    bubble.appendChild(alertDiv);
  }

  if (message.metadata && message.metadata.tools_used && message.metadata.tools_used.length) {
    const toolsBar = document.createElement("div");
    toolsBar.className = "tools-executed-bar";
    message.metadata.tools_used.forEach((t) => {
      const chip = document.createElement("span");
      chip.className = "tool-chip";
      const name = t.name || "";
      const icon = name.includes("guidelines")
        ? "📚"
        : name.includes("pressure")
          ? "🩺"
          : "🔬";
      chip.innerText = `${icon} ${name}`;
      toolsBar.appendChild(chip);
    });
    bubble.appendChild(toolsBar);
  }

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  const raw = message.content || "";
  contentDiv.innerHTML =
    typeof marked !== "undefined" ? marked.parse(raw) : `<p>${escapeHtml(raw)}</p>`;
  bubble.appendChild(contentDiv);

  if (message.metadata && message.metadata.guideline_sources) {
    const citPanel = document.createElement("div");
    citPanel.className = "citations-panel";
    citPanel.innerHTML = `<strong>Referenced Guidelines:</strong><br>`;
    message.metadata.guideline_sources.forEach((s) => {
      citPanel.innerHTML += `<span class="citation-tag">📄 ${escapeHtml(s.title)} (p. ${escapeHtml(String(s.page))})</span>`;
    });
    bubble.appendChild(citPanel);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesContainer.appendChild(row);
  if (shouldScroll) scrollToBottom();
}

function formatTimestamp(ts) {
  if (!ts) return "";
  try {
    return new Date(ts).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function scrollToBottom() {
  const messagesContainer = document.getElementById("chatMessages");
  if (messagesContainer) {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
}

function showTypingIndicator() {
  const messagesContainer = document.getElementById("chatMessages");
  const id = `typing_${Date.now()}`;
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.id = id;
  row.innerHTML = `
    <div class="avatar ai-avatar">
      <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
    </div>
    <div class="bubble">
      <div class="typing-dots" aria-label="AuraHealth is thinking">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  messagesContainer.appendChild(row);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

async function exportConversation() {
  if (!chatState.sessionId || !chatState.currentConversationId) {
    alert("Start or open a conversation before exporting.");
    return;
  }
  const url = `/api/export/conversation/${encodeURIComponent(
    chatState.currentConversationId
  )}?session_id=${encodeURIComponent(chatState.sessionId)}`;
  window.open(url, "_blank");
}

async function loadProfile() {
  if (!chatState.sessionId) return;
  try {
    const res = await fetch(
      `/api/profile?session_id=${encodeURIComponent(chatState.sessionId)}`
    );
    const profile = await res.json();
    chatState.profile = profile;
    renderProfileSummary(profile);
  } catch (err) {
    console.error("loadProfile failed", err);
  }
}

function renderProfileSummary(profile) {
  const dx = document.getElementById("profileDiagnoses");
  const meds = document.getElementById("profileMedications");
  const al = document.getElementById("profileAllergies");
  if (!dx || !meds || !al) return;

  const diagnoses = (profile.diagnoses || [])
    .map((d) => d.condition_name)
    .join(", ");
  const medications = (profile.medications || [])
    .map((m) => `${m.name} ${m.dosage}`)
    .join(", ");
  const allergies = (profile.allergies || [])
    .map((a) => a.allergen)
    .join(", ");

  dx.innerHTML = diagnoses
    ? `<strong>Diagnoses:</strong> ${escapeHtml(diagnoses)}`
    : `<span class="muted">No diagnoses yet</span>`;
  meds.innerHTML = medications
    ? `<strong>Meds:</strong> ${escapeHtml(medications)}`
    : `<span class="muted">No medications yet</span>`;
  al.innerHTML = allergies
    ? `<strong>Allergies:</strong> ${escapeHtml(allergies)}`
    : `<span class="muted">No allergies yet</span>`;
}

async function updateProfile(event) {
  event.preventDefault();
  if (!chatState.sessionId) return;

  const name = document.getElementById("profileName").value.trim();
  const ageRaw = document.getElementById("profileAge").value;
  const history = document.getElementById("profileHistory").value.trim();
  const updates = {};
  if (name) updates.name = name;
  if (ageRaw !== "") updates.age = parseInt(ageRaw, 10);
  if (history) updates.medical_history = history;

  try {
    if (Object.keys(updates).length) {
      await fetch("/api/profile/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: chatState.sessionId, updates }),
      });
    }

    const diagnosis = document.getElementById("profileDiagnosis").value.trim();
    if (diagnosis) {
      await fetch("/api/profile/diagnosis/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: chatState.sessionId,
          diagnosis: { condition_name: diagnosis },
        }),
      });
    }

    const medName = document.getElementById("profileMedName").value.trim();
    const medDose = document.getElementById("profileMedDose").value.trim();
    const medFreq = document.getElementById("profileMedFreq").value.trim();
    if (medName && medDose && medFreq) {
      await fetch("/api/profile/medication/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: chatState.sessionId,
          medication: {
            name: medName,
            dosage: medDose,
            frequency: medFreq,
          },
        }),
      });
    }

    const allergen = document.getElementById("profileAllergen").value.trim();
    const reaction = document.getElementById("profileReaction").value.trim();
    const severity = document.getElementById("profileSeverity").value;
    if (allergen && reaction) {
      await fetch("/api/profile/allergy/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: chatState.sessionId,
          allergy: { allergen, reaction, severity },
        }),
      });
    }

    document.getElementById("profileModal").classList.remove("active");
    await loadProfile();
  } catch (err) {
    alert(`Failed to update profile: ${err.message}`);
  }
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setupEventListeners() {
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSendMessage();
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
  });

  document.querySelectorAll(".prompt-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      chatInput.value = chip.dataset.prompt || chip.innerText.trim();
      chatInput.focus();
    });
  });

  const bpForm = document.getElementById("bpForm");
  if (bpForm) bpForm.addEventListener("submit", handleBPCheck);
  const glucoseForm = document.getElementById("glucoseForm");
  if (glucoseForm) glucoseForm.addEventListener("submit", handleGlucoseCheck);
  const hba1cForm = document.getElementById("hba1cForm");
  if (hba1cForm) hba1cForm.addEventListener("submit", handleHbA1cCheck);

  const settingsBtn = document.getElementById("settingsBtn");
  const settingsModal = document.getElementById("settingsModal");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const saveSettingsBtn = document.getElementById("saveSettingsBtn");
  const apiKeyInput = document.getElementById("apiKeyInput");

  if (settingsBtn && settingsModal) {
    settingsBtn.addEventListener("click", () => {
      apiKeyInput.value = localApiKey;
      settingsModal.classList.add("active");
    });
  }
  if (closeModalBtn && settingsModal) {
    closeModalBtn.addEventListener("click", () => {
      settingsModal.classList.remove("active");
    });
  }
  if (saveSettingsBtn) {
    saveSettingsBtn.addEventListener("click", handleSaveSettings);
  }

  const newConversationBtn = document.getElementById("newConversationBtn");
  if (newConversationBtn) {
    newConversationBtn.addEventListener("click", startNewConversation);
  }

  // Health tools drawer
  const toolsToggleBtn = document.getElementById("toolsToggleBtn");
  const toolsPanel = document.getElementById("toolsPanel");
  const closeToolsBtn = document.getElementById("closeToolsBtn");
  const toolsBackdrop = document.getElementById("toolsBackdrop");

  const setToolsDrawer = (open) => {
    if (!toolsPanel) return;
    toolsPanel.classList.toggle("open", open);
    if (toolsBackdrop) toolsBackdrop.classList.toggle("visible", open);
    if (toolsToggleBtn) toolsToggleBtn.classList.toggle("active", open);
  };

  if (toolsToggleBtn) {
    toolsToggleBtn.addEventListener("click", () =>
      setToolsDrawer(!toolsPanel.classList.contains("open"))
    );
  }
  if (closeToolsBtn) {
    closeToolsBtn.addEventListener("click", () => setToolsDrawer(false));
  }
  if (toolsBackdrop) {
    toolsBackdrop.addEventListener("click", () => setToolsDrawer(false));
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setToolsDrawer(false);
  });
  const exportPdfBtn = document.getElementById("exportPdfBtn");
  if (exportPdfBtn) {
    exportPdfBtn.addEventListener("click", exportConversation);
  }

  const editProfileBtn = document.getElementById("editProfileBtn");
  const profileModal = document.getElementById("profileModal");
  const closeProfileModalBtn = document.getElementById("closeProfileModalBtn");
  const cancelProfileBtn = document.getElementById("cancelProfileBtn");
  const profileForm = document.getElementById("profileForm");

  if (editProfileBtn && profileModal) {
    editProfileBtn.addEventListener("click", () => {
      if (chatState.profile) {
        document.getElementById("profileName").value = chatState.profile.name || "";
        document.getElementById("profileAge").value =
          chatState.profile.age != null ? chatState.profile.age : "";
        document.getElementById("profileHistory").value =
          chatState.profile.medical_history || "";
      }
      profileModal.classList.add("active");
    });
  }
  if (closeProfileModalBtn && profileModal) {
    closeProfileModalBtn.addEventListener("click", () => {
      profileModal.classList.remove("active");
    });
  }
  if (cancelProfileBtn && profileModal) {
    cancelProfileBtn.addEventListener("click", () => {
      profileModal.classList.remove("active");
    });
  }
  if (profileForm) {
    profileForm.addEventListener("submit", updateProfile);
  }
}

async function handleSendMessage() {
  const chatInput = document.getElementById("chatInput");
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = "";
  chatInput.style.height = "auto";
  chatInput.focus();
  await sendMessage(text);
}

// ================= Tool Handlers =================

async function handleBPCheck(e) {
  e.preventDefault();
  const sys = parseInt(document.getElementById("bpSys").value);
  const dia = parseInt(document.getElementById("bpDia").value);
  const pulseVal = document.getElementById("bpPulse").value;
  const pulse = pulseVal ? parseInt(pulseVal) : null;
  const resultBox = document.getElementById("bpResultBox");

  try {
    const res = await fetch("/api/tools/bp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ systolic: sys, diastolic: dia, pulse: pulse }),
    });
    const data = await res.json();
    resultBox.className = `tool-result-box active`;
    resultBox.style.borderLeftColor = getSeverityColor(data.severity);
    resultBox.innerHTML = `
      <div class="tool-result-title">
        <span>${data.category}</span>
        <span style="font-size: 0.72rem; color: ${getSeverityColor(data.severity)};">[${data.severity}]</span>
      </div>
      <div class="tool-result-desc">
        <p>${data.interpretation}</p>
        <ul style="margin-top: 6px; padding-left: 1rem;">
          ${data.recommendations.map((r) => `<li>${r}</li>`).join("")}
        </ul>
      </div>
    `;
  } catch (err) {
    resultBox.className = "tool-result-box active";
    resultBox.innerHTML = `<span style="color: #ef4444;">Error checking blood pressure.</span>`;
  }
}

async function handleGlucoseCheck(e) {
  e.preventDefault();
  const val = parseFloat(document.getElementById("glucoseVal").value);
  const unit = document.getElementById("glucoseUnit").value;
  const timing = document.getElementById("glucoseTiming").value;
  const resultBox = document.getElementById("glucoseResultBox");

  try {
    const res = await fetch("/api/tools/glucose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value: val, unit: unit, timing: timing }),
    });
    const data = await res.json();
    resultBox.className = `tool-result-box active`;
    resultBox.style.borderLeftColor = getSeverityColor(data.severity);
    resultBox.innerHTML = `
      <div class="tool-result-title">
        <span>${data.category}</span>
        <span style="font-size: 0.72rem; color: ${getSeverityColor(data.severity)};">[${data.severity}]</span>
      </div>
      <div class="tool-result-desc">
        <p>${data.interpretation}</p>
        <ul style="margin-top: 6px; padding-left: 1rem;">
          ${data.recommendations.map((r) => `<li>${r}</li>`).join("")}
        </ul>
      </div>
    `;
  } catch (err) {
    resultBox.className = "tool-result-box active";
    resultBox.innerHTML = `<span style="color: #ef4444;">Error checking blood glucose.</span>`;
  }
}

async function handleHbA1cCheck(e) {
  e.preventDefault();
  const val = parseFloat(document.getElementById("hba1cVal").value);
  const resultBox = document.getElementById("hba1cResultBox");

  try {
    const res = await fetch("/api/tools/hba1c", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hba1c: val }),
    });
    const data = await res.json();
    resultBox.className = `tool-result-box active`;
    resultBox.innerHTML = `
      <div class="tool-result-title">
        <span>HbA1c ${data.hba1c}%</span>
        <span style="color: var(--accent-cyan);">≈ ${data.eag_mg_dl} mg/dL (${data.eag_mmol_l} mmol/L)</span>
      </div>
      <div class="tool-result-desc">
        <p>${data.interpretation}</p>
        <small style="color: var(--text-muted);">${data.guideline_reference}</small>
      </div>
    `;
  } catch (err) {
    resultBox.className = "tool-result-box active";
    resultBox.innerHTML = `<span style="color: #ef4444;">Error converting HbA1c.</span>`;
  }
}

function getSeverityColor(sev) {
  switch (sev) {
    case "CRITICAL":
      return "#dc2626";
    case "HIGH":
      return "#ea580c";
    case "WARNING":
    case "MODERATE":
      return "#d97706";
    case "MILD":
      return "#ca8a04";
    case "OPTIMAL":
      return "#059669";
    default:
      return "#0e7490";
  }
}

async function fetchKBStats() {
  try {
    const res = await fetch("/api/kb/stats");
    const data = await res.json();
    const pill = document.getElementById("kbStatusPill");
    if (pill && data.is_indexed) {
      pill.innerHTML = `<span class="status-dot"></span> <span>${data.total_chunks} Guidelines Indexed</span>`;
    }
  } catch (e) {
    console.log("Could not load KB stats", e);
  }
}

async function handleSaveSettings() {
  const apiKeyInput = document.getElementById("apiKeyInput");
  const modelSelect = document.getElementById("modelSelect");
  const key = apiKeyInput.value.trim();
  const model = modelSelect ? modelSelect.value : "gemini-3.6-flash";

  if (!key) {
    alert("Please enter a valid Google Gemini API Key.");
    return;
  }

  try {
    const res = await fetch("/api/settings/key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key, model_name: model }),
    });
    const data = await res.json();

    if (res.ok) {
      localApiKey = key;
      localStorage.setItem("aurahealth_gemini_api_key", key);
      updateApiStatusBadge(true);
      document.getElementById("settingsModal").classList.remove("active");
      alert("✅ Google Gemini API Key saved and initialized!");
    } else {
      alert(`⚠️ Error: ${data.detail || data.message}`);
    }
  } catch (e) {
    alert(`Failed to save settings: ${e.message}`);
  }
}

function updateApiStatusBadge(connected) {
  const pill = document.getElementById("apiStatusPill");
  if (pill) {
    pill.innerHTML = connected
      ? `<span class="status-dot"></span> <span>Gemini Connected</span>`
      : `<span class="status-dot warning"></span> <span>Gemini API Key Needed</span>`;
  }
}
