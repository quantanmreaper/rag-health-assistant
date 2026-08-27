"""
Core RAG Health Assistant Agent orchestrator using LangChain, LangGraph, and Google Gemini.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from ..chatbot.models import Message, MessageRole, PatientProfile
from ..config import DEFAULT_LLM_MODEL, GEMINI_API_KEY, TOP_K_RETRIEVAL, resolve_llm_model
from ..retrieval.hybrid_retriever import get_hybrid_retriever
from ..tools.bp_classifier import classify_blood_pressure
from ..tools.emergency_triage import check_emergency_symptoms
from ..tools.glucose_analyzer import analyze_blood_glucose, convert_hba1c_to_eag
from .prompts import SYSTEM_PROMPT


def _format_guideline_results(docs, patient_profile: Optional[PatientProfile] = None) -> str:
    """Format retrieved guideline passages, optionally noting patient relevance."""
    if not docs:
        return "No specific guideline passages found for this query in the local knowledge base."

    relevance_note = ""
    if patient_profile and patient_profile.diagnoses:
        names = ", ".join(d.condition_name for d in patient_profile.diagnoses)
        relevance_note = f"\n(Patient diagnoses considered for prioritization: {names})\n"

    results = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("title", doc.metadata.get("source", "Medical Guideline"))
        org = doc.metadata.get("organization", "Guideline Committee")
        page = doc.metadata.get("page", "N/A")
        score = doc.metadata.get("retrieval_score", "")
        results.append(
            f"[Source {idx}]: {source} ({org}, Page {page}) [Score: {score}]\n"
            f"Content:\n{doc.page_content}\n"
        )
    return relevance_note + "\n---\n".join(results)


def _condition_from_profile(patient_profile: Optional[PatientProfile]) -> Optional[str]:
    if not patient_profile or not patient_profile.diagnoses:
        return None
    diagnoses = [d.condition_name.lower() for d in patient_profile.diagnoses]
    has_diabetes = any("diabetes" in d for d in diagnoses)
    has_htn = any("hypertension" in d or "blood pressure" in d for d in diagnoses)
    if has_diabetes and has_htn:
        return "both"
    if has_diabetes:
        return "diabetes"
    if has_htn:
        return "hypertension"
    return None


def create_agent_tools(
    api_key: Optional[str] = None,
    patient_profile: Optional[PatientProfile] = None,
    profile_holder: Optional[Dict[str, Any]] = None,
):
    """
    Creates and binds tools for the LangChain agent.

    `profile_holder` is a mutable dict `{"profile": PatientProfile|None}` so the
    running agent can pick up the latest patient profile per chat() call without
    rebuilding the graph.
    """
    retriever = get_hybrid_retriever(api_key=api_key)
    holder = profile_holder if profile_holder is not None else {"profile": patient_profile}

    @tool
    def retrieve_medical_guidelines(query: str, condition: Optional[str] = None) -> str:
        """
        Retrieves authoritative medical guidelines and clinical evidence for Diabetes and Hypertension.
        Use this tool to ground answers in WHO, ADA, ESH, CDC, and AHA clinical recommendations.

        Args:
            query: The clinical or lifestyle query
            condition: Optional filter ('diabetes', 'hypertension', or 'both')
        """
        active_profile = holder.get("profile")
        if not condition:
            condition = _condition_from_profile(active_profile)

        docs = retriever.get_relevant_documents(
            query=query, k=TOP_K_RETRIEVAL, condition_filter=condition
        )
        return _format_guideline_results(docs, active_profile)

    @tool
    def evaluate_blood_pressure(
        systolic: int, diastolic: int, pulse: Optional[int] = None
    ) -> str:
        """
        Classifies blood pressure readings according to AHA/ACC and ESH clinical practice guidelines.
        Use when the user mentions systolic and diastolic values (e.g. 135/85 mmHg).
        """
        res = classify_blood_pressure(systolic=systolic, diastolic=diastolic, pulse=pulse)
        return json.dumps(res, indent=2)

    @tool
    def evaluate_blood_glucose(
        value: float, unit: str = "mg/dL", timing: str = "random"
    ) -> str:
        """
        Analyzes a blood glucose reading against ADA clinical glycemic targets.
        """
        res = analyze_blood_glucose(value=value, unit=unit, timing=timing)
        return json.dumps(res, indent=2)

    @tool
    def convert_hba1c_to_average_glucose(hba1c: float) -> str:
        """
        Converts an HbA1c percentage to estimated Average Glucose (eAG in mg/dL and mmol/L).
        """
        res = convert_hba1c_to_eag(hba1c=hba1c)
        return json.dumps(res, indent=2)

    @tool
    def triage_emergency_symptoms(symptoms: str) -> str:
        """
        Checks patient symptoms for acute emergency red-flags.
        """
        res = check_emergency_symptoms(text=symptoms)
        return res.model_dump_json(indent=2)

    return [
        retrieve_medical_guidelines,
        evaluate_blood_pressure,
        evaluate_blood_glucose,
        convert_hba1c_to_average_glucose,
        triage_emergency_symptoms,
    ]


class HealthAgent:
    """
    Agentic Health Assistant wrapping LangGraph ReAct agent with Google Gemini.
    """

    def __init__(
        self, api_key: Optional[str] = None, model_name: str = DEFAULT_LLM_MODEL
    ):
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY
        )
        self.model_name = resolve_llm_model(model_name)
        self._profile_holder: Dict[str, Any] = {"profile": None}
        self.tools = create_agent_tools(
            api_key=self.api_key, profile_holder=self._profile_holder
        )
        self.agent = None

        if self.api_key:
            self._init_agent()

    def _init_agent(self):
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.2,
                max_output_tokens=2048,
            )
            self.agent = create_react_agent(
                model=llm, tools=self.tools, prompt=SYSTEM_PROMPT
            )
        except Exception as e:
            print(f"Error initializing LangGraph ReAct Agent: {e}")
            self.agent = None

    def _build_system_context(
        self,
        profile: Optional[PatientProfile],
        window: Optional[List[Message]] = None,
    ) -> str:
        """Build enhanced system prompt with patient-specific clinical context."""
        base_prompt = SYSTEM_PROMPT
        if profile:
            diagnoses = (
                ", ".join(d.condition_name for d in profile.diagnoses)
                if profile.diagnoses
                else "None recorded"
            )
            medications = (
                ", ".join(f"{m.name} {m.dosage}" for m in profile.medications)
                if profile.medications
                else "None recorded"
            )
            allergies = (
                ", ".join(a.allergen for a in profile.allergies)
                if profile.allergies
                else "None recorded"
            )
            base_prompt += f"""

PATIENT CONTEXT:
- Diagnoses: {diagnoses}
- Current Medications: {medications}
- Known Allergies: {allergies}

When providing advice, consider these conditions and medications. Check for drug interactions and contraindications with known allergies. If the patient has both diabetes and hypertension, emphasize cardio-renal protective guidance.
"""
        return base_prompt

    def chat(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        patient_profile: Optional[PatientProfile] = None,
        conversation_window: Optional[List[Message]] = None,
    ) -> Dict[str, Any]:
        """
        Processes a user message and returns the response, tool calls, and citations.

        Optional patient_profile and conversation_window enable personalized,
        multi-turn chatbot responses. Existing chat_history remains supported
        for backward compatibility with the legacy /api/chat endpoint.
        """
        # Always check emergency guardrails first — never bypassed by conversation features
        emergency_assessment = check_emergency_symptoms(user_message)

        # Expose profile to tools for diagnosis-based retrieval prioritization
        self._profile_holder["profile"] = patient_profile

        if not self.api_key or not self.agent:
            return self._fallback_response(user_message, emergency_assessment)

        messages: List[Any] = []

        # Inject patient context as a system message when profile is present
        if patient_profile is not None:
            messages.append(
                SystemMessage(
                    content=self._build_system_context(patient_profile, conversation_window)
                )
            )

        # Prefer structured conversation window; else legacy dict history
        if conversation_window:
            for msg in conversation_window:
                if msg.role == MessageRole.USER:
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == MessageRole.ASSISTANT:
                    messages.append(AIMessage(content=msg.content))
        elif chat_history:
            for item in chat_history:
                role = item.get("role", "user")
                content = item.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role in ["assistant", "ai"]:
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=user_message))

        try:
            result = self.agent.invoke({"messages": messages})
            output_messages = result.get("messages", [])

            final_content = ""
            tools_used = []

            for msg in output_messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.append(
                            {"name": tc.get("name"), "args": tc.get("args")}
                        )
                if isinstance(msg, AIMessage) and msg.content:
                    final_content = msg.content

            if isinstance(final_content, list):
                final_content = "\n".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in final_content
                )

            return {
                "response": final_content
                or "I have processed your request. Please consult with your physician for personalized medical advice.",
                "tools_used": tools_used,
                "emergency": emergency_assessment.model_dump(),
                "status": "success",
            }

        except Exception as e:
            print(f"Agent execution error: {e}")
            return self._fallback_response(
                user_message, emergency_assessment, error_msg=str(e)
            )

    def _fallback_response(
        self,
        query: str,
        emergency: Any,
        error_msg: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fallback response using direct tool & retriever lookup when LLM is unconfigured or unavailable.
        """
        doc_excerpts = []
        try:
            retriever = get_hybrid_retriever(api_key=self.api_key)
            condition = _condition_from_profile(self._profile_holder.get("profile"))
            docs = retriever.get_relevant_documents(
                query=query, k=3, condition_filter=condition
            )
            for d in docs:
                doc_excerpts.append(
                    {
                        "title": d.metadata.get("title", d.metadata.get("source")),
                        "page": d.metadata.get("page"),
                        "organization": d.metadata.get("organization"),
                        "snippet": d.page_content[:300] + "...",
                    }
                )
        except Exception as retrieval_error:
            print(f"Fallback retrieval failed: {retrieval_error}")

        warning_note = ""
        if not self.api_key:
            warning_note = (
                "\n\n*(Note: Google Gemini API key is not yet configured. "
                "Please configure your GEMINI_API_KEY in the settings menu or .env file to enable full clinical AI reasoning.)*"
            )
        elif error_msg:
            warning_note = (
                "\n\n*(Note: The clinical AI model could not complete this request. "
                "Showing guideline excerpts when available. "
                f"Details: {error_msg[:240]})*"
            )

        if doc_excerpts:
            resp_text = (
                f"Here are relevant excerpts from the clinical guidelines regarding your question:{warning_note}\n\n"
            )
            for i, de in enumerate(doc_excerpts, 1):
                resp_text += (
                    f"**{i}. {de['title']} ({de['organization']}, Page {de['page']})**\n"
                    f"> {de['snippet']}\n\n"
                )
        else:
            resp_text = (
                "I could not generate a full AI response right now"
                f"{' (model/API error)' if error_msg else ''}. "
                "Please verify your Gemini model setting (use **gemini-3.6-flash**) "
                "and try again. For emergencies, call local emergency services."
                f"{warning_note}"
            )

        resp_text += (
            "\n*Always consult with your primary healthcare provider before making any changes to your medication or diet.*"
        )

        return {
            "response": resp_text,
            "tools_used": [{"name": "retrieve_medical_guidelines", "args": {"query": query}}],
            "emergency": emergency.model_dump()
            if hasattr(emergency, "model_dump")
            else emergency,
            "guideline_sources": doc_excerpts,
            "status": "fallback",
            "error": error_msg,
        }


# Factory singleton helper
_global_agent: Optional[HealthAgent] = None


def get_health_agent(
    api_key: Optional[str] = None, model_name: Optional[str] = None
) -> HealthAgent:
    global _global_agent
    resolved = resolve_llm_model(model_name or DEFAULT_LLM_MODEL)
    if _global_agent is None or api_key or (
        model_name and resolve_llm_model(model_name) != getattr(_global_agent, "model_name", None)
    ):
        _global_agent = HealthAgent(api_key=api_key, model_name=resolved)
    return _global_agent
