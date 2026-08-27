"""
FastAPI Web Application for Diabetes & Hypertension RAG Health Assistant.
Includes additive patient chatbot endpoints for conversations, profiles, and PDF export.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ..agent.assistant import HealthAgent, get_health_agent
from ..chatbot import (
    Allergy,
    ConversationStore,
    ExportService,
    MedicalCondition,
    Medication,
    Message,
    MessageRole,
    ProfileManager,
    SessionManager,
    SessionType,
)
from ..chatbot.models import utc_now
from ..config import (
    ANONYMOUS_MODE_ENABLED,
    AUTHENTICATED_MODE_ENABLED,
    BASE_DIR,
    CONVERSATION_LIST_LIMIT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    GEMINI_API_KEY,
    PDF_INCLUDE_PROFILE,
    resolve_llm_model,
    validate_chatbot_config,
)
from ..ingestion.indexer import build_indices, get_indexed_stats
from ..tools.bp_classifier import classify_blood_pressure
from ..tools.emergency_triage import check_emergency_symptoms
from ..tools.glucose_analyzer import analyze_blood_glucose, convert_hba1c_to_eag

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AuraHealth AI - Diabetes & Hypertension Assistant",
    description="Agentic RAG clinical health assistant for Diabetes and Hypertension management",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = Path(__file__).resolve().parent
STATIC_DIR = UI_DIR / "static"
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Chatbot services (initialized on startup)
conversation_store: Optional[ConversationStore] = None
profile_manager: Optional[ProfileManager] = None
session_manager: Optional[SessionManager] = None
export_service: Optional[ExportService] = None


@app.on_event("startup")
async def startup_chatbot_services() -> None:
    global conversation_store, profile_manager, session_manager, export_service
    settings = validate_chatbot_config()
    logger.info("Chatbot config: %s", settings)

    conversation_store = ConversationStore()
    profile_manager = ProfileManager()
    session_manager = SessionManager(
        auth_enabled=AUTHENTICATED_MODE_ENABLED,
        anonymous_enabled=ANONYMOUS_MODE_ENABLED,
        conversation_store=conversation_store,
        profile_manager=profile_manager,
    )
    export_service = ExportService(conversation_store, profile_manager)


def _require_services() -> None:
    if not all([conversation_store, profile_manager, session_manager, export_service]):
        raise HTTPException(status_code=503, detail="Chatbot services not initialized")


def _client_meta(request: Request) -> tuple[Optional[str], Optional[str]]:
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    return ip, ua


def _resolve_session(
    request: Request,
    session_id: Optional[str] = None,
    patient_id: Optional[str] = None,
):
    _require_services()
    assert session_manager is not None
    ip, ua = _client_meta(request)
    return session_manager.get_or_create_session(
        session_id=session_id,
        patient_id=patient_id,
        ip_address=ip,
        user_agent=ua,
    )


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Legacy request models (unchanged)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message/query")
    history: Optional[List[ChatMessage]] = Field(
        default=[], description="Chat history"
    )
    api_key: Optional[str] = Field(
        None, description="Optional Gemini API key from UI session"
    )


class BPCheckRequest(BaseModel):
    systolic: int
    diastolic: int
    pulse: Optional[int] = None


class GlucoseCheckRequest(BaseModel):
    value: float
    unit: str = "mg/dL"
    timing: str = "random"


class HbA1cRequest(BaseModel):
    hba1c: float


class KeyUpdateRequest(BaseModel):
    api_key: str
    model_name: Optional[str] = DEFAULT_LLM_MODEL


# ---------------------------------------------------------------------------
# Chatbot request models
# ---------------------------------------------------------------------------


class ChatSendRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    api_key: Optional[str] = None


class NewSessionRequest(BaseModel):
    session_id: Optional[str] = None
    title: Optional[str] = None


class ProfileUpdateRequest(BaseModel):
    session_id: str
    updates: Dict[str, Any]


class AddDiagnosisRequest(BaseModel):
    session_id: str
    diagnosis: MedicalCondition


class AddMedicationRequest(BaseModel):
    session_id: str
    medication: Medication


class AddAllergyRequest(BaseModel):
    session_id: str
    allergy: Allergy


class MigrateSessionRequest(BaseModel):
    anonymous_session_id: str
    patient_id: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Legacy endpoints (preserved)
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the main patient dashboard UI."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """
    Main chat endpoint invoking the LangChain Agent with tool calling and hybrid RAG.
    """
    try:
        agent = get_health_agent(api_key=req.api_key)
        history_dicts = (
            [{"role": m.role, "content": m.content} for m in req.history]
            if req.history
            else []
        )
        result = agent.chat(user_message=req.message, chat_history=history_dicts)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "message": "An error occurred while processing your request.",
            },
        )


@app.post("/api/tools/bp")
async def api_bp_tool(req: BPCheckRequest):
    res = classify_blood_pressure(
        systolic=req.systolic, diastolic=req.diastolic, pulse=req.pulse
    )
    return JSONResponse(res)


@app.post("/api/tools/glucose")
async def api_glucose_tool(req: GlucoseCheckRequest):
    res = analyze_blood_glucose(value=req.value, unit=req.unit, timing=req.timing)
    return JSONResponse(res)


@app.post("/api/tools/hba1c")
async def api_hba1c_tool(req: HbA1cRequest):
    res = convert_hba1c_to_eag(hba1c=req.hba1c)
    return JSONResponse(res)


@app.post("/api/tools/emergency")
async def api_emergency_check(req: Dict[str, str]):
    text = req.get("text", "")
    res = check_emergency_symptoms(text=text)
    return JSONResponse(res.model_dump())


@app.get("/api/kb/stats")
async def api_kb_stats():
    stats = get_indexed_stats()
    return JSONResponse(stats)


@app.post("/api/kb/reindex")
async def api_kb_reindex(background_tasks: BackgroundTasks):
    background_tasks.add_task(build_indices)
    return JSONResponse(
        {
            "status": "indexing_started",
            "message": "Re-indexing started in the background.",
        }
    )


@app.post("/api/settings/key")
async def api_update_key(req: KeyUpdateRequest):
    if not req.api_key or len(req.api_key.strip()) < 10:
        raise HTTPException(status_code=400, detail="Invalid Gemini API key provided.")

    os.environ["GEMINI_API_KEY"] = req.api_key.strip()
    model = resolve_llm_model(req.model_name or DEFAULT_LLM_MODEL)
    get_health_agent(api_key=req.api_key.strip(), model_name=model)
    return JSONResponse(
        {
            "status": "success",
            "message": "API key successfully updated and agent initialized!",
            "model_name": model,
        }
    )


# ---------------------------------------------------------------------------
# Conversation endpoints
# ---------------------------------------------------------------------------


@app.post("/api/chat/send")
async def api_chat_send(req: ChatSendRequest, request: Request):
    """Send a message with conversation persistence and profile context."""
    try:
        _require_services()
        assert conversation_store and profile_manager and session_manager

        if not req.message or not req.message.strip():
            raise HTTPException(status_code=400, detail="message is required")

        session = _resolve_session(request, session_id=req.session_id)
        patient_id = session.patient_id
        is_anonymous = session.session_type == SessionType.ANONYMOUS

        if req.conversation_id:
            conversation = conversation_store.load_conversation(
                req.conversation_id, patient_id
            )
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            conversation = conversation_store.create_conversation(
                patient_id=patient_id, is_anonymous=is_anonymous
            )

        conv_id = conversation.metadata.conversation_id
        profile = profile_manager.load_profile(patient_id)
        window = conversation_store.get_conversation_window(conv_id, patient_id)

        agent = get_health_agent(api_key=req.api_key)
        result = agent.chat(
            user_message=req.message,
            patient_profile=profile,
            conversation_window=window,
        )

        user_msg = Message(
            message_id=_new_message_id(),
            role=MessageRole.USER,
            content=req.message,
            timestamp=utc_now(),
        )
        conversation_store.append_message(conv_id, patient_id, user_msg)

        assistant_msg = Message(
            message_id=_new_message_id(),
            role=MessageRole.ASSISTANT,
            content=result["response"],
            timestamp=utc_now(),
            metadata={
                "tools_used": result.get("tools_used", []),
                "emergency": result.get("emergency"),
            },
        )
        conversation_store.append_message(conv_id, patient_id, assistant_msg)

        return JSONResponse(
            {
                "response": result["response"],
                "message_id": assistant_msg.message_id,
                "conversation_id": conv_id,
                "session_id": session.session_id,
                "timestamp": assistant_msg.timestamp.isoformat(),
                "tools_used": result.get("tools_used", []),
                "emergency": result.get("emergency"),
                "status": result.get("status", "success"),
                "guideline_sources": result.get("guideline_sources"),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_chat_send failed")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "message": "An error occurred while processing your request.",
                "status": "error",
            },
        )


@app.get("/api/chat/history/{conversation_id}")
async def api_chat_history(
    conversation_id: str,
    request: Request,
    session_id: str = Query(...),
):
    _require_services()
    assert conversation_store is not None
    session = _resolve_session(request, session_id=session_id)
    conversation = conversation_store.load_conversation(
        conversation_id, session.patient_id
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return JSONResponse(conversation.model_dump(mode="json"))


@app.get("/api/chat/sessions")
async def api_chat_sessions(
    request: Request,
    session_id: str = Query(...),
    limit: int = Query(CONVERSATION_LIST_LIMIT, ge=1, le=200),
):
    _require_services()
    assert conversation_store is not None
    session = _resolve_session(request, session_id=session_id)
    conversations = conversation_store.list_conversations(session.patient_id, limit)
    return JSONResponse(
        {
            "session_id": session.session_id,
            "sessions": [c.model_dump(mode="json") for c in conversations],
        }
    )


@app.post("/api/chat/sessions/new")
async def api_chat_new_session(req: NewSessionRequest, request: Request):
    _require_services()
    assert conversation_store is not None
    session = _resolve_session(request, session_id=req.session_id)
    conversation = conversation_store.create_conversation(
        patient_id=session.patient_id,
        is_anonymous=(session.session_type == SessionType.ANONYMOUS),
        title=req.title,
    )
    return JSONResponse(
        {
            "conversation_id": conversation.metadata.conversation_id,
            "session_id": session.session_id,
            "created_at": conversation.metadata.created_at.isoformat(),
            "title": conversation.metadata.title,
        }
    )


@app.post("/api/session/init")
async def api_session_init(request: Request, session_id: Optional[str] = None):
    """Ensure a chatbot session exists (used by UI on load)."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    sid = session_id or body.get("session_id")
    session = _resolve_session(request, session_id=sid)
    return JSONResponse(
        {
            "session_id": session.session_id,
            "patient_id": session.patient_id,
            "session_type": session.session_type.value,
            "anonymous_mode": ANONYMOUS_MODE_ENABLED,
            "authenticated_mode": AUTHENTICATED_MODE_ENABLED,
        }
    )


# ---------------------------------------------------------------------------
# Profile endpoints
# ---------------------------------------------------------------------------


@app.get("/api/profile")
async def api_get_profile(request: Request, session_id: str = Query(...)):
    _require_services()
    assert profile_manager is not None
    session = _resolve_session(request, session_id=session_id)
    profile = profile_manager.load_profile(session.patient_id)
    if not profile:
        profile = profile_manager.create_profile(
            patient_id=session.patient_id,
            is_anonymous=(session.session_type == SessionType.ANONYMOUS),
        )
    return JSONResponse(profile.model_dump(mode="json"))


@app.post("/api/profile/update")
async def api_update_profile(req: ProfileUpdateRequest, request: Request):
    _require_services()
    assert profile_manager is not None
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    session = _resolve_session(request, session_id=req.session_id)
    try:
        profile = profile_manager.update_profile(session.patient_id, req.updates or {})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "success", "profile": profile.model_dump(mode="json")})


@app.post("/api/profile/diagnosis/add")
async def api_add_diagnosis(req: AddDiagnosisRequest, request: Request):
    _require_services()
    assert profile_manager is not None
    session = _resolve_session(request, session_id=req.session_id)
    try:
        profile = profile_manager.add_diagnosis(session.patient_id, req.diagnosis)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "success", "profile": profile.model_dump(mode="json")})


@app.post("/api/profile/medication/add")
async def api_add_medication(req: AddMedicationRequest, request: Request):
    _require_services()
    assert profile_manager is not None
    session = _resolve_session(request, session_id=req.session_id)
    try:
        profile = profile_manager.add_medication(session.patient_id, req.medication)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "success", "profile": profile.model_dump(mode="json")})


@app.post("/api/profile/allergy/add")
async def api_add_allergy(req: AddAllergyRequest, request: Request):
    _require_services()
    assert profile_manager is not None
    session = _resolve_session(request, session_id=req.session_id)
    try:
        profile = profile_manager.add_allergy(session.patient_id, req.allergy)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "success", "profile": profile.model_dump(mode="json")})


# ---------------------------------------------------------------------------
# Export & migration
# ---------------------------------------------------------------------------


@app.get("/api/export/conversation/{conversation_id}")
async def api_export_conversation(
    conversation_id: str,
    request: Request,
    session_id: str = Query(...),
    include_profile: bool = Query(PDF_INCLUDE_PROFILE),
):
    _require_services()
    assert export_service is not None
    session = _resolve_session(request, session_id=session_id)
    try:
        pdf_buffer = export_service.generate_pdf(
            conversation_id=conversation_id,
            patient_id=session.patient_id,
            include_profile=include_profile,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF export failed")
        raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}") from exc

    return Response(
        content=pdf_buffer.getvalue(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="conversation_{conversation_id}.pdf"'
        },
    )


@app.post("/api/session/migrate")
async def api_migrate_session(req: MigrateSessionRequest, request: Request):
    _require_services()
    assert session_manager is not None
    if not AUTHENTICATED_MODE_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="Authenticated mode is disabled; cannot migrate session.",
        )
    try:
        # Ensure anonymous session exists
        if not session_manager.validate_session(req.anonymous_session_id):
            _resolve_session(request, session_id=req.anonymous_session_id)
        session = session_manager.migrate_anonymous_to_authenticated(
            session_id=req.anonymous_session_id,
            patient_id=req.patient_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(
        {
            "status": "success",
            "message": "Session migrated successfully",
            "session_id": session.session_id,
            "patient_id": session.patient_id,
            "session_type": session.session_type.value,
        }
    )
