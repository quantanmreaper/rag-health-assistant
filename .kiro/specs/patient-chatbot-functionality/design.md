# Design Document: Patient Chatbot Functionality

## Overview

This document details the design for adding comprehensive patient chatbot functionality to the existing AuraHealth AI RAG health assistant. The enhancement transforms the system from a single-interaction query-response interface into a persistent, personalized conversational health assistant while maintaining all existing clinical safety features.

### Design Goals

1. **Conversation Persistence**: Enable multi-session conversations with complete history storage and retrieval
2. **Personalization**: Provide patient-specific responses based on medical profiles and conversation context
3. **Clinical Safety**: Maintain existing emergency triage, guideline grounding, and clinical tool integration
4. **Hybrid Access**: Support both anonymous guest conversations and authenticated patient accounts
5. **Professional Export**: Generate clinical-quality PDF reports for healthcare provider sharing
6. **Lightweight Storage**: Use JSON file-based storage to avoid database dependencies

### Programming Language

The implementation will use **Python 3.12+** to match the existing codebase, leveraging:
- FastAPI for API endpoints
- LangChain/LangGraph for agent integration
- ReportLab or similar for PDF generation
- JSON for data serialization
- Pathlib for file system operations

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Chat UI Layer                            │
│  (Enhanced templates/index.html + static/app.js, style.css)    │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ HTTP/JSON API
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Web Layer                           │
│              (ui/web.py - Extended Endpoints)                   │
│  /api/chat/send | /api/chat/history | /api/chat/sessions       │
│  /api/profile/* | /api/export/*                                │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─────────────┬──────────────┬─────────────────┐
             ▼             ▼              ▼                 ▼
   ┌──────────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐
   │Conversation  │  │  Profile │  │  Session  │  │   Export     │
   │  Manager     │  │  Manager │  │  Manager  │  │   Service    │
   └──────┬───────┘  └────┬─────┘  └─────┬─────┘  └──────┬───────┘
          │               │              │               │
          │               │              │               │
          ▼               ▼              ▼               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │              Storage Layer (JSON Files)                     │
   │  data/conversations/    |    data/profiles/                │
   │  {patient_id}_{conv_id}.json | {patient_id}_profile.json  │
   └─────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────┐
   │         Enhanced LangGraph Agent (agent/assistant.py)       │
   │  - Conversation Window Context                              │
   │  - Patient Profile Context                                  │
   │  - Existing Clinical Tools Integration                      │
   │  - Hybrid Retrieval with Diagnosis-based Prioritization     │
   └─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/rag_health_assistant/
├── chatbot/                    # NEW MODULE
│   ├── __init__.py
│   ├── conversation_store.py   # Conversation persistence
│   ├── profile_manager.py      # Patient profile management
│   ├── session_manager.py      # Session tracking (auth/anon)
│   ├── export_service.py       # PDF generation
│   └── models.py               # Pydantic data models
├── agent/
│   └── assistant.py            # ENHANCED - Context window support
├── ui/
│   ├── web.py                  # ENHANCED - New chatbot endpoints
│   ├── templates/
│   │   └── index.html          # ENHANCED - Chat UI components
│   └── static/
│       ├── app.js              # ENHANCED - Chat functionality
│       └── style.css           # ENHANCED - Chat styling
└── config.py                   # ENHANCED - Chatbot configuration
```

## Component Design

### 1. Conversation Store

**File**: `chatbot/conversation_store.py`

**Responsibility**: Persistent storage and retrieval of conversation messages using JSON files

**Data Model**:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    """Single message in a conversation"""
    message_id: str = Field(..., description="Unique message identifier")
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Optional[dict] = Field(default_factory=dict)  # For tool calls, citations, etc.

class ConversationMetadata(BaseModel):
    """Conversation-level metadata"""
    conversation_id: str
    patient_id: str  # Can be session ID for anonymous
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    is_anonymous: bool = False
    message_count: int = 0

class Conversation(BaseModel):
    """Complete conversation with metadata and messages"""
    metadata: ConversationMetadata
    messages: List[Message] = Field(default_factory=list)
```

**Key Methods**:

```python
class ConversationStore:
    def __init__(self, storage_dir: Path):
        """Initialize with data/conversations/ directory"""
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._locks = {}  # File locks for concurrent writes
    
    def save_conversation(self, conversation: Conversation) -> None:
        """
        Atomically save conversation to JSON file.
        Filename: {patient_id}_{conversation_id}.json
        Uses file locking to prevent concurrent write conflicts.
        """
    
    def load_conversation(self, conversation_id: str, patient_id: str) -> Optional[Conversation]:
        """
        Load conversation from JSON file.
        Handles malformed JSON gracefully with logging.
        """
    
    def append_message(self, conversation_id: str, patient_id: str, message: Message) -> None:
        """
        Append a single message to existing conversation.
        Updates updated_at timestamp and message_count.
        """
    
    def list_conversations(self, patient_id: str, limit: int = 50) -> List[ConversationMetadata]:
        """
        List all conversations for a patient, sorted by updated_at descending.
        Returns only metadata for performance.
        """
    
    def delete_conversation(self, conversation_id: str, patient_id: str) -> bool:
        """Soft delete by adding deleted flag to metadata"""
    
    def get_conversation_window(
        self,
        conversation_id: str,
        patient_id: str,
        window_size: int = 20,
        window_days: int = 7
    ) -> List[Message]:
        """
        Retrieve recent messages for context window.
        Returns last N messages OR messages from last M days, whichever is smaller.
        """
```

**File Format Example**:

```json
{
  "metadata": {
    "conversation_id": "conv_abc123",
    "patient_id": "patient_xyz789",
    "title": "Questions about blood pressure medication",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T11:45:00Z",
    "is_anonymous": false,
    "message_count": 8
  },
  "messages": [
    {
      "message_id": "msg_001",
      "role": "user",
      "content": "I'm taking lisinopril 10mg daily. Is it safe to take ibuprofen?",
      "timestamp": "2024-01-15T10:30:00Z",
      "metadata": {}
    },
    {
      "message_id": "msg_002",
      "role": "assistant",
      "content": "Based on your profile showing hypertension diagnosis and current lisinopril medication, I need to advise caution...",
      "timestamp": "2024-01-15T10:30:15Z",
      "metadata": {
        "tools_used": ["retrieve_medical_guidelines", "triage_emergency_symptoms"],
        "citations": ["AHA Hypertension Guidelines 2023", "FDA Drug Interaction Database"]
      }
    }
  ]
}
```

### 2. Profile Manager

**File**: `chatbot/profile_manager.py`

**Responsibility**: Patient profile creation, storage, retrieval, and updates

**Data Model**:

```python
class MedicalCondition(BaseModel):
    """Single diagnosed condition"""
    condition_name: str  # e.g., "Type 2 Diabetes", "Hypertension"
    icd_code: Optional[str] = None
    diagnosed_date: Optional[str] = None
    status: str = "active"  # active, resolved, managed

class Medication(BaseModel):
    """Current medication"""
    name: str
    dosage: str
    frequency: str
    started_date: Optional[str] = None
    prescriber: Optional[str] = None

class Allergy(BaseModel):
    """Known allergy"""
    allergen: str
    reaction: str
    severity: str  # mild, moderate, severe

class PatientProfile(BaseModel):
    """Complete patient medical profile"""
    patient_id: str
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    age: Optional[int] = None
    diagnoses: List[MedicalCondition] = Field(default_factory=list)
    medications: List[Medication] = Field(default_factory=list)
    allergies: List[Allergy] = Field(default_factory=list)
    medical_history: Optional[str] = None
    last_updated: datetime
    is_anonymous: bool = False
```

**Key Methods**:

```python
class ProfileManager:
    def __init__(self, storage_dir: Path):
        """Initialize with data/profiles/ directory"""
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}  # LRU cache for frequent access
    
    def create_profile(self, patient_id: str, is_anonymous: bool = False) -> PatientProfile:
        """Create new patient profile with defaults"""
    
    def save_profile(self, profile: PatientProfile) -> None:
        """
        Save profile to JSON file.
        Filename: {patient_id}_profile.json
        Updates cache and last_updated timestamp.
        """
    
    def load_profile(self, patient_id: str) -> Optional[PatientProfile]:
        """
        Load profile from JSON file or cache.
        Returns None if profile doesn't exist.
        """
    
    def update_profile(self, patient_id: str, updates: dict) -> PatientProfile:
        """
        Partial update of profile fields.
        Validates required fields are present after update.
        """
    
    def add_diagnosis(self, patient_id: str, condition: MedicalCondition) -> None:
        """Add a diagnosis to patient profile"""
    
    def add_medication(self, patient_id: str, medication: Medication) -> None:
        """Add a medication to patient profile"""
    
    def add_allergy(self, patient_id: str, allergy: Allergy) -> None:
        """Add an allergy to patient profile"""
    
    def get_clinical_context_summary(self, patient_id: str) -> str:
        """
        Generate formatted summary for LLM context.
        Returns string with diagnoses, medications, allergies.
        """
```

**File Format Example**:

```json
{
  "patient_id": "patient_xyz789",
  "name": "Jane Doe",
  "date_of_birth": "1965-03-20",
  "age": 59,
  "diagnoses": [
    {
      "condition_name": "Type 2 Diabetes Mellitus",
      "icd_code": "E11",
      "diagnosed_date": "2018-06-15",
      "status": "active"
    },
    {
      "condition_name": "Essential Hypertension",
      "icd_code": "I10",
      "diagnosed_date": "2016-02-10",
      "status": "active"
    }
  ],
  "medications": [
    {
      "name": "Metformin",
      "dosage": "1000mg",
      "frequency": "twice daily",
      "started_date": "2018-06-15"
    },
    {
      "name": "Lisinopril",
      "dosage": "10mg",
      "frequency": "once daily",
      "started_date": "2016-02-10"
    }
  ],
  "allergies": [
    {
      "allergen": "Penicillin",
      "reaction": "Hives",
      "severity": "moderate"
    }
  ],
  "medical_history": "Family history of cardiovascular disease. Non-smoker. Moderate physical activity.",
  "last_updated": "2024-01-15T10:00:00Z",
  "is_anonymous": false
}
```

### 3. Session Manager

**File**: `chatbot/session_manager.py`

**Responsibility**: Session tracking for both authenticated and anonymous users

**Key Concepts**:
- **Authenticated Mode**: Patient ID comes from authentication system (future OAuth/JWT integration point)
- **Anonymous Mode**: Generate temporary session ID, store in browser session/cookie

**Data Model**:

```python
class SessionType(str, Enum):
    AUTHENTICATED = "authenticated"
    ANONYMOUS = "anonymous"

class Session(BaseModel):
    """User session tracking"""
    session_id: str
    patient_id: str  # For authenticated, real patient ID; for anonymous, session ID
    session_type: SessionType
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
```

**Key Methods**:

```python
class SessionManager:
    def __init__(self, auth_enabled: bool = False):
        """Initialize with authentication mode configuration"""
        self.auth_enabled = auth_enabled
        self._sessions = {}  # In-memory session tracking
    
    def create_anonymous_session(self, request: Request) -> Session:
        """
        Generate temporary session ID for anonymous user.
        Uses UUID + timestamp for uniqueness.
        """
    
    def get_or_create_session(self, request: Request, patient_id: Optional[str] = None) -> Session:
        """
        Get existing session or create new one.
        If patient_id provided, creates authenticated session.
        Otherwise, creates anonymous session.
        """
    
    def migrate_anonymous_to_authenticated(
        self,
        session_id: str,
        patient_id: str
    ) -> None:
        """
        Migrate anonymous session data to authenticated account.
        Updates conversation and profile ownership.
        """
    
    def validate_session(self, session_id: str) -> bool:
        """Check if session is valid and not expired"""
    
    def get_patient_id(self, session_id: str) -> Optional[str]:
        """Get patient ID associated with session"""
```

### 4. Export Service

**File**: `chatbot/export_service.py`

**Responsibility**: Generate professional PDF reports from conversations

**Dependencies**: `reportlab` for PDF generation

**Key Methods**:

```python
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from io import BytesIO

class ExportService:
    def __init__(
        self,
        conversation_store: ConversationStore,
        profile_manager: ProfileManager
    ):
        self.conversation_store = conversation_store
        self.profile_manager = profile_manager
    
    def generate_pdf(
        self,
        conversation_id: str,
        patient_id: str,
        include_profile: bool = True
    ) -> BytesIO:
        """
        Generate PDF report from conversation.
        Returns BytesIO buffer containing PDF bytes.
        """
        # Load conversation and profile
        conversation = self.conversation_store.load_conversation(conversation_id, patient_id)
        profile = self.profile_manager.load_profile(patient_id) if include_profile else None
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        # Build PDF content
        story.extend(self._build_header(conversation, profile))
        story.extend(self._build_profile_section(profile))
        story.extend(self._build_messages_section(conversation))
        story.extend(self._build_footer())
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _build_header(self, conversation: Conversation, profile: Optional[PatientProfile]) -> List:
        """Build PDF header with metadata"""
    
    def _build_profile_section(self, profile: Optional[PatientProfile]) -> List:
        """Build patient profile summary section"""
    
    def _build_messages_section(self, conversation: Conversation) -> List:
        """
        Build conversation messages section.
        Highlights emergency triage warnings.
        Formats with timestamps and sender labels.
        """
    
    def _build_footer(self) -> List:
        """Build disclaimer footer"""
    
    def _sanitize_content(self, text: str) -> str:
        """
        Sanitize text for PDF compatibility.
        Handles special characters, markdown, etc.
        """
```

**PDF Report Structure**:

```
╔══════════════════════════════════════════════════════════════════╗
║                   CONVERSATION HEALTH REPORT                     ║
║                      AuraHealth AI Assistant                     ║
╠══════════════════════════════════════════════════════════════════╣
║ Patient ID: patient_xyz789                                       ║
║ Conversation ID: conv_abc123                                     ║
║ Date Range: January 15, 2024 10:30 AM - 11:45 AM               ║
║ Report Generated: January 15, 2024 2:00 PM                      ║
╠══════════════════════════════════════════════════════════════════╣
║ PATIENT PROFILE SUMMARY                                          ║
║ Name: Jane Doe (Age 59)                                          ║
║ Diagnoses: Type 2 Diabetes Mellitus, Essential Hypertension     ║
║ Medications: Metformin 1000mg BID, Lisinopril 10mg QD          ║
║ Allergies: Penicillin (Moderate - Hives)                        ║
╠══════════════════════════════════════════════════════════════════╣
║ CONVERSATION TRANSCRIPT                                          ║
║                                                                  ║
║ [10:30 AM] Patient:                                             ║
║ I'm taking lisinopril 10mg daily. Is it safe to take ibuprofen?║
║                                                                  ║
║ [10:30 AM] AuraHealth Assistant:                                ║
║ Based on your profile showing hypertension diagnosis and current║
║ lisinopril medication, I need to advise caution...              ║
║ [Guidelines: AHA Hypertension Guidelines 2023]                  ║
║                                                                  ║
║ ⚠ [10:35 AM] CLINICAL ALERT:                                   ║
║ Emergency triage detected potential concern - immediate medical ║
║ attention recommended.                                           ║
║                                                                  ║
║ [... continued messages ...]                                    ║
╠══════════════════════════════════════════════════════════════════╣
║ DISCLAIMER                                                       ║
║ This conversation report is for informational purposes only and ║
║ does not constitute medical advice. Always consult with your    ║
║ healthcare provider before making any changes to your treatment.║
╚══════════════════════════════════════════════════════════════════╝
```

### 5. Enhanced LangGraph Agent Integration

**File**: `agent/assistant.py` (Enhanced)

**Enhancements**:

1. **Conversation Window Context Injection**

```python
class HealthAgent:
    def chat(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        patient_profile: Optional[PatientProfile] = None,  # NEW
        conversation_window: Optional[List[Message]] = None  # NEW
    ) -> Dict[str, Any]:
        """
        Enhanced chat method with profile and window context.
        """
        # Build enhanced system prompt with patient context
        system_context = self._build_system_context(patient_profile, conversation_window)
        
        # Build message sequence with conversation window
        messages = [SystemMessage(content=system_context)]
        
        # Add conversation window for context
        if conversation_window:
            for msg in conversation_window:
                if msg.role == MessageRole.USER:
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == MessageRole.ASSISTANT:
                    messages.append(AIMessage(content=msg.content))
        
        # Add current message
        messages.append(HumanMessage(content=user_message))
        
        # Invoke agent with enhanced context
        result = self.agent.invoke({"messages": messages})
        
        return self._format_response(result)
    
    def _build_system_context(
        self,
        profile: Optional[PatientProfile],
        window: Optional[List[Message]]
    ) -> str:
        """
        Build enhanced system prompt with patient-specific context.
        """
        base_prompt = SYSTEM_PROMPT
        
        if profile:
            profile_context = f"""

PATIENT CONTEXT:
- Diagnoses: {', '.join(d.condition_name for d in profile.diagnoses)}
- Current Medications: {', '.join(f"{m.name} {m.dosage}" for m in profile.medications)}
- Known Allergies: {', '.join(a.allergen for a in profile.allergies)}

When providing advice, consider these conditions and medications. Check for drug interactions and contraindications.
"""
            base_prompt += profile_context
        
        return base_prompt
```

2. **Diagnosis-Based Retrieval Prioritization**

```python
def create_agent_tools(api_key: Optional[str] = None, patient_profile: Optional[PatientProfile] = None):
    """Enhanced tool creation with profile context"""
    
    retriever = get_hybrid_retriever(api_key=api_key)
    
    @tool
    def retrieve_medical_guidelines(query: str, condition: Optional[str] = None) -> str:
        """
        ENHANCED: Retrieves guidelines with diagnosis-based prioritization.
        """
        # Determine condition filter from patient profile if not specified
        if not condition and patient_profile:
            # Prioritize based on patient diagnoses
            diagnoses = [d.condition_name.lower() for d in patient_profile.diagnoses]
            if any('diabetes' in d for d in diagnoses):
                condition = 'diabetes'
            elif any('hypertension' in d or 'blood pressure' in d for d in diagnoses):
                condition = 'hypertension'
        
        # Retrieve with condition filter
        docs = retriever.get_relevant_documents(
            query=query,
            k=TOP_K_RETRIEVAL,
            condition_filter=condition
        )
        
        # Format results with patient-relevant context
        return self._format_guideline_results(docs, patient_profile)
```

### 6. FastAPI Endpoint Extensions

**File**: `ui/web.py` (Enhanced)

**New Endpoints**:

```python
# ============= CONVERSATION ENDPOINTS =============

@app.post("/api/chat/send")
async def api_chat_send(req: ChatSendRequest):
    """
    Enhanced chat endpoint with conversation persistence.
    
    Request:
    {
        "message": "What should my target blood pressure be?",
        "conversation_id": "conv_abc123",  // Optional, creates new if not provided
        "session_id": "sess_xyz789"
    }
    
    Response:
    {
        "response": "Based on your hypertension diagnosis...",
        "message_id": "msg_456",
        "conversation_id": "conv_abc123",
        "timestamp": "2024-01-15T10:30:15Z",
        "tools_used": [...],
        "emergency": {...}
    }
    """
    # Get or create session
    session = session_manager.get_or_create_session(request, req.session_id)
    patient_id = session.patient_id
    
    # Get or create conversation
    if req.conversation_id:
        conversation = conversation_store.load_conversation(req.conversation_id, patient_id)
    else:
        conversation = conversation_store.create_conversation(patient_id, session.session_type == SessionType.ANONYMOUS)
    
    # Load patient profile and conversation window
    profile = profile_manager.load_profile(patient_id)
    window = conversation_store.get_conversation_window(conversation.metadata.conversation_id, patient_id)
    
    # Get agent response with enhanced context
    agent = get_health_agent()
    result = agent.chat(
        user_message=req.message,
        patient_profile=profile,
        conversation_window=window
    )
    
    # Save user message
    user_msg = Message(
        message_id=generate_message_id(),
        role=MessageRole.USER,
        content=req.message,
        timestamp=datetime.utcnow()
    )
    conversation_store.append_message(conversation.metadata.conversation_id, patient_id, user_msg)
    
    # Save assistant response
    assistant_msg = Message(
        message_id=generate_message_id(),
        role=MessageRole.ASSISTANT,
        content=result["response"],
        timestamp=datetime.utcnow(),
        metadata={"tools_used": result["tools_used"], "emergency": result["emergency"]}
    )
    conversation_store.append_message(conversation.metadata.conversation_id, patient_id, assistant_msg)
    
    return JSONResponse({
        "response": result["response"],
        "message_id": assistant_msg.message_id,
        "conversation_id": conversation.metadata.conversation_id,
        "timestamp": assistant_msg.timestamp.isoformat(),
        "tools_used": result["tools_used"],
        "emergency": result["emergency"]
    })


@app.get("/api/chat/history/{conversation_id}")
async def api_chat_history(conversation_id: str, session_id: str = Query(...)):
    """
    Retrieve complete conversation history.
    
    Response:
    {
        "conversation_id": "conv_abc123",
        "title": "Blood pressure medication questions",
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T11:45:00Z",
        "messages": [...]
    }
    """
    session = session_manager.get_or_create_session(request, session_id)
    conversation = conversation_store.load_conversation(conversation_id, session.patient_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return JSONResponse(conversation.model_dump())


@app.get("/api/chat/sessions")
async def api_chat_sessions(session_id: str = Query(...), limit: int = 50):
    """
    List all conversation sessions for patient.
    
    Response:
    {
        "sessions": [
            {
                "conversation_id": "conv_abc123",
                "title": "Blood pressure medication questions",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T11:45:00Z",
                "message_count": 8
            },
            ...
        ]
    }
    """
    session = session_manager.get_or_create_session(request, session_id)
    conversations = conversation_store.list_conversations(session.patient_id, limit)
    
    return JSONResponse({"sessions": [c.model_dump() for c in conversations]})


@app.post("/api/chat/sessions/new")
async def api_chat_new_session(req: NewSessionRequest):
    """
    Create new conversation session.
    
    Response:
    {
        "conversation_id": "conv_xyz123",
        "created_at": "2024-01-15T12:00:00Z"
    }
    """
    session = session_manager.get_or_create_session(request, req.session_id)
    conversation = conversation_store.create_conversation(
        patient_id=session.patient_id,
        is_anonymous=(session.session_type == SessionType.ANONYMOUS),
        title=req.title
    )
    
    return JSONResponse({
        "conversation_id": conversation.metadata.conversation_id,
        "created_at": conversation.metadata.created_at.isoformat()
    })


# ============= PROFILE ENDPOINTS =============

@app.get("/api/profile")
async def api_get_profile(session_id: str = Query(...)):
    """Get patient profile"""
    session = session_manager.get_or_create_session(request, session_id)
    profile = profile_manager.load_profile(session.patient_id)
    
    if not profile:
        # Create empty profile for new users
        profile = profile_manager.create_profile(
            patient_id=session.patient_id,
            is_anonymous=(session.session_type == SessionType.ANONYMOUS)
        )
        profile_manager.save_profile(profile)
    
    return JSONResponse(profile.model_dump())


@app.post("/api/profile/update")
async def api_update_profile(req: ProfileUpdateRequest):
    """Update patient profile"""
    session = session_manager.get_or_create_session(request, req.session_id)
    profile = profile_manager.update_profile(session.patient_id, req.updates)
    
    return JSONResponse({"status": "success", "profile": profile.model_dump()})


@app.post("/api/profile/diagnosis/add")
async def api_add_diagnosis(req: AddDiagnosisRequest):
    """Add diagnosis to profile"""
    session = session_manager.get_or_create_session(request, req.session_id)
    profile_manager.add_diagnosis(session.patient_id, req.diagnosis)
    
    return JSONResponse({"status": "success"})


@app.post("/api/profile/medication/add")
async def api_add_medication(req: AddMedicationRequest):
    """Add medication to profile"""
    session = session_manager.get_or_create_session(request, req.session_id)
    profile_manager.add_medication(session.patient_id, req.medication)
    
    return JSONResponse({"status": "success"})


@app.post("/api/profile/allergy/add")
async def api_add_allergy(req: AddAllergyRequest):
    """Add allergy to profile"""
    session = session_manager.get_or_create_session(request, req.session_id)
    profile_manager.add_allergy(session.patient_id, req.allergy)
    
    return JSONResponse({"status": "success"})


# ============= EXPORT ENDPOINTS =============

@app.get("/api/export/conversation/{conversation_id}")
async def api_export_conversation(
    conversation_id: str,
    session_id: str = Query(...),
    include_profile: bool = True
):
    """
    Export conversation as PDF.
    
    Returns: PDF file for download
    """
    session = session_manager.get_or_create_session(request, session_id)
    
    # Generate PDF
    pdf_buffer = export_service.generate_pdf(
        conversation_id=conversation_id,
        patient_id=session.patient_id,
        include_profile=include_profile
    )
    
    # Return as downloadable file
    return Response(
        content=pdf_buffer.getvalue(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=conversation_{conversation_id}.pdf"
        }
    )


# ============= SESSION MANAGEMENT ENDPOINTS =============

@app.post("/api/session/migrate")
async def api_migrate_session(req: MigrateSessionRequest):
    """
    Migrate anonymous session to authenticated account.
    
    Request:
    {
        "anonymous_session_id": "sess_anon_123",
        "patient_id": "patient_authenticated_456"
    }
    """
    session_manager.migrate_anonymous_to_authenticated(
        session_id=req.anonymous_session_id,
        patient_id=req.patient_id
    )
    
    # Migrate conversations and profile
    # (Implementation in session_manager.migrate_anonymous_to_authenticated)
    
    return JSONResponse({"status": "success", "message": "Session migrated successfully"})
```

### 7. Enhanced Chat UI

**File**: `ui/templates/index.html` (Enhanced)

**New UI Components**:

```html
<!-- Enhanced layout with chat sidebar and conversation area -->
<div class="dashboard-container">
    <!-- Existing header -->
    <header>...</header>
    
    <!-- New: Chat layout -->
    <div class="chat-container">
        <!-- Conversation sidebar -->
        <aside class="conversation-sidebar">
            <div class="sidebar-header">
                <button id="new-conversation-btn" class="btn-primary">
                    <i class="icon-plus"></i> New Conversation
                </button>
            </div>
            
            <!-- Conversation list -->
            <div class="conversation-list" id="conversation-list">
                <!-- Dynamically populated -->
            </div>
            
            <!-- Profile summary -->
            <div class="profile-summary" id="profile-summary">
                <h3>Your Profile</h3>
                <div id="profile-diagnoses"></div>
                <div id="profile-medications"></div>
                <button id="edit-profile-btn">Edit Profile</button>
            </div>
        </aside>
        
        <!-- Main chat area -->
        <main class="chat-main">
            <!-- Chat header -->
            <div class="chat-header">
                <h2 id="conversation-title">New Conversation</h2>
                <div class="chat-actions">
                    <button id="export-pdf-btn" class="btn-secondary">
                        <i class="icon-download"></i> Export PDF
                    </button>
                    <span id="mode-indicator" class="badge">Anonymous Mode</span>
                </div>
            </div>
            
            <!-- Messages area -->
            <div class="messages-container" id="messages-container">
                <!-- Welcome message -->
                <div class="welcome-message">
                    <h3>Welcome to AuraHealth AI</h3>
                    <p>I'm here to help with your diabetes and hypertension questions.</p>
                    <p>Ask me about medications, lifestyle changes, or interpreting your health readings.</p>
                </div>
                
                <!-- Messages dynamically added here -->
            </div>
            
            <!-- Typing indicator -->
            <div class="typing-indicator" id="typing-indicator" style="display: none;">
                <span></span><span></span><span></span>
                AuraHealth is thinking...
            </div>
            
            <!-- Message input -->
            <div class="message-input-container">
                <textarea
                    id="message-input"
                    class="message-input"
                    placeholder="Type your message here... (Shift+Enter for new line)"
                    rows="3"
                ></textarea>
                <button id="send-btn" class="btn-send">
                    <i class="icon-send"></i> Send
                </button>
            </div>
        </main>
    </div>
    
    <!-- Profile edit modal -->
    <div id="profile-modal" class="modal">
        <!-- Profile editing form -->
    </div>
</div>
```

**File**: `ui/static/app.js` (Enhanced)

**Key JavaScript Functions**:

```javascript
// Chat state management
const chatState = {
    currentConversationId: null,
    sessionId: null,
    messages: [],
    profile: null,
    conversations: []
};

// Initialize chat on page load
async function initializeChat() {
    // Get or create session
    chatState.sessionId = getSessionIdFromCookie() || await createNewSession();
    
    // Load conversations list
    await loadConversations();
    
    // Load profile
    await loadProfile();
    
    // Setup event listeners
    setupEventListeners();
}

// Load conversation list
async function loadConversations() {
    const response = await fetch(`/api/chat/sessions?session_id=${chatState.sessionId}`);
    const data = await response.json();
    chatState.conversations = data.sessions;
    renderConversationList();
}

// Load specific conversation
async function loadConversation(conversationId) {
    const response = await fetch(
        `/api/chat/history/${conversationId}?session_id=${chatState.sessionId}`
    );
    const conversation = await response.json();
    
    chatState.currentConversationId = conversationId;
    chatState.messages = conversation.messages;
    
    renderMessages();
    updateConversationTitle(conversation.title);
}

// Send message
async function sendMessage(content) {
    // Add user message to UI immediately
    addMessageToUI({
        role: 'user',
        content: content,
        timestamp: new Date().toISOString()
    });
    
    // Show typing indicator
    showTypingIndicator();
    
    // Send to API
    const response = await fetch('/api/chat/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: content,
            conversation_id: chatState.currentConversationId,
            session_id: chatState.sessionId
        })
    });
    
    const data = await response.json();
    
    // Hide typing indicator
    hideTypingIndicator();
    
    // Add assistant response to UI
    addMessageToUI({
        role: 'assistant',
        content: data.response,
        timestamp: data.timestamp,
        metadata: {
            tools_used: data.tools_used,
            emergency: data.emergency
        }
    });
    
    // Update conversation ID if new conversation
    if (!chatState.currentConversationId) {
        chatState.currentConversationId = data.conversation_id;
        await loadConversations(); // Refresh list
    }
    
    // Check for emergency alerts
    if (data.emergency && data.emergency.is_emergency) {
        showEmergencyAlert(data.emergency);
    }
}

// Render message in UI
function addMessageToUI(message) {
    const messagesContainer = document.getElementById('messages-container');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${message.role}`;
    
    const timeString = formatTimestamp(message.timestamp);
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="message-sender">${message.role === 'user' ? 'You' : 'AuraHealth'}</span>
            <span class="message-time">${timeString}</span>
        </div>
        <div class="message-content">${formatMessageContent(message.content)}</div>
    `;
    
    // Add emergency styling if needed
    if (message.metadata?.emergency?.is_emergency) {
        messageDiv.classList.add('emergency-message');
    }
    
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Export conversation as PDF
async function exportConversation() {
    if (!chatState.currentConversationId) {
        alert('No conversation to export');
        return;
    }
    
    window.location.href = `/api/export/conversation/${chatState.currentConversationId}?session_id=${chatState.sessionId}&include_profile=true`;
}

// Profile management
async function loadProfile() {
    const response = await fetch(`/api/profile?session_id=${chatState.sessionId}`);
    chatState.profile = await response.json();
    renderProfileSummary();
}

async function updateProfile(updates) {
    const response = await fetch('/api/profile/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: chatState.sessionId,
            updates: updates
        })
    });
    
    const data = await response.json();
    chatState.profile = data.profile;
    renderProfileSummary();
}
```

**File**: `ui/static/style.css` (Enhanced)

```css
/* Chat container layout */
.chat-container {
    display: flex;
    height: calc(100vh - 80px); /* Adjust for header */
    max-width: 1400px;
    margin: 0 auto;
}

/* Conversation sidebar */
.conversation-sidebar {
    width: 300px;
    border-right: 1px solid var(--border-color);
    background: var(--sidebar-bg);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.conversation-list {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
}

.conversation-item {
    padding: 1rem;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s;
    margin-bottom: 0.5rem;
}

.conversation-item:hover {
    background: var(--hover-bg);
}

.conversation-item.active {
    background: var(--primary-color);
    color: white;
}

/* Main chat area */
.chat-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--chat-bg);
}

.chat-header {
    padding: 1.5rem;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

/* Message styling */
.message {
    max-width: 70%;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.message-user {
    align-self: flex-end;
    background: var(--user-message-bg);
    color: white;
}

.message-assistant {
    align-self: flex-start;
    background: var(--assistant-message-bg);
    border: 1px solid var(--border-color);
}

.message.emergency-message {
    border: 2px solid var(--error-color);
    background: var(--error-bg);
}

.message-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.875rem;
    margin-bottom: 0.5rem;
    opacity: 0.8;
}

.message-content {
    line-height: 1.6;
}

/* Typing indicator */
.typing-indicator {
    padding: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-secondary);
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--primary-color);
    animation: typing 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-10px); }
}

/* Message input */
.message-input-container {
    padding: 1.5rem;
    border-top: 1px solid var(--border-color);
    display: flex;
    gap: 1rem;
    background: white;
}

.message-input {
    flex: 1;
    padding: 1rem;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    resize: none;
    font-family: inherit;
    font-size: 1rem;
}

.message-input:focus {
    outline: none;
    border-color: var(--primary-color);
}

.btn-send {
    padding: 1rem 2rem;
    background: var(--primary-color);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s;
}

.btn-send:hover {
    background: var(--primary-dark);
}
```

## Data Models Summary

### Core Entities

1. **Message**: Single chat message with role, content, timestamp, metadata
2. **Conversation**: Collection of messages with metadata (ID, patient_id, timestamps, title)
3. **PatientProfile**: Medical information (diagnoses, medications, allergies, demographics)
4. **Session**: User session tracking (authenticated or anonymous)
5. **ConversationWindow**: Sliding window of recent messages for context

### File Storage Structure

```
data/
├── conversations/
│   ├── patient_xyz789_conv_abc123.json
│   ├── patient_xyz789_conv_def456.json
│   └── session_anon_123_conv_ghi789.json  # Anonymous conversations
└── profiles/
    ├── patient_xyz789_profile.json
    └── session_anon_123_profile.json  # Temporary anonymous profiles
```

## Configuration

**File**: `config.py` (Enhanced)

```python
# ============= CHATBOT CONFIGURATION =============

# Conversation storage
CONVERSATIONS_DIR = DATA_DIR / "conversations"
PROFILES_DIR = DATA_DIR / "profiles"

# Ensure chatbot directories exist
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# Conversation window configuration
CONVERSATION_WINDOW_SIZE = int(os.getenv("CONVERSATION_WINDOW_SIZE", "20"))  # Last N messages
CONVERSATION_WINDOW_DAYS = int(os.getenv("CONVERSATION_WINDOW_DAYS", "7"))   # Last M days

# Session configuration
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
ANONYMOUS_MODE_ENABLED = os.getenv("ANONYMOUS_MODE_ENABLED", "true").lower() == "true"
AUTHENTICATED_MODE_ENABLED = os.getenv("AUTHENTICATED_MODE_ENABLED", "false").lower() == "true"

# Export configuration
PDF_PAGE_SIZE = "letter"
PDF_INCLUDE_PROFILE = True
PDF_INCLUDE_METADATA = True

# Performance configuration
PROFILE_CACHE_SIZE = int(os.getenv("PROFILE_CACHE_SIZE", "100"))  # LRU cache size
CONVERSATION_LIST_LIMIT = int(os.getenv("CONVERSATION_LIST_LIMIT", "50"))

# Data retention (future feature)
CONVERSATION_RETENTION_DAYS = int(os.getenv("CONVERSATION_RETENTION_DAYS", "365"))
```

## Error Handling

### Error Scenarios and Handling

1. **File System Errors**
   - Graceful handling with user-friendly error messages
   - Logging for debugging
   - Retry logic for transient failures

2. **JSON Parsing Errors**
   - Validation before writing
   - Graceful handling of malformed files on read
   - Backup/recovery mechanisms

3. **Concurrent Write Conflicts**
   - File locking to serialize writes
   - Atomic write operations (write to temp, then rename)

4. **Missing Profile/Conversation**
   - Return empty profile for new users
   - Create new conversation on first message
   - Clear error messages for invalid IDs

5. **LLM API Errors**
   - Fallback to guideline retrieval only
   - Error messages to user
   - Retry logic with exponential backoff

## Security Considerations

1. **Patient Data Privacy**
   - No passwords stored (authentication delegated to future OAuth)
   - File-based storage with appropriate permissions
   - No PII in logs

2. **Session Security**
   - Secure session ID generation (UUID + timestamp + random)
   - Session timeout enforcement
   - HTTPS enforcement (deployment recommendation)

3. **Input Validation**
   - Pydantic models for all inputs
   - Sanitization for PDF export
   - Parameter validation on all endpoints

4. **Clinical Safety**
   - Maintain all existing emergency triage integration
   - Cannot bypass safety checks through conversation context
   - Emergency alerts highlighted in UI and exports

## Performance Optimization

1. **Caching Strategy**
   - LRU cache for patient profiles (frequently accessed)
   - In-memory session cache
   - No caching for conversations (always fresh)

2. **File Operations**
   - Conversation window retrieval optimized (load metadata first, then partial messages)
   - Atomic writes to prevent corruption
   - Efficient JSON serialization

3. **API Response Times**
   - Target: <200ms for message retrieval
   - Target: <100ms for message save
   - Target: <5s for agent response (LLM dependent)

## Testing Strategy

### Unit Tests

1. **ConversationStore Tests**
   - File creation, reading, writing
   - Conversation window retrieval with size/time limits
   - Message appending
   - List conversations with filtering

2. **ProfileManager Tests**
   - Profile CRUD operations
   - Validation of required fields
   - Cache behavior
   - Clinical context formatting

3. **SessionManager Tests**
   - Anonymous session generation
   - Session validation
   - Migration from anonymous to authenticated

4. **ExportService Tests**
   - PDF generation with various conversation sizes
   - Profile inclusion
   - Special character sanitization
   - Emergency highlighting

### Integration Tests

1. **End-to-End Conversation Flow**
   - Send message → save → retrieve → display
   - Multi-turn conversations with context
   - Profile-based personalization

2. **Agent Context Integration**
   - Profile data in agent prompts
   - Conversation window in agent context
   - Diagnosis-based retrieval prioritization

3. **Clinical Safety Integration**
   - Emergency triage still functional
   - Safety checks not bypassed
   - Emergency alerts displayed

### Property-Based Tests

See **Correctness Properties** section below for comprehensive property test specifications.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Message Persistence Round-Trip

*For any* message with content, timestamp, sender role, and conversation identifier, saving the message to the Conversation_Store and then loading the conversation should preserve all message fields exactly.

**Validates: Requirements 1.2, 1.3**

**Test Strategy**: Generate random messages with varying content types (text, special characters, unicode), roles (user/assistant/system), and timestamps. Save to store, load conversation, verify all fields match exactly including metadata.

### Property 2: Conversation Structure Completeness

*For any* conversation, the JSON file structure should contain all required metadata fields: conversation_id, patient_id, created_at, updated_at, and messages array.

**Validates: Requirements 1.4**

**Test Strategy**: Generate random conversations with varying message counts and metadata. Save to JSON, parse file, verify all required metadata fields are present and correctly typed.

### Property 3: Patient Association Consistency

*For any* conversation saved in authenticated mode with a patient_id, retrieving the conversation should return the same patient_id association.

**Validates: Requirements 1.5, 5.5**

**Test Strategy**: Generate random patient IDs and conversations in authenticated mode. Save conversations, retrieve by conversation_id, verify patient_id matches.

### Property 4: Timestamp Update Monotonicity

*For any* conversation, when a new message is appended, the updated_at timestamp should be greater than or equal to the previous updated_at timestamp.

**Validates: Requirements 1.7**

**Test Strategy**: Create conversation, record updated_at, append message, verify new updated_at >= old updated_at.

### Property 5: Export-Store Format Compatibility

*For any* conversation saved via Conversation_Store, the Export_Service should successfully parse and generate a valid PDF without errors.

**Validates: Requirements 1.8, 7.1**

**Test Strategy**: Generate random conversations with varying message counts and content types. Save via Conversation_Store, load via Export_Service, verify successful PDF generation and valid PDF structure.

### Property 6: Profile Update Merging

*For any* existing patient profile and a set of update fields, applying the updates should preserve all non-updated fields while correctly updating specified fields.

**Validates: Requirements 2.3**

**Test Strategy**: Create profiles with complete data, generate partial updates targeting specific fields, apply updates, verify only specified fields changed.

### Property 7: Profile Structure Validation

*For any* patient profile, all required fields (patient_id, diagnoses array, medications array, allergies array, last_updated) must be present after creation or update.

**Validates: Requirements 2.5, 2.7**

**Test Strategy**: Generate profiles with various field combinations. Attempt to save profiles missing required fields (should fail validation). Verify valid profiles contain all required fields.

### Property 8: Clinical Context Accessibility

*For any* patient profile and any clinical tool request, the tool should have access to the complete profile data including diagnoses, medications, and allergies.

**Validates: Requirements 2.4, 2.8**

**Test Strategy**: Create random profiles, invoke clinical tools (BP classifier, glucose analyzer, emergency triage), verify profile data is accessible in tool context.

### Property 9: Conversation Window Context Injection

*For any* new message in a conversation, the LangGraph_Agent should receive the Conversation_Window contents in the prompt context.

**Validates: Requirements 3.2**

**Test Strategy**: Create conversations with messages, send new message, capture agent prompt context, verify window messages are present.

### Property 10: Sliding Window Recency

*For any* conversation exceeding the window size limit, retrieving the conversation window should return only the most recent messages within the size and time constraints.

**Validates: Requirements 3.4**

**Test Strategy**: Generate conversations with 50+ messages spanning different time periods. Retrieve window with size=20 and days=7. Verify only recent 20 messages within 7 days are returned, chronologically ordered.

### Property 11: Conversation Window Chronological Ordering

*For any* conversation, retrieving the conversation window should return messages in chronological order by timestamp (oldest to newest).

**Validates: Requirements 3.6**

**Test Strategy**: Generate conversations with messages having random timestamps. Retrieve window, verify messages are sorted by timestamp ascending.

### Property 12: Storage-Window Independence

*For any* conversation exceeding the window size, all messages should be preserved in storage even when the conversation window only contains a subset.

**Validates: Requirements 3.8**

**Test Strategy**: Create conversations with 50+ messages. Retrieve window (returns 20), load full conversation from storage (should return all 50+).

### Property 13: Anonymous Session Uniqueness

*For any* set of anonymous session creation requests, each generated session identifier should be unique.

**Validates: Requirements 1.6, 5.2**

**Test Strategy**: Create 100+ anonymous sessions concurrently, collect session IDs, verify all are unique (no collisions).

### Property 14: Conversation Listing Completeness

*For any* patient with N conversations, listing conversations for that patient should return all N conversation metadata records.

**Validates: Requirements 8.2**

**Test Strategy**: Create random number of conversations (0-100) for a patient. List conversations, verify count matches and all conversation IDs are present.

### Property 15: Conversation Sorting by Recency

*For any* set of conversations for a patient, listing conversations should return them sorted by updated_at timestamp in descending order (most recent first).

**Validates: Requirements 8.8**

**Test Strategy**: Create conversations with varied updated_at timestamps. List conversations, verify order is descending by updated_at.

### Property 16: Authenticated Conversation Isolation

*For any* two different authenticated patients, listing conversations for patient A should return only conversations associated with patient A, not patient B.

**Validates: Requirements 8.9**

**Test Strategy**: Create conversations for multiple patients. List for patient A, verify no conversations from patient B are included.

### Property 17: Anonymous Conversation Isolation

*For any* two different anonymous sessions, listing conversations for session A should return only conversations associated with session A, not session B.

**Validates: Requirements 8.10**

**Test Strategy**: Create multiple anonymous sessions with conversations. List for session A, verify isolation from session B.

### Property 18: Title Generation from First Message

*For any* conversation without an explicit title, the generated title should be derived from the content of the first user message.

**Validates: Requirements 8.5**

**Test Strategy**: Create conversations with first user messages of varying length/content, no explicit title. Verify generated title uses first message content.

### Property 19: New Conversation Unique ID

*For any* set of new conversation creation requests, each conversation should receive a unique conversation_id.

**Validates: Requirements 8.7**

**Test Strategy**: Create 100+ new conversations concurrently, verify all conversation IDs are unique.

### Property 20: Export Message Completeness

*For any* conversation with N messages, exporting to PDF should include all N messages in the document.

**Validates: Requirements 7.2**

**Test Strategy**: Generate conversations with varying message counts (1-100). Export to PDF, parse PDF, count messages, verify count matches source conversation.

### Property 21: Export Metadata Inclusion

*For any* conversation export, the PDF should contain all required metadata fields: patient identifier, date range, and conversation identifier in the header.

**Validates: Requirements 7.4**

**Test Strategy**: Generate random conversations, export to PDF, parse PDF header, verify all metadata fields are present.

### Property 22: Export Profile Conditional Inclusion

*For any* conversation export where a patient profile exists, the PDF should include the profile information; where no profile exists, the PDF should omit the profile section.

**Validates: Requirements 7.5**

**Test Strategy**: Create conversations with and without associated profiles. Export both, verify profile section present only when profile exists.

### Property 23: Export Message Formatting

*For any* message in an exported PDF, the message should include a timestamp and sender label in the formatted output.

**Validates: Requirements 7.6**

**Test Strategy**: Generate conversations, export to PDF, parse message sections, verify each message has timestamp and sender label.

### Property 24: Export Emergency Highlighting

*For any* conversation containing messages with emergency triage warnings, the exported PDF should visually highlight those messages differently from normal messages.

**Validates: Requirements 7.7**

**Test Strategy**: Create conversations with emergency-flagged messages. Export to PDF, parse styling/formatting, verify emergency messages have distinct formatting.

### Property 25: Export Special Character Sanitization

*For any* conversation containing special characters or unicode in message content, exporting to PDF should produce a valid PDF file without corruption.

**Validates: Requirements 7.10**

**Test Strategy**: Generate conversations with special characters (emoji, unicode, HTML entities, markdown). Export to PDF, verify valid PDF structure and readable content.

### Property 26: Atomic Write Data Integrity

*For any* concurrent write operations to the same conversation file, the resulting file should contain valid JSON representing the complete state of one of the writes (no partial writes or corruption).

**Validates: Requirements 10.1, 10.8**

**Test Strategy**: Simulate concurrent writes to the same conversation. After completion, verify file contains valid JSON with complete conversation structure.

### Property 27: File System Error Graceful Handling

*For any* file system error during conversation save (disk full, permission denied), the Conversation_Store should log the error and return an error message without crashing.

**Validates: Requirements 10.2, 10.4**

**Test Strategy**: Simulate file system errors (mock disk full, permission denied). Attempt save operations, verify error logged and graceful error response returned.

### Property 28: UTF-8 Encoding Preservation

*For any* conversation containing unicode characters (non-ASCII), saving to JSON and re-loading should preserve all unicode characters exactly.

**Validates: Requirements 10.5**

**Test Strategy**: Create conversations with various unicode characters (emoji, non-Latin scripts, special symbols). Save, reload, verify character preservation.

### Property 29: JSON Structure Validation

*For any* invalid JSON structure (missing required fields, wrong types), attempting to save should fail with validation error before writing to disk.

**Validates: Requirements 10.6**

**Test Strategy**: Construct invalid conversation objects (missing conversation_id, wrong type for timestamp). Attempt save, verify validation error raised before file write.

### Property 30: Malformed JSON Read Handling

*For any* malformed JSON file in the conversations directory, attempting to load should return None or raise a handled exception without crashing the application.

**Validates: Requirements 10.7**

**Test Strategy**: Create malformed JSON files (syntax errors, truncated). Attempt to load, verify graceful handling with logging.

### Property 31: File Naming Convention Consistency

*For any* saved conversation, the filename should follow the pattern `{patient_id}_{conversation_id}.json`.

**Validates: Requirements 10.9**

**Test Strategy**: Create conversations with various patient IDs and conversation IDs. Save, list files, verify all follow naming pattern.

### Property 32: JSON Human-Readable Formatting

*For any* saved conversation JSON file, the file should be formatted with indentation (pretty-printed) for human readability.

**Validates: Requirements 10.10**

**Test Strategy**: Save conversations, read raw file contents, verify JSON is indented (not minified).

### Property 33: API Input Validation

*For any* API endpoint receiving a request with missing required parameters, the endpoint should return a 400 Bad Request status code with a descriptive error message.

**Validates: Requirements 11.7**

**Test Strategy**: Send requests to all endpoints with missing required parameters. Verify 400 status and error messages.

### Property 34: API Error Response Consistency

*For any* API endpoint encountering an error condition, the endpoint should return an appropriate HTTP status code (4xx for client errors, 5xx for server errors) with a JSON error message.

**Validates: Requirements 11.8**

**Test Strategy**: Trigger various error conditions (not found, validation error, server error). Verify correct status codes and JSON error format.

### Property 35: API Response Format Consistency

*For any* API endpoint except export, the response should be valid JSON; the export endpoint should return a valid PDF file.

**Validates: Requirements 11.11**

**Test Strategy**: Call all API endpoints, verify response content types (JSON for most, PDF for export).

### Property 36: Authenticated API Token Validation

*For any* authenticated API endpoint receiving a request with an invalid or missing authentication token, the endpoint should return a 401 Unauthorized status code.

**Validates: Requirements 11.12**

**Test Strategy**: Send requests to authenticated endpoints with invalid/missing tokens. Verify 401 status returned.

### Property 37: Profile Cache Consistency

*For any* patient profile accessed multiple times within a short period, subsequent accesses should use the cached version without re-reading from disk, and the cached version should match the disk version.

**Validates: Requirements 12.5**

**Test Strategy**: Load profile, modify in cache, access again, verify cache hit. Load from disk independently, verify cache matches disk.

### Property 38: Concurrent Request Data Integrity

*For any* set of concurrent requests from multiple patients, each request should operate on the correct patient's data without cross-contamination.

**Validates: Requirements 12.7**

**Test Strategy**: Simulate 100+ concurrent requests for different patients. Verify each operation touches only the correct patient's data.

### Property 39: Configuration Environment Variable Respect

*For any* configurable parameter (window size, retention period, storage paths), setting the corresponding environment variable should result in the system using that value.

**Validates: Requirements 13.2, 13.3, 13.4**

**Test Strategy**: Set various configuration environment variables. Start system, verify configuration values match environment variables.

### Property 40: Configuration Validation at Startup

*For any* invalid configuration parameter value (negative numbers, invalid paths), the system should detect the invalidity at startup, log a warning, and use default values.

**Validates: Requirements 13.5, 13.6**

**Test Strategy**: Provide invalid configuration values (negative window size, non-existent paths). Start system, verify warnings logged and defaults used.

## Migration and Deployment

### Migration from Current System

The chatbot functionality is purely additive - no existing features are modified or removed:

1. **Backward Compatibility**: All existing endpoints (`/api/chat`, `/api/tools/*`) remain unchanged
2. **Optional Features**: Chatbot features are opt-in through new endpoints
3. **Data Isolation**: New storage directories don't conflict with existing data
4. **Gradual Adoption**: Users can continue using single-query mode or adopt conversation mode

### Deployment Steps

1. **Install Dependencies**:
   ```bash
   pip install reportlab
   ```

2. **Create Storage Directories**:
   ```bash
   mkdir -p data/conversations
   mkdir -p data/profiles
   ```

3. **Update Configuration** (optional):
   ```bash
   export CONVERSATION_WINDOW_SIZE=20
   export CONVERSATION_WINDOW_DAYS=7
   export ANONYMOUS_MODE_ENABLED=true
   ```

4. **Run Migrations** (if needed):
   - None required for initial deployment (purely additive)

5. **Deploy Updated Code**:
   - Deploy new modules: `chatbot/`
   - Deploy enhanced: `agent/assistant.py`, `ui/web.py`, `ui/templates/`, `ui/static/`
   - Deploy updated: `config.py`

6. **Test Deployment**:
   - Verify existing functionality unchanged
   - Test new chatbot endpoints
   - Test PDF export
   - Verify file storage working

### Rollback Plan

If issues arise, rollback is safe because:
1. New features don't modify existing functionality
2. Data stored in separate directories
3. Can disable new endpoints via configuration
4. Existing users unaffected

## Future Enhancements

1. **Authentication System**: OAuth2/JWT integration for proper user authentication
2. **Conversation Search**: Full-text search across conversation history
3. **Voice Input**: Speech-to-text for message input
4. **Multi-language Support**: Internationalization for global users
5. **Real-time Notifications**: WebSocket for real-time message delivery
6. **Conversation Sharing**: Share conversations with healthcare providers via secure links
7. **Advanced Analytics**: Insights from conversation patterns and health trends
8. **Mobile App**: Native mobile clients for iOS/Android
9. **Database Migration**: Move from JSON files to PostgreSQL/MongoDB for scale
10. **Conversation Summarization**: LLM-powered conversation summaries

## Conclusion

This design provides a comprehensive enhancement to the existing RAG health assistant, transforming it into a fully-featured conversational chatbot while maintaining all clinical safety features. The JSON-based storage approach ensures simplicity and zero database dependencies, making deployment straightforward. The hybrid authentication model supports both anonymous exploration and authenticated persistent use, maximizing accessibility while enabling personalization.

The property-based testing approach ensures correctness across all conversation storage, profile management, export, and API operations. The modular design allows for future enhancements without disrupting existing functionality.

