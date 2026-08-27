# Implementation Plan: Patient Chatbot Functionality

## Overview

This implementation plan transforms the existing AuraHealth AI RAG health assistant into a comprehensive conversational chatbot with persistent conversation history, patient profile management, multi-turn memory, and PDF export capabilities. The implementation leverages Python 3.12+, FastAPI, LangChain/LangGraph, and ReportLab for PDF generation, using JSON file-based storage for simplicity and zero database dependencies.

## Tasks

- [x] 1. Set up chatbot module structure and core data models
  - Create the `src/rag_health_assistant/chatbot/` directory
  - Implement `chatbot/__init__.py` with module exports
  - Implement `chatbot/models.py` with Pydantic data models for Message, Conversation, PatientProfile, Session
  - Define MessageRole, SessionType, MedicalCondition, Medication, Allergy classes
  - Ensure all models have proper validation, default values, and type hints
  - _Requirements: 1.2, 1.4, 2.1, 2.5_

- [x]* 1.1 Write property tests for data models
  - **Property 7: Profile Structure Validation** - Validates: Requirements 2.5, 2.7
  - Generate profiles with various field combinations, verify validation works
  - **Property 29: JSON Structure Validation** - Validates: Requirements 10.6
  - Test invalid structures fail validation before disk write
  - _Requirements: 2.5, 2.7, 10.6_

- [x] 2. Implement ConversationStore for persistent storage
  - [x] 2.1 Create `chatbot/conversation_store.py` with ConversationStore class
    - Implement `__init__` with storage directory initialization
    - Implement `save_conversation` with atomic write operations and file locking
    - Implement `load_conversation` with malformed JSON handling
    - Implement `append_message` with timestamp updates
    - Implement `list_conversations` with sorting by updated_at descending
    - Implement `delete_conversation` with soft delete flag
    - Implement `get_conversation_window` with size and time constraints
    - Implement `create_conversation` helper method for new conversations
    - Use pathlib for cross-platform file path handling
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 10.1, 10.2, 10.5, 10.8, 10.9_

  - [x]* 2.2 Write property tests for ConversationStore - Message Persistence
    - **Property 1: Message Persistence Round-Trip** - Validates: Requirements 1.2, 1.3
    - Generate random messages with special characters, save and load, verify exact match
    - **Property 2: Conversation Structure Completeness** - Validates: Requirements 1.4
    - Verify all required metadata fields present in saved conversations
    - **Property 28: UTF-8 Encoding Preservation** - Validates: Requirements 10.5
    - Test unicode characters are preserved in save/load cycle
    - _Requirements: 1.2, 1.3, 1.4, 10.5_

  - [x]* 2.3 Write property tests for ConversationStore - Associations and Timestamps
    - **Property 3: Patient Association Consistency** - Validates: Requirements 1.5, 5.5
    - Verify patient_id associations are preserved
    - **Property 4: Timestamp Update Monotonicity** - Validates: Requirements 1.7
    - Verify updated_at increases monotonically with appends
    - **Property 31: File Naming Convention Consistency** - Validates: Requirements 10.9
    - Verify all saved files follow {patient_id}_{conversation_id}.json pattern
    - _Requirements: 1.5, 1.7, 5.5, 10.9_

  - [x]* 2.4 Write property tests for ConversationStore - Error Handling
    - **Property 26: Atomic Write Data Integrity** - Validates: Requirements 10.1, 10.8
    - Simulate concurrent writes, verify no corruption
    - **Property 27: File System Error Graceful Handling** - Validates: Requirements 10.2, 10.4
    - Mock disk full/permission errors, verify graceful handling
    - **Property 30: Malformed JSON Read Handling** - Validates: Requirements 10.7
    - Create corrupted JSON files, verify graceful load failure
    - _Requirements: 10.1, 10.2, 10.4, 10.7, 10.8_

  - [x]* 2.5 Write property tests for ConversationStore - Conversation Window
    - **Property 10: Sliding Window Recency** - Validates: Requirements 3.4
    - Generate 50+ messages, verify window returns only recent 20 within 7 days
    - **Property 11: Conversation Window Chronological Ordering** - Validates: Requirements 3.6
    - Verify messages in window are sorted by timestamp ascending
    - **Property 12: Storage-Window Independence** - Validates: Requirements 3.8
    - Verify full conversation storage preserved independent of window size
    - _Requirements: 3.4, 3.6, 3.8_

- [x] 3. Implement ProfileManager for patient medical context
  - [x] 3.1 Create `chatbot/profile_manager.py` with ProfileManager class
    - Implement `__init__` with storage directory and LRU cache initialization
    - Implement `create_profile` for new patients (authenticated and anonymous)
    - Implement `save_profile` with cache updates and timestamp management
    - Implement `load_profile` with cache lookup and disk fallback
    - Implement `update_profile` with partial update support and validation
    - Implement `add_diagnosis`, `add_medication`, `add_allergy` helpers
    - Implement `get_clinical_context_summary` for LLM prompt formatting
    - Ensure UTF-8 encoding and proper JSON serialization
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.6, 2.7, 10.5_

  - [x]* 3.2 Write property tests for ProfileManager
    - **Property 6: Profile Update Merging** - Validates: Requirements 2.3
    - Test partial updates preserve non-updated fields
    - **Property 8: Clinical Context Accessibility** - Validates: Requirements 2.4, 2.8
    - Verify profile data is accessible to clinical tools
    - **Property 37: Profile Cache Consistency** - Validates: Requirements 12.5
    - Test cache hits match disk versions
    - _Requirements: 2.3, 2.4, 2.8, 12.5_

- [x] 4. Implement SessionManager for authentication tracking
  - [x] 4.1 Create `chatbot/session_manager.py` with SessionManager class
    - Implement `__init__` with authentication mode configuration
    - Implement `create_anonymous_session` with UUID-based ID generation
    - Implement `get_or_create_session` for both authenticated and anonymous modes
    - Implement `migrate_anonymous_to_authenticated` for account creation flow
    - Implement `validate_session` with timeout enforcement
    - Implement `get_patient_id` helper method
    - Use in-memory session tracking with thread-safe data structures
    - _Requirements: 1.6, 5.1, 5.2, 5.3, 5.4, 5.5, 5.7_

  - [x]* 4.2 Write property tests for SessionManager
    - **Property 13: Anonymous Session Uniqueness** - Validates: Requirements 1.6, 5.2
    - Create 100+ anonymous sessions concurrently, verify all IDs are unique
    - **Property 19: New Conversation Unique ID** - Validates: Requirements 8.7
    - Create 100+ new conversations, verify unique conversation IDs
    - _Requirements: 1.6, 5.2, 8.7_

- [x] 5. Checkpoint - Ensure core infrastructure tests pass
  - Run all property tests for data models, ConversationStore, ProfileManager, and SessionManager
  - Verify file storage operations work correctly
  - Verify JSON serialization and UTF-8 encoding
  - Ask the user if questions arise

- [x] 6. Implement ExportService for PDF generation
  - [x] 6.1 Create `chatbot/export_service.py` with ExportService class
    - Add `reportlab` dependency to project requirements
    - Implement `__init__` with ConversationStore and ProfileManager dependencies
    - Implement `generate_pdf` with BytesIO buffer output
    - Implement `_build_header` for PDF metadata section
    - Implement `_build_profile_section` for patient clinical summary
    - Implement `_build_messages_section` with timestamp and sender formatting
    - Implement `_build_footer` with medical disclaimer
    - Implement `_sanitize_content` for special character handling
    - Use reportlab SimpleDocTemplate, Paragraph, Table for professional styling
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.10_

  - [x]* 6.2 Write property tests for ExportService
    - **Property 5: Export-Store Format Compatibility** - Validates: Requirements 1.8, 7.1
    - Generate random conversations, verify successful PDF generation
    - **Property 20: Export Message Completeness** - Validates: Requirements 7.2
    - Verify all N messages appear in exported PDF
    - **Property 21: Export Metadata Inclusion** - Validates: Requirements 7.4
    - Verify PDF header contains all required metadata
    - **Property 22: Export Profile Conditional Inclusion** - Validates: Requirements 7.5
    - Test profile section present only when profile exists
    - **Property 23: Export Message Formatting** - Validates: Requirements 7.6
    - Verify timestamps and sender labels in PDF messages
    - **Property 24: Export Emergency Highlighting** - Validates: Requirements 7.7
    - Verify emergency messages have distinct formatting in PDF
    - **Property 25: Export Special Character Sanitization** - Validates: Requirements 7.10
    - Test emoji, unicode, HTML entities produce valid PDFs
    - _Requirements: 1.8, 7.1, 7.2, 7.4, 7.5, 7.6, 7.7, 7.10_

- [x] 7. Enhance LangGraph agent for conversation context
  - [x] 7.1 Modify `agent/assistant.py` HealthAgent.chat method
    - Add `patient_profile` parameter (Optional[PatientProfile])
    - Add `conversation_window` parameter (Optional[List[Message]])
    - Implement `_build_system_context` to inject patient diagnoses, medications, allergies into system prompt
    - Modify message sequence construction to include conversation window messages
    - Convert window messages to appropriate LangChain message types (HumanMessage, AIMessage)
    - Preserve existing emergency triage and clinical tools integration
    - _Requirements: 3.1, 3.2, 3.7, 4.1, 4.2, 9.1, 9.2_

  - [x] 7.2 Enhance tool creation for diagnosis-based retrieval prioritization
    - Modify `create_agent_tools` to accept `patient_profile` parameter
    - Update `retrieve_medical_guidelines` tool to prioritize based on patient diagnoses
    - Implement condition filtering logic (diabetes, hypertension priority)
    - Ensure existing hybrid retrieval (ChromaDB + BM25) integration maintained
    - Update `_format_guideline_results` to include patient-relevant context
    - _Requirements: 4.3, 4.4, 4.7, 9.5, 9.6_

  - [x]* 7.3 Write property tests for agent context integration
    - **Property 9: Conversation Window Context Injection** - Validates: Requirements 3.2
    - Send message, capture agent prompt, verify window messages present
    - _Requirements: 3.2_

- [x] 8. Implement chatbot configuration in config.py
  - Update `src/rag_health_assistant/config.py` with chatbot settings
  - Define `CONVERSATIONS_DIR` and `PROFILES_DIR` paths under data/
  - Create directories if they don't exist
  - Add `CONVERSATION_WINDOW_SIZE` (default: 20)
  - Add `CONVERSATION_WINDOW_DAYS` (default: 7)
  - Add `SESSION_TIMEOUT_HOURS` (default: 24)
  - Add `ANONYMOUS_MODE_ENABLED` (default: true)
  - Add `AUTHENTICATED_MODE_ENABLED` (default: false)
  - Add PDF configuration: `PDF_PAGE_SIZE`, `PDF_INCLUDE_PROFILE`, `PDF_INCLUDE_METADATA`
  - Add performance configuration: `PROFILE_CACHE_SIZE`, `CONVERSATION_LIST_LIMIT`
  - Add environment variable support for all configuration parameters
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.7, 13.8_

- [x]* 8.1 Write property tests for configuration
  - **Property 39: Configuration Environment Variable Respect** - Validates: Requirements 13.2, 13.3, 13.4
  - Set environment variables, verify system uses those values
  - **Property 40: Configuration Validation at Startup** - Validates: Requirements 13.5, 13.6
  - Provide invalid config values, verify warnings logged and defaults used
  - _Requirements: 13.2, 13.3, 13.4, 13.5, 13.6_

- [x] 9. Checkpoint - Ensure agent enhancements and configuration work
  - Test agent with profile context and conversation window
  - Verify diagnosis-based retrieval prioritization
  - Test configuration loading from environment variables
  - Ask the user if questions arise

- [x] 10. Extend FastAPI web.py with conversation endpoints
  - [x] 10.1 Add conversation management endpoints to `ui/web.py`
    - Initialize ConversationStore, ProfileManager, SessionManager, ExportService in app startup
    - Implement `POST /api/chat/send` for sending messages with conversation persistence
    - Implement `GET /api/chat/history/{conversation_id}` for retrieving full conversation
    - Implement `GET /api/chat/sessions` for listing all patient conversations
    - Implement `POST /api/chat/sessions/new` for creating new conversation sessions
    - Add request/response Pydantic models: ChatSendRequest, NewSessionRequest, etc.
    - Ensure all endpoints validate session_id and extract patient_id correctly
    - _Requirements: 1.3, 8.1, 8.2, 8.3, 8.4, 8.6, 8.7, 11.1, 11.2, 11.5_

  - [x]* 10.2 Write property tests for conversation endpoints
    - **Property 14: Conversation Listing Completeness** - Validates: Requirements 8.2
    - Create N conversations, list for patient, verify all returned
    - **Property 15: Conversation Sorting by Recency** - Validates: Requirements 8.8
    - Create conversations with varied timestamps, verify descending sort
    - **Property 16: Authenticated Conversation Isolation** - Validates: Requirements 8.9
    - Verify patient A cannot see patient B conversations
    - **Property 17: Anonymous Conversation Isolation** - Validates: Requirements 8.10
    - Verify session A cannot see session B conversations
    - **Property 18: Title Generation from First Message** - Validates: Requirements 8.5
    - Create conversations without titles, verify title derived from first message
    - _Requirements: 8.2, 8.5, 8.8, 8.9, 8.10_

- [x] 11. Extend FastAPI web.py with profile endpoints
  - [x] 11.1 Add profile management endpoints to `ui/web.py`
    - Implement `GET /api/profile` for retrieving patient profile
    - Implement `POST /api/profile/update` for partial profile updates
    - Implement `POST /api/profile/diagnosis/add` for adding diagnoses
    - Implement `POST /api/profile/medication/add` for adding medications
    - Implement `POST /api/profile/allergy/add` for adding allergies
    - Add request models: ProfileUpdateRequest, AddDiagnosisRequest, AddMedicationRequest, AddAllergyRequest
    - Ensure proper validation and error handling for all endpoints
    - _Requirements: 2.3, 11.3, 11.7, 11.8_

  - [x]* 11.2 Write unit tests for profile endpoints
    - Test profile creation for new users
    - Test profile updates preserve non-updated fields
    - Test diagnosis/medication/allergy additions
    - Test validation errors for invalid inputs
    - _Requirements: 2.3, 2.7_

- [x] 12. Extend FastAPI web.py with export and session endpoints
  - [x] 12.1 Add export and session migration endpoints to `ui/web.py`
    - Implement `GET /api/export/conversation/{conversation_id}` returning PDF Response
    - Implement `POST /api/session/migrate` for anonymous-to-authenticated migration
    - Ensure export endpoint sets proper Content-Disposition header for download
    - Implement migration logic to transfer conversations and profile ownership
    - _Requirements: 5.7, 5.8, 7.8, 7.9, 11.6, 11.11_

  - [x]* 12.2 Write property tests for API validation and errors
    - **Property 33: API Input Validation** - Validates: Requirements 11.7
    - Send requests with missing parameters, verify 400 status
    - **Property 34: API Error Response Consistency** - Validates: Requirements 11.8
    - Trigger various errors, verify correct status codes and JSON format
    - **Property 35: API Response Format Consistency** - Validates: Requirements 11.11
    - Verify all endpoints return JSON except export (PDF)
    - _Requirements: 11.7, 11.8, 11.11_

- [x] 13. Checkpoint - Ensure all API endpoints work correctly
  - Test all conversation, profile, export, and session endpoints
  - Verify request validation and error handling
  - Test PDF export end-to-end
  - Ask the user if questions arise

- [x] 14. Enhance chat UI templates
  - [x] 14.1 Update `ui/templates/index.html` with chat interface
    - Add chat container layout with sidebar and main chat area
    - Create conversation sidebar with conversation list and new conversation button
    - Create profile summary section in sidebar
    - Create main chat area with header, messages container, typing indicator, message input
    - Add message display components with sender labels and timestamps
    - Add export button in chat header
    - Add mode indicator badge (Anonymous/Authenticated)
    - Create profile edit modal for updating medical information
    - Integrate with existing FastAPI template rendering
    - _Requirements: 6.1, 6.2, 6.4, 6.5, 6.6, 6.8, 6.9, 6.10, 6.12, 7.8_

  - [x]* 14.2 Write integration tests for UI rendering
    - Test template renders without errors
    - Verify all required UI elements present
    - Test emergency message display styling
    - _Requirements: 6.1, 6.8, 9.3_

- [x] 15. Enhance chat UI JavaScript functionality
  - [x] 15.1 Update `ui/static/app.js` with chat logic
    - Implement `initializeChat()` for session initialization and data loading
    - Implement `loadConversations()` to fetch and render conversation list
    - Implement `loadConversation(conversationId)` to load specific conversation history
    - Implement `sendMessage(content)` with API call and UI updates
    - Implement `addMessageToUI(message)` for rendering messages with timestamps
    - Implement typing indicator show/hide functions
    - Implement `exportConversation()` for PDF download
    - Implement `loadProfile()` and `updateProfile()` for profile management
    - Implement `showEmergencyAlert(emergency)` for clinical alerts
    - Add event listeners for send button, Enter key, new conversation button
    - Implement auto-scroll to latest message
    - Add session ID management with cookies
    - _Requirements: 6.3, 6.4, 6.7, 6.10, 7.8, 9.3_

  - [x]* 15.2 Write unit tests for JavaScript functions
    - Test message rendering with various content types
    - Test emergency alert display
    - Test session management
    - _Requirements: 6.3, 6.4, 9.3_

- [x] 16. Enhance chat UI styling
  - [x] 16.1 Update `ui/static/style.css` with chat styles
    - Add chat container layout styles (flexbox for sidebar and main area)
    - Add conversation sidebar styles with hover and active states
    - Add message styling with sender-based alignment (user right, assistant left)
    - Add typing indicator animation (three dots)
    - Add message input container and button styles
    - Add emergency message highlighting styles (red border, warning background)
    - Add profile summary section styles
    - Add modal styles for profile editing
    - Add responsive design for mobile devices
    - Ensure accessibility compliance (color contrast, focus states)
    - _Requirements: 6.2, 6.8, 6.9, 9.3_

  - [x]* 16.2 Write accessibility tests for UI
    - Test color contrast ratios meet WCAG AA standards
    - Test keyboard navigation works for all interactive elements
    - Test screen reader compatibility
    - _Requirements: 6.2, 6.8_

- [x] 17. Implement clinical safety integration verification
  - [x] 17.1 Verify emergency triage integration preserved
    - Test that Emergency_Triage tool is invoked during message processing
    - Test emergency alerts displayed in UI when triggered
    - Test emergency messages highlighted in PDF exports
    - Verify conversation features don't bypass safety checks
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.8_

  - [x]* 17.2 Write integration tests for clinical safety
    - Test emergency symptoms trigger immediate alerts
    - Test BP_Classifier and Glucose_Analyzer still accessible
    - Test guideline citations appear in responses
    - Verify no safety checks bypassed through conversation context
    - _Requirements: 9.1, 9.2, 9.4, 9.5, 9.7, 9.8_

- [x] 18. Implement performance optimizations
  - [x] 18.1 Add caching and performance tuning
    - Implement LRU cache in ProfileManager (already in design)
    - Add conversation window retrieval optimization (load metadata first)
    - Implement atomic write operations with temp files
    - Add connection pooling for concurrent requests
    - Verify performance targets: <200ms message retrieval, <100ms save, <5s agent response
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8_

  - [x]* 18.2 Write property tests for concurrency and performance
    - **Property 38: Concurrent Request Data Integrity** - Validates: Requirements 12.7
    - Simulate 100+ concurrent requests for different patients, verify data isolation
    - **Property 32: JSON Human-Readable Formatting** - Validates: Requirements 10.10
    - Verify saved JSON files are pretty-printed (indented)
    - _Requirements: 10.10, 12.7_

- [x] 19. Final checkpoint - End-to-end testing
  - Test complete conversation flow: create session → send message → view history → export PDF
  - Test profile creation and updates with conversation personalization
  - Test anonymous mode and authenticated mode separately
  - Test conversation list and session management
  - Verify all property tests pass
  - Test emergency triage integration
  - Verify PDF exports are valid and complete
  - Test with multiple concurrent users
  - Ask the user if questions arise

- [x] 20. Final integration and documentation
  - Verify backward compatibility with existing single-query endpoints
  - Test that existing functionality unchanged by new features
  - Ensure all environment variables documented in config.py
  - Verify data storage directories created on startup
  - Test deployment steps from design document
  - Run full test suite (unit tests, property tests, integration tests)
  - Ensure all 13 requirements and 40 correctness properties validated

## Notes

- Tasks marked with `*` are optional property-based tests that validate correctness properties - these can be skipped for faster MVP but are highly recommended for production robustness
- Core implementation tasks (without `*`) must be completed to deliver working chatbot functionality
- Each task references specific requirements from requirements.md for traceability
- The implementation is purely additive - no existing features are modified or removed
- Property tests validate universal correctness properties across all scenarios
- Unit tests and integration tests validate specific examples and end-to-end flows
- Checkpoints ensure incremental validation at logical breakpoints
- All code should follow Python best practices: type hints, docstrings, error handling
- Use pathlib for cross-platform file path compatibility
- Ensure UTF-8 encoding for all file operations
- Implement proper logging for debugging and monitoring
- Follow existing codebase conventions and patterns

## Task Dependency Graph

```json
{
  "waves": [
    {
      "id": 0,
      "tasks": ["1", "8"]
    },
    {
      "id": 1,
      "tasks": ["1.1", "2.1", "3.1", "4.1", "8.1"]
    },
    {
      "id": 2,
      "tasks": ["2.2", "2.3", "2.4", "2.5", "3.2", "4.2", "6.1"]
    },
    {
      "id": 3,
      "tasks": ["6.2", "7.1", "7.2"]
    },
    {
      "id": 4,
      "tasks": ["7.3", "10.1", "11.1", "12.1"]
    },
    {
      "id": 5,
      "tasks": ["10.2", "11.2", "12.2", "14.1", "15.1", "16.1"]
    },
    {
      "id": 6,
      "tasks": ["14.2", "15.2", "16.2", "17.1", "18.1"]
    },
    {
      "id": 7,
      "tasks": ["17.2", "18.2"]
    }
  ]
}
```
