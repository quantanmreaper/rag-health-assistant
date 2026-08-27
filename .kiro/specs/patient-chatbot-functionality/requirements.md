# Requirements Document

## Introduction

This document specifies requirements for adding comprehensive patient chatbot functionality to the existing AuraHealth AI RAG health assistant for diabetes and hypertension. The system currently provides clinical tools (BP classifier, glucose analyzer, emergency triage) through a LangGraph ReAct agent with hybrid retrieval. The chatbot enhancement will enable natural, persistent conversations with clinical grounding, including conversation history storage, patient profiles with medical context, multi-turn memory, personalized responses, and conversation export capabilities.

## Glossary

- **Chatbot_System**: The conversational interface module that manages patient interactions, conversation persistence, and personalization
- **Conversation_Store**: The JSON-based storage system in the data/ directory that persists conversation history
- **Patient_Profile**: A structured data record containing patient medical context including diagnoses, medications, and demographic information
- **Session**: A single continuous interaction period between a patient and the Chatbot_System
- **Conversation_History**: The complete record of messages exchanged between a patient and the system across one or more sessions
- **Conversation_Window**: The subset of recent messages from Conversation_History used to maintain context (sliding window strategy)
- **Anonymous_Mode**: Operating mode where the Chatbot_System functions without patient authentication
- **Authenticated_Mode**: Operating mode where patients have registered accounts with persistent profiles
- **Clinical_Context**: Medical information from Patient_Profile and guideline knowledge base used to ground responses
- **Emergency_Triage**: Existing clinical safety tool that identifies medical emergencies
- **Export_Service**: The component that generates PDF reports of conversations for healthcare provider sharing
- **Chat_UI**: The enhanced web interface for natural conversational interaction
- **Message**: A single unit of communication in a conversation, containing text content, timestamp, and sender identification
- **LangGraph_Agent**: The existing ReAct agent implementation using Google Gemini
- **Hybrid_Retrieval**: The existing combination of ChromaDB vector search and BM25 keyword search
- **Clinical_Tools**: The existing set of tools including BP_Classifier, Glucose_Analyzer, and Emergency_Triage

## Requirements

### Requirement 1: Conversation History Storage

**User Story:** As a patient, I want my conversations to be saved and retrievable, so that I can continue discussions across multiple sessions without repeating information.

#### Acceptance Criteria

1. THE Conversation_Store SHALL store conversations as JSON files in the data/conversations/ directory
2. WHEN a Message is sent, THE Conversation_Store SHALL persist the Message with content, timestamp, sender role, and conversation identifier
3. WHEN a patient requests conversation history, THE Conversation_Store SHALL retrieve all messages for the specified conversation identifier
4. THE Conversation_Store SHALL structure each conversation file with metadata including conversation_id, patient_id, created_at, updated_at, and messages array
5. WHERE Authenticated_Mode is active, THE Conversation_Store SHALL associate conversations with patient account identifiers
6. WHERE Anonymous_Mode is active, THE Conversation_Store SHALL generate temporary conversation identifiers for session tracking
7. WHEN a conversation is saved, THE Conversation_Store SHALL update the updated_at timestamp
8. THE Conversation_Store SHALL ensure conversation files are readable by the Export_Service

### Requirement 2: Patient Profile Management

**User Story:** As a patient, I want the system to remember my medical information, so that I receive personalized advice relevant to my conditions and medications.

#### Acceptance Criteria

1. THE Chatbot_System SHALL support Patient_Profile storage containing diagnoses, medications, allergies, age, and relevant medical history
2. WHERE Authenticated_Mode is active, THE Chatbot_System SHALL store Patient_Profile as JSON files in data/profiles/ directory
3. WHEN a patient provides medical information, THE Chatbot_System SHALL update the Patient_Profile with the new information
4. WHEN generating responses, THE LangGraph_Agent SHALL access Patient_Profile data as Clinical_Context
5. THE Patient_Profile SHALL include fields for patient_id, name, date_of_birth, diagnoses array, medications array, allergies array, and last_updated timestamp
6. WHERE Anonymous_Mode is active, THE Chatbot_System SHALL support temporary session-based profiles
7. WHEN a Patient_Profile is created or updated, THE Chatbot_System SHALL validate required fields are present
8. THE Chatbot_System SHALL ensure Patient_Profile data is accessible to all Clinical_Tools

### Requirement 3: Multi-Turn Conversation Memory

**User Story:** As a patient, I want the chatbot to remember what we discussed earlier in the conversation, so that I can have natural, coherent discussions without repeating context.

#### Acceptance Criteria

1. THE Chatbot_System SHALL maintain a Conversation_Window containing recent messages for context
2. WHEN processing a new Message, THE LangGraph_Agent SHALL include the Conversation_Window in the prompt context
3. THE Chatbot_System SHALL configure the Conversation_Window to include the last 20 messages or 7 days of history, whichever is smaller
4. WHEN a conversation exceeds the Conversation_Window size, THE Chatbot_System SHALL retain only the most recent messages within the window
5. THE Chatbot_System SHALL include both user messages and assistant responses in the Conversation_Window
6. WHEN retrieving conversation context, THE Chatbot_System SHALL load messages in chronological order
7. THE LangGraph_Agent SHALL reference previous messages in the Conversation_Window when generating contextually relevant responses
8. THE Chatbot_System SHALL preserve the full Conversation_History in storage independent of the Conversation_Window size

### Requirement 4: Personalized Clinical Responses

**User Story:** As a patient with diabetes and hypertension, I want advice tailored to my specific conditions and medications, so that the guidance is relevant and safe for my situation.

#### Acceptance Criteria

1. WHEN generating responses, THE LangGraph_Agent SHALL incorporate Patient_Profile diagnoses into Clinical_Context
2. WHEN generating responses, THE LangGraph_Agent SHALL incorporate Patient_Profile medications into Clinical_Context
3. WHEN a patient has diabetes diagnosis, THE LangGraph_Agent SHALL prioritize diabetes-related guidelines from Hybrid_Retrieval
4. WHEN a patient has hypertension diagnosis, THE LangGraph_Agent SHALL prioritize hypertension-related guidelines from Hybrid_Retrieval
5. WHEN suggesting interventions, THE LangGraph_Agent SHALL check Patient_Profile allergies and medications for contraindications
6. THE LangGraph_Agent SHALL format responses to acknowledge the patient's specific conditions when relevant
7. WHERE a Patient_Profile contains multiple diagnoses, THE LangGraph_Agent SHALL consider interactions between conditions
8. THE Chatbot_System SHALL maintain existing clinical safety features including Emergency_Triage integration

### Requirement 5: Hybrid Authentication Support

**User Story:** As a user, I want the option to use the chatbot without creating an account or with a registered account, so that I can choose my level of engagement based on my needs.

#### Acceptance Criteria

1. THE Chatbot_System SHALL support both Anonymous_Mode and Authenticated_Mode
2. WHERE Anonymous_Mode is active, THE Chatbot_System SHALL generate temporary session identifiers without requiring login
3. WHERE Authenticated_Mode is active, THE Chatbot_System SHALL require patient authentication credentials
4. WHEN operating in Anonymous_Mode, THE Chatbot_System SHALL store conversations with session-based identifiers
5. WHEN operating in Authenticated_Mode, THE Chatbot_System SHALL associate conversations with patient account identifiers
6. THE Chat_UI SHALL display mode indicators showing whether the patient is authenticated or anonymous
7. THE Chatbot_System SHALL provide a mechanism for anonymous users to create accounts and migrate their session data
8. WHERE a patient switches from Anonymous_Mode to Authenticated_Mode, THE Chatbot_System SHALL offer to preserve the current session conversation

### Requirement 6: Enhanced Chat User Interface

**User Story:** As a patient, I want an intuitive chat interface that feels natural and responsive, so that I can easily communicate with the health assistant.

#### Acceptance Criteria

1. THE Chat_UI SHALL extend the existing FastAPI web interface with conversational components
2. THE Chat_UI SHALL display messages in a chronological chat layout with sender identification
3. WHEN a Message is sent, THE Chat_UI SHALL display the Message immediately with a sending indicator
4. WHEN a response is received, THE Chat_UI SHALL display the response with appropriate formatting
5. THE Chat_UI SHALL provide a text input field for composing messages
6. THE Chat_UI SHALL support multi-line message input
7. THE Chat_UI SHALL display typing indicators when the LangGraph_Agent is processing a response
8. THE Chat_UI SHALL provide visual distinction between user messages and assistant responses
9. THE Chat_UI SHALL include timestamps for messages
10. THE Chat_UI SHALL auto-scroll to the latest Message when new messages arrive
11. THE Chat_UI SHALL integrate with existing static/app.js and style.css files
12. WHERE Authenticated_Mode is active, THE Chat_UI SHALL display patient profile information in the interface

### Requirement 7: Conversation Export for Healthcare Providers

**User Story:** As a patient, I want to export my conversation history as a professional PDF report, so that I can share relevant discussions with my healthcare providers.

#### Acceptance Criteria

1. THE Export_Service SHALL generate PDF documents from Conversation_History
2. WHEN a patient requests export, THE Export_Service SHALL include all messages from the selected conversation
3. THE Export_Service SHALL format PDF documents with clinical report styling including headers, sections, and metadata
4. THE Export_Service SHALL include conversation metadata in the PDF header: patient identifier, date range, conversation identifier
5. WHERE a Patient_Profile exists, THE Export_Service SHALL include relevant profile information in the PDF
6. THE Export_Service SHALL format messages with timestamps and sender labels in the PDF
7. THE Export_Service SHALL highlight Emergency_Triage warnings or clinical alerts in the PDF
8. THE Chat_UI SHALL provide an export button to trigger PDF generation
9. WHEN PDF generation is complete, THE Export_Service SHALL provide the file for download
10. THE Export_Service SHALL sanitize conversation content to ensure PDF formatting compatibility

### Requirement 8: Conversation Session Management

**User Story:** As a patient, I want to start new conversations or continue previous ones, so that I can organize my discussions by topic or timeframe.

#### Acceptance Criteria

1. THE Chatbot_System SHALL support creating new conversation sessions
2. THE Chatbot_System SHALL support listing existing conversation sessions for a patient
3. WHEN a patient selects an existing conversation, THE Chatbot_System SHALL load the Conversation_History for that conversation
4. THE Chat_UI SHALL display a list of recent conversation sessions with titles and timestamps
5. WHERE a conversation has no explicit title, THE Chatbot_System SHALL generate a title from the first user Message
6. THE Chat_UI SHALL provide a control to start a new conversation session
7. WHEN a new conversation is started, THE Chatbot_System SHALL create a new conversation identifier
8. THE Chatbot_System SHALL sort conversation sessions by updated_at timestamp in descending order
9. WHERE Authenticated_Mode is active, THE Chatbot_System SHALL load conversation sessions associated with the authenticated patient
10. WHERE Anonymous_Mode is active, THE Chatbot_System SHALL load conversation sessions associated with the current temporary session identifier

### Requirement 9: Clinical Safety Integration

**User Story:** As a patient discussing health concerns, I want the system to maintain safety features while providing conversational support, so that emergencies are detected and I receive guideline-grounded information.

#### Acceptance Criteria

1. THE Chatbot_System SHALL maintain integration with the existing Emergency_Triage tool
2. WHEN processing patient messages, THE LangGraph_Agent SHALL invoke Emergency_Triage for symptom assessment
3. IF Emergency_Triage identifies an emergency condition, THEN THE Chatbot_System SHALL display an immediate alert message
4. THE Chatbot_System SHALL maintain integration with existing Clinical_Tools including BP_Classifier and Glucose_Analyzer
5. THE LangGraph_Agent SHALL continue using Hybrid_Retrieval to ground responses in medical guidelines
6. THE Chatbot_System SHALL preserve the existing guideline knowledge base integration
7. WHEN generating clinical advice, THE LangGraph_Agent SHALL cite relevant guideline sources from the knowledge base
8. THE Chatbot_System SHALL ensure conversational features do not bypass clinical safety checks

### Requirement 10: Data Persistence and Reliability

**User Story:** As a patient, I want my conversation data and profile to be reliably saved, so that I don't lose important health information or discussion history.

#### Acceptance Criteria

1. THE Conversation_Store SHALL write conversation data atomically to prevent corruption
2. WHEN saving conversation data, THE Conversation_Store SHALL handle file system errors gracefully
3. THE Chatbot_System SHALL create data/conversations/ and data/profiles/ directories if they do not exist
4. WHEN a write operation fails, THE Chatbot_System SHALL log the error and return an error message to the user
5. THE Conversation_Store SHALL use UTF-8 encoding for JSON files
6. THE Chatbot_System SHALL validate JSON structure before writing files
7. WHEN reading conversation data, THE Chatbot_System SHALL handle malformed JSON gracefully
8. THE Chatbot_System SHALL implement file locking to prevent concurrent write conflicts
9. THE Chatbot_System SHALL include conversation_id and patient_id in file names for organizational clarity
10. THE Chatbot_System SHALL ensure JSON files are human-readable for debugging and data recovery purposes

### Requirement 11: API Integration and Backend Support

**User Story:** As a developer, I want well-defined API endpoints for chatbot functionality, so that the frontend can interact with conversation and profile management reliably.

#### Acceptance Criteria

1. THE Chatbot_System SHALL expose FastAPI endpoints for sending messages
2. THE Chatbot_System SHALL expose FastAPI endpoints for retrieving conversation history
3. THE Chatbot_System SHALL expose FastAPI endpoints for creating and updating Patient_Profile
4. THE Chatbot_System SHALL expose FastAPI endpoints for listing conversation sessions
5. THE Chatbot_System SHALL expose FastAPI endpoints for starting new conversations
6. THE Chatbot_System SHALL expose FastAPI endpoints for exporting conversations
7. WHEN an API endpoint receives a request, THE Chatbot_System SHALL validate required parameters
8. WHEN an API endpoint encounters an error, THE Chatbot_System SHALL return appropriate HTTP status codes and error messages
9. THE Chatbot_System SHALL integrate with the existing ui/web.py FastAPI application
10. THE Chatbot_System SHALL support CORS configuration for frontend integration
11. THE Chatbot_System SHALL return JSON responses for all API endpoints except export which returns PDF
12. WHERE Authenticated_Mode is active, THE Chatbot_System SHALL validate authentication tokens in API requests

### Requirement 12: Performance and Scalability

**User Story:** As a patient, I want the chatbot to respond quickly even with conversation history, so that interactions feel natural and responsive.

#### Acceptance Criteria

1. WHEN loading a Conversation_Window, THE Chatbot_System SHALL retrieve messages within 200 milliseconds
2. WHEN saving a Message, THE Conversation_Store SHALL complete the write operation within 100 milliseconds
3. THE Chatbot_System SHALL limit Conversation_Window size to prevent performance degradation
4. WHEN generating responses, THE LangGraph_Agent SHALL complete within 5 seconds for standard queries
5. THE Chatbot_System SHALL implement caching for frequently accessed Patient_Profile data
6. THE Export_Service SHALL generate PDF documents within 10 seconds for conversations up to 100 messages
7. THE Chatbot_System SHALL handle concurrent requests from multiple patients without data corruption
8. THE Chat_UI SHALL render up to 100 messages without performance degradation

### Requirement 13: Configuration and Extensibility

**User Story:** As a system administrator, I want configurable parameters for conversation storage, so that I can tune the system based on deployment requirements.

#### Acceptance Criteria

1. THE Chatbot_System SHALL define configuration parameters in the existing config.py file
2. THE Chatbot_System SHALL support configurable Conversation_Window size through environment variables
3. THE Chatbot_System SHALL support configurable conversation retention period through environment variables
4. THE Chatbot_System SHALL support configurable storage paths for conversations and profiles through environment variables
5. THE Chatbot_System SHALL validate configuration parameters at startup
6. WHERE configuration parameters are invalid, THE Chatbot_System SHALL log warnings and use default values
7. THE Chatbot_System SHALL document all configuration parameters in config.py with inline comments
8. THE Chatbot_System SHALL support toggling between Anonymous_Mode and Authenticated_Mode through configuration
