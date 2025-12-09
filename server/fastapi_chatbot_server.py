#!/usr/bin/env python3
"""
FastAPI Server for Instagram RAG + Calendar Chatbot
Provides REST API endpoints for chat functionality
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import sys
import os
import json
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    from pinecone_rag import PineconeRAG, SearchResult
    from calendar_creator import InstagramEventCreator
    from goog_oauth import get_service, validate_token, get_token_status
except ImportError as e:
    print(f" Failed to import required modules: {e}")
    print("Please ensure all src/ modules exist and dependencies are installed")
    sys.exit(1)

# Import EventExtractionHelper from the streamlit app
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from streamlit_calendar_chatbot import EventExtractionHelper

# Constants
DATA_FILE = "data/metadata/scraped_provenance.json"
CHAT_HISTORY_FILE = "data/chat_history_api.json"
MAX_CHAT_HISTORY = 100

# Global variables
chatbot_instance = None

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message", min_length=1, max_length=1000)
    user_id: Optional[str] = Field(None, description="Optional user identifier")

class EventData(BaseModel):
    """Event data model"""
    title: str
    description: str
    location: Optional[str] = ""
    organizer: Optional[str] = "Instagram"
    source_url: Optional[str] = ""
    start_date: Optional[str] = None
    start_datetime: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    confidence: float = 0.0

class ChatResponse(BaseModel):
    """Chat response model"""
    message: str = Field(..., description="Assistant response")
    events: List[EventData] = Field(default_factory=list, description="Detected events")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source posts")
    confidence: float = Field(0.0, description="Response confidence score")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EventCreateRequest(BaseModel):
    """Event creation request"""
    event: EventData
    user_id: Optional[str] = None

class EventCreateResponse(BaseModel):
    """Event creation response"""
    success: bool
    message: str
    event_link: Optional[str] = None
    event_id: Optional[str] = None

class SystemStatus(BaseModel):
    """System status model"""
    rag_system: Dict[str, Any]
    calendar_system: Dict[str, Any]
    chat_history_count: int
    last_updated: str

class FastAPIChatbot:
    """FastAPI Chatbot with calendar integration"""
    
    def __init__(self):
        self.rag_system = None
        self.calendar_creator = None
        self.chat_history = {}  # Dict[user_id, List[ChatMessage]]
        self.detected_events = {}  # Dict[user_id, List[EventData]]
        self.setup_logging()
        self.load_chat_history()
    
    def setup_logging(self):
        """Setup logging"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/fastapi_chatbot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize_systems(self):
        """Initialize RAG and Calendar systems"""
        try:
            # Initialize RAG system
            if not self.rag_system:
                self.logger.info(" Initializing RAG system...")
                self.rag_system = PineconeRAG(DATA_FILE, index_name="acm-hackathon")
                await asyncio.to_thread(self.rag_system.load_posts)
                self.logger.info(" RAG system ready")
            
            # Initialize Calendar creator
            if not self.calendar_creator:
                self.calendar_creator = InstagramEventCreator(timezone="America/New_York")
                self.logger.info(" Calendar system ready")
            
            return True
            
        except Exception as e:
            self.logger.error(f" System initialization failed: {e}")
            return False
    
    def load_chat_history(self):
        """Load chat history from file"""
        try:
            if os.path.exists(CHAT_HISTORY_FILE):
                with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chat_history = data.get('chat_history', {})
                    self.detected_events = data.get('detected_events', {})
                self.logger.info(f" Loaded chat history for {len(self.chat_history)} users")
            else:
                self.chat_history = {}
                self.detected_events = {}
        except Exception as e:
            self.logger.error(f" Error loading chat history: {e}")
            self.chat_history = {}
            self.detected_events = {}
    
    def save_chat_history(self):
        """Save chat history to file"""
        try:
            os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)
            
            # Limit history per user
            for user_id in self.chat_history:
                if len(self.chat_history[user_id]) > MAX_CHAT_HISTORY:
                    self.chat_history[user_id] = self.chat_history[user_id][-MAX_CHAT_HISTORY:]
            
            data = {
                'chat_history': self.chat_history,
                'detected_events': self.detected_events,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f" Error saving chat history: {e}")
    
    def add_message(self, user_id: str, role: str, content: str, metadata: Dict = None):
        """Add message to user's chat history"""
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.chat_history[user_id].append(message)
        self.save_chat_history()
    
    async def search_and_extract_events(self, query: str) -> tuple[SearchResult, List[Dict]]:
        """Search for information and extract events"""
        # Perform RAG search
        result = await asyncio.to_thread(self.rag_system.search, query, 5)
        
        # Extract events from the search results
        events = await asyncio.to_thread(
            EventExtractionHelper.extract_events_from_response,
            result.answer, 
            result.relevant_posts
        )
        
        return result, events
    
    async def create_calendar_event(self, event_data: Dict) -> Dict:
        """Create a Google Calendar event"""
        try:
            # Authenticate if needed
            auth_result = await asyncio.to_thread(self.calendar_creator.authenticate)
            if not auth_result:
                return {'success': False, 'error': 'Calendar authentication failed'}
            
            # Create the event
            result = await asyncio.to_thread(self.calendar_creator.create_event, event_data)
            
            if result['success']:
                self.logger.info(f" Created calendar event: {event_data['title']}")
            else:
                self.logger.error(f" Failed to create event: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            error_msg = f"Calendar event creation failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        rag_status = {"status": "offline", "details": {}}
        calendar_status = {"status": "offline", "details": {}}
        
        # RAG System Status
        if self.rag_system:
            try:
                stats = self.rag_system.get_stats()
                rag_status = {
                    "status": "online",
                    "details": {
                        "posts_loaded": stats.get('local_posts_loaded', 0),
                        "vectors_indexed": stats.get('pinecone_total_vectors', 0),
                        "average_confidence": stats.get('average_confidence', 0),
                        "unique_keywords": stats.get('unique_keywords', 0)
                    }
                }
            except Exception as e:
                rag_status = {"status": "error", "error": str(e)}
        
        # Calendar System Status
        try:
            token_status = get_token_status()
            calendar_status = {
                "status": "online" if token_status.get('token_valid') else "needs_auth",
                "details": {
                    "credentials_exists": token_status.get('credentials_exists', False),
                    "token_exists": token_status.get('token_exists', False),
                    "token_valid": token_status.get('token_valid', False),
                    "token_expired": token_status.get('token_expired', False)
                }
            }
        except Exception as e:
            calendar_status = {"status": "error", "error": str(e)}
        
        return {
            "rag_system": rag_status,
            "calendar_system": calendar_status,
            "chat_history_count": sum(len(history) for history in self.chat_history.values()),
            "users_count": len(self.chat_history),
            "last_updated": datetime.now().isoformat()
        }
    
    def check_requirements(self) -> List[str]:
        """Check system requirements"""
        issues = []
        
        # Check data file
        if not os.path.exists(DATA_FILE):
            issues.append(f" Data file missing: {DATA_FILE}")
        
        # Check environment variables
        if not os.getenv("PINECONE_API_KEY"):
            issues.append(" PINECONE_API_KEY missing")
        
        if not os.getenv("ANTHROPIC_API_KEY"):
            issues.append(" ANTHROPIC_API_KEY missing")
        
        # Check credentials file
        if not os.path.exists("credentials.json"):
            issues.append(" credentials.json missing for Google Calendar")
        
        return issues

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global chatbot_instance
    
    # Startup
    print(" Starting FastAPI Chatbot Server...")
    chatbot_instance = FastAPIChatbot()
    
    # Check requirements
    issues = chatbot_instance.check_requirements()
    if issues:
        print(" Setup Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        print("  Server starting with limited functionality")
    else:
        # Initialize systems
        success = await chatbot_instance.initialize_systems()
        if success:
            print(" All systems initialized successfully")
        else:
            print(" System initialization failed")
    
    yield
    
    # Shutdown
    print(" Shutting down FastAPI Chatbot Server...")
    if chatbot_instance:
        chatbot_instance.save_chat_history()

# Create FastAPI app
app = FastAPI(
    title="Instagram RAG + Calendar Chatbot API",
    description="REST API for Instagram event search and Google Calendar integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Instagram RAG + Calendar Chatbot API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    status = chatbot_instance.get_system_status()
    
    # Determine overall health
    overall_status = "healthy"
    if status["rag_system"]["status"] == "error" or status["calendar_system"]["status"] == "error":
        overall_status = "degraded"
    elif status["rag_system"]["status"] == "offline" and status["calendar_system"]["status"] == "offline":
        overall_status = "offline"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "details": status
    }

@app.get("/status", response_model=SystemStatus, tags=["System"])
async def get_system_status():
    """Get detailed system status"""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    status_data = chatbot_instance.get_system_status()
    return SystemStatus(
        rag_system=status_data["rag_system"],
        calendar_system=status_data["calendar_system"],
        chat_history_count=status_data["chat_history_count"],
        last_updated=status_data["last_updated"]
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not chatbot_instance.rag_system:
        raise HTTPException(status_code=503, detail="RAG system not ready")
    
    user_id = request.user_id or "default"
    
    try:
        # Add user message to history
        chatbot_instance.add_message(user_id, "user", request.message)
        
        # Search and extract events
        search_result, detected_events = await chatbot_instance.search_and_extract_events(request.message)
        
        # Add source links to response
        response_with_links = EventExtractionHelper.add_source_links_to_response(
            search_result.answer, search_result.relevant_posts
        )
        
        # Convert events to EventData models
        event_models = []
        for event in detected_events:
            event_models.append(EventData(
                title=event.get('title', ''),
                description=event.get('description', ''),
                location=event.get('location', ''),
                organizer=event.get('organizer', 'Instagram'),
                source_url=event.get('source_url', ''),
                start_date=event.get('start_date'),
                start_datetime=event.get('start_datetime'),
                keywords=event.get('keywords', []),
                confidence=event.get('confidence', 0.0)
            ))
        
        # Prepare sources
        sources = []
        for post in search_result.relevant_posts[:5]:
            sources.append({
                "post_id": post.get('post_id', ''),
                "post_url": post.get('post_url', ''),
                "timestamp": post.get('timestamp', ''),
                "similarity_score": post.get('similarity_score', 0.0),
                "caption": post.get('caption', '')[:200] + "..." if len(post.get('caption', '')) > 200 else post.get('caption', ''),
                "transcribed_text": post.get('transcribed_text', '')[:200] + "..." if len(post.get('transcribed_text', '')) > 200 else post.get('transcribed_text', '')
            })
        
        # Save assistant response to history
        chatbot_instance.add_message(user_id, "assistant", search_result.answer, {
            "events": detected_events,
            "confidence": search_result.confidence_score,
            "sources": len(search_result.relevant_posts)
        })
        
        # Store detected events for user
        if user_id not in chatbot_instance.detected_events:
            chatbot_instance.detected_events[user_id] = []
        chatbot_instance.detected_events[user_id].extend(detected_events)
        
        return ChatResponse(
            message=response_with_links,
            events=event_models,
            sources=sources,
            confidence=search_result.confidence_score,
            metadata={
                "events_detected": len(detected_events),
                "sources_found": len(search_result.relevant_posts),
                "user_id": user_id
            }
        )
        
    except Exception as e:
        chatbot_instance.logger.error(f"Chat error: {e}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        
        # Save error to history
        chatbot_instance.add_message(user_id, "assistant", error_msg, {"error": str(e)})
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/events/create", response_model=EventCreateResponse, tags=["Calendar"])
async def create_event(request: EventCreateRequest):
    """Create a calendar event"""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not chatbot_instance.calendar_creator:
        raise HTTPException(status_code=503, detail="Calendar system not ready")
    
    try:
        # Convert EventData to dict
        event_dict = request.event.dict()
        
        # Create the event
        result = await chatbot_instance.create_calendar_event(event_dict)
        
        return EventCreateResponse(
            success=result['success'],
            message=result.get('message', result.get('error', 'Unknown result')),
            event_link=result.get('event_link'),
            event_id=result.get('event_id')
        )
        
    except Exception as e:
        chatbot_instance.logger.error(f"Event creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Event creation failed: {str(e)}")

@app.get("/chat/history/{user_id}", tags=["Chat"])
async def get_chat_history(user_id: str, limit: int = 50):
    """Get chat history for a user"""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    user_history = chatbot_instance.chat_history.get(user_id, [])
    
    # Apply limit
    if limit > 0:
        user_history = user_history[-limit:]
    
    return {
        "user_id": user_id,
        "message_count": len(user_history),
        "messages": user_history,
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/chat/history/{user_id}", tags=["Chat"])
async def clear_chat_history(user_id: str):
    """Clear chat history for a user"""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    # Clear history and events for user
    if user_id in chatbot_instance.chat_history:
        del chatbot_instance.chat_history[user_id]
    if user_id in chatbot_instance.detected_events:
        del chatbot_instance.detected_events[user_id]
    
    chatbot_instance.save_chat_history()
    
    return {
        "message": f"Chat history cleared for user: {user_id}",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/events/detected/{user_id}", tags=["Calendar"])
async def get_detected_events(user_id: str):
    """Get detected events for a user"""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    user_events = chatbot_instance.detected_events.get(user_id, [])
    
    return {
        "user_id": user_id,
        "events_count": len(user_events),
        "events": user_events,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    print(" Starting Instagram RAG + Calendar Chatbot API Server...")
    print(" API Documentation will be available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "server.fastapi_chatbot_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )