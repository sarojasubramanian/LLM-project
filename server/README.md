# FastAPI Chatbot Server

REST API server for Instagram RAG + Google Calendar Chatbot functionality.

##  Features

- **RESTful API**: Send/receive chatbot responses via HTTP
- **Event Detection**: Automatically detect events from Instagram posts
- **Calendar Integration**: Create Google Calendar events via API
- **Multi-User Support**: Separate chat history per user
- **Real-time Processing**: Async operations for better performance
- **Interactive Documentation**: Auto-generated API docs at `/docs`

##  API Endpoints

### Health & Status
- `GET /` - Root endpoint
- `GET /health` - Health check with system status
- `GET /status` - Detailed system status

### Chat Operations
- `POST /chat` - Send message and get response with detected events
- `GET /chat/history/{user_id}` - Get chat history for user
- `DELETE /chat/history/{user_id}` - Clear chat history for user

### Calendar Operations
- `POST /events/create` - Create Google Calendar event
- `GET /events/detected/{user_id}` - Get detected events for user

## 🛠️ Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install fastapi uvicorn pydantic requests
```

### 2. Setup Environment Variables

Create `.env` file:
```bash
PINECONE_API_KEY=your_pinecone_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key  # Optional
EMBEDDING_MODEL=sentence_transformer  # or "openai"
```

### 3. Setup Google Calendar (Optional)

- Download OAuth credentials as `credentials.json`
- Place in project root directory

### 4. Start Server

```bash
uvicorn server.fastapi_chatbot_server:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access API Documentation

Open your browser to:
- **API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

##  Usage Examples

### Basic Chat

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What robotics events are happening?",
    "user_id": "user123"
  }'
```
