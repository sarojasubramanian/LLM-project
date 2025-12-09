# Instagram RAG + Google Calendar Chatbot 

A powerful Streamlit chatbot that combines Instagram RAG search with **automatic Google Calendar event scheduling**. Ask about events from Instagram posts and schedule them directly to your Google Calendar with one click!

##  Key Features

###  Intelligent Event Detection
- **Smart Search**: Uses Pinecone RAG to find relevant Instagram posts
- **Event Recognition**: Automatically detects events, workshops, meetings, auditions, etc.
- **Context Understanding**: Extracts dates, times, locations, and organizers from posts
- **Multi-source Analysis**: Combines Instagram captions, image text, and AI analysis

###  One-Click Calendar Scheduling  
- **Automatic Event Creation**: Extract event details and create Google Calendar events
- **Smart Date/Time Parsing**: Handles various date and time formats
- **Complete Event Info**: Includes titles, descriptions, locations, and source links
- **Google Calendar Integration**: Direct OAuth integration with your Google account

###  Enhanced Chat Experience
- **Persistent Memory**: Remembers conversation history across sessions
- **Interactive Buttons**: Click "Schedule Event" buttons right in the chat
- **Source Attribution**: Shows which Instagram posts were used for responses
- **Confidence Scoring**: Indicates reliability of event detection

##  How It Works

1. **Ask About Events**: "What dance auditions are happening this week?"
2. **AI Searches**: RAG system finds relevant Instagram posts
3. **Events Detected**: AI extracts structured event information
4. **Schedule Options**: Click " Schedule" buttons for any detected events
5. **Calendar Updated**: Events appear in your Google Calendar automatically!

##  Quick Start

### 1. Prerequisites Setup

```bash
# Install dependencies
uv sync

# Prepare Instagram data
python pilots/APIScraper.py
python pilots/image_transcibe.py --all

# Setup environment variables (.env file)
PINECONE_API_KEY=your_pinecone_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 2. Google Calendar Setup

1. **Get Google Cloud Credentials**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable Google Calendar API
   - Create OAuth 2.0 credentials (Desktop application)
   - Download as `credentials.json`

2. **Place Credentials**:
   ```bash
   # Move the downloaded file to project root
   mv ~/Downloads/credentials.json ./credentials.json
   ```

### 3. Launch the Chatbot

```bash
# manual launch
streamlit run streamlit_calendar_chatbot.py --server.port 8502
```

### 4. First Use Authorization

1. Open http://localhost:8502
2. First time: Google will ask for calendar permissions
3. Grant access to create events
4. You're ready to go! 🎉

##  Usage Examples

### Event Discovery
```
You: "What events is NU Sanskriti hosting?"
Bot: "NU Sanskriti is hosting several cultural events including..."
     [ Schedule] [ Schedule] [ Schedule]
```

### Specific Searches  
```
You: "Any dance auditions this month?"
Bot: "Yes! There are dance auditions for..."
     [ Schedule Bollywood Dance Audition]
```

### Follow-up Questions
```
You: "Tell me more about that workshop"
Bot: "The workshop I mentioned earlier..." [remembers context]
     [ Schedule Workshop Event]
```

### Automatic Event Scheduling
- Click any " Schedule" button
- Event automatically appears in Google Calendar
- Includes all details: date, time, location, description
- Links back to original Instagram post

##  Configuration

### Event Detection Settings
The chatbot automatically detects events by looking for:
- **Event Keywords**: workshop, audition, meeting, celebration, festival
- **Time Indicators**: today, tomorrow, next week, save the date
- **Action Words**: join us, register, attend, participate

### Calendar Event Creation
Automatically extracted information includes:
- **Title**: Event name from post content
- **Date/Time**: Parsed from various formats (MM/DD/YYYY, "January 15", etc.)
- **Location**: Building names, addresses, campus locations
- **Description**: Combines original post + AI analysis
- **Source Link**: Link back to original Instagram post

### Chat Memory
- Stores up to 100 messages by default
- Remembers context for follow-up questions
- Persists between browser sessions
- Export chat history as JSON
