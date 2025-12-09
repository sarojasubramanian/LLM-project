#!/usr/bin/env python3
"""
Streamlit Chatbot with Google Calendar Integration
Features: Instagram RAG search + Automatic event scheduling
"""

import streamlit as st
import sys
import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pinecone_rag import PineconeRAG, SearchResult
    from calendar_creator import InstagramEventCreator
    from goog_oauth import get_service, validate_token, get_token_status
except ImportError as e:
    st.error(f" Failed to import required modules: {e}")
    st.error("Please ensure all src/ modules exist and dependencies are installed")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Instagram RAG + Calendar Bot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_FILE = "data/metadata/scraped_provenance.json"
CHAT_HISTORY_FILE = "data/chat_history_calendar.json"
MAX_CHAT_HISTORY = 100

class EventExtractionHelper:
    """Helper class to extract event information from LLM responses"""
    
    @staticmethod
    def extract_events_from_response(response_text: str, relevant_posts: List[Dict]) -> List[Dict[str, Any]]:
        """Extract structured event data from LLM response and posts (only top 3 results)"""
        events = []
        
        # First, try to extract events directly from the LLM response
        llm_events = EventExtractionHelper._extract_events_from_llm_response(response_text, relevant_posts[:3])
        events.extend(llm_events)
        
        # If no events found in LLM response, fall back to post analysis
        if not events:
            # Only process top 3 most relevant posts for event scheduling
            top_posts = sorted(relevant_posts, key=lambda x: x.get('similarity_score', 0), reverse=True)[:3]
            
            # Process each top relevant post to extract event data
            for post in top_posts:
                try:
                    event_data = EventExtractionHelper._extract_from_post(post, response_text)
                    if event_data:
                        events.append(event_data)
                except Exception as e:
                    st.warning(f" Could not extract event from post {post.get('post_id', 'unknown')}: {e}")
        
        return events
    
    @staticmethod
    def _extract_from_post(post: Dict, llm_response: str) -> Optional[Dict[str, Any]]:
        """Extract event data from a single post"""
        # Get text content
        text_content = []
        
        if post.get('caption'):
            text_content.append(post['caption'])
        if post.get('transcribed_text'):
            text_content.append(post['transcribed_text'])
        
        full_text = '\n'.join(text_content)
        
        # Check if this looks like an event
        if not EventExtractionHelper._is_event_text(full_text):
            return None
        
        # Extract event information
        event_data = {
            'title': EventExtractionHelper._extract_title(full_text, llm_response),
            'description': EventExtractionHelper._build_description(post, llm_response),
            'location': EventExtractionHelper._extract_location(full_text),
            'organizer': post.get('post_id', 'Instagram'),
            'source_url': post.get('post_url', ''),
            'keywords': post.get('keywords', []),
            'confidence': post.get('confidence', 0.0),
            'post_data': post
        }
        
        # Extract date/time
        datetime_info = EventExtractionHelper._extract_datetime(full_text, llm_response)
        event_data.update(datetime_info)
        
        # Validate minimum requirements
        if event_data['title'] and (datetime_info.get('start_date') or datetime_info.get('start_datetime')):
            return event_data
        
        return None
    
    @staticmethod
    def _is_event_text(text: str) -> bool:
        """Check if text contains event indicators"""
        event_indicators = [
            r'\b(event|events|workshop|meeting|audition|celebration|festival|competition)\b',
            r'\b(registration|deadline|ceremony|performance|show|concert)\b',
            r'\b(today|tomorrow|this\s+week|next\s+week|save\s+the\s+date)\b',
            r'\b(join\s+us|come\s+to|attend|participate)\b'
        ]
        
        matches = sum(1 for indicator in event_indicators if re.search(indicator, text, re.IGNORECASE))
        return matches >= 1
    
    @staticmethod
    def _extract_title(text: str, llm_response: str) -> str:
        """Extract event title"""
        # Try to get title from LLM response first
        title_patterns = [
            r'(?i)"([^"]+(?:event|workshop|meeting|audition|celebration|festival)[^"]*)"',
            r'(?i)(.*?(?:event|workshop|meeting|audition|celebration|festival))',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, llm_response)
            if match:
                title = match.group(1).strip()
                if 5 <= len(title) <= 80:
                    return title
        
        # Fallback to text extraction
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if 5 <= len(line) <= 80 and not line.startswith('#'):
                clean_title = re.sub(r'[]+', '', line).strip()
                if clean_title:
                    return clean_title
        
        return "Event from Instagram"
    
    @staticmethod
    def _extract_location(text: str) -> str:
        """Extract location information"""
        location_patterns = [
            r'(?i)(?:at|@|location:?)\s*([A-Za-z0-9\s,.-]{5,50})',
            r'\b([A-Za-z\s]+(?:Hall|Room|Center|Building|Auditorium|Campus))\b',
            r'\b(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd))\b'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = re.sub(r'\s+', ' ', match.group(1).strip())
                if 3 <= len(location) <= 100:
                    return location
        return ""
    
    @staticmethod
    def _extract_datetime(text: str, llm_response: str) -> Dict[str, str]:
        """Extract date and time from text and LLM response"""
        combined_text = f"{text}\n{llm_response}"
        
        # Date patterns
        date_patterns = [
            (r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', ['%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y']),
            (r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4})\b', 
             ['%B %d, %Y', '%B %d %Y']),
            (r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s*\d{4})\b', 
             ['%b %d, %Y', '%b %d %Y']),
            (r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2})\b', 
             ['%B %d']),
        ]
        
        # Time patterns
        time_patterns = [
            (r'\b(\d{1,2}:\d{2}\s*(?:AM|PM))\b', ['%I:%M %p']),
            (r'\b(\d{1,2}\s*(?:AM|PM))\b', ['%I %p']),
            (r'\b(\d{1,2}:\d{2})\b', ['%H:%M']),
        ]
        
        result = {}
        
        # Extract date
        for pattern, formats in date_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                for fmt in formats:
                    try:
                        if '%Y' not in fmt:
                            date_str = f"{date_str} {datetime.now().year}"
                            fmt = f"{fmt} %Y"
                        
                        parsed_date = datetime.strptime(date_str, fmt).date()
                        
                        # Extract time
                        extracted_time = None
                        for time_pattern, time_formats in time_patterns:
                            time_match = re.search(time_pattern, combined_text, re.IGNORECASE)
                            if time_match:
                                time_str = time_match.group(1).upper().replace(' ', ' ')
                                for time_fmt in time_formats:
                                    try:
                                        extracted_time = datetime.strptime(time_str, time_fmt).time()
                                        break
                                    except ValueError:
                                        continue
                                if extracted_time:
                                    break
                        
                        # Combine date and time
                        if extracted_time:
                            event_datetime = datetime.combine(parsed_date, extracted_time)
                            result['start_datetime'] = event_datetime.isoformat()
                        else:
                            result['start_date'] = parsed_date.isoformat()
                        
                        return result
                    except ValueError:
                        continue
        
        return result
    
    @staticmethod
    def _extract_events_from_llm_response(response_text: str, relevant_posts: List[Dict]) -> List[Dict[str, Any]]:
        """Extract events directly from structured LLM response"""
        events = []
        
        # Look for event descriptions in the LLM response
        # Pattern to find event names in quotes or after "called"
        event_name_patterns = [
            r'(?i)(?:event is called|called)\s*["\']([^"\']+)["\']',
            r'(?i)(?:event is called|called)\s*([A-Za-z\s]+?)(?:\s+and|\s+event|\s+happening)',
            r'(?i)["\']([^"\']*(?:event|day|workshop|meeting|audition|celebration|festival|competition)[^"\']*)["\']',
            r'(?i)(?:there is|appears to be).*?([A-Za-z\s]+(?:event|day|workshop|meeting|audition|celebration|festival|competition))',
        ]
        
        # Look for dates and times in the response
        datetime_patterns = [
            # Specific date + time patterns
            r'(?i)(?:on|from)\s+((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s*)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\s*(?:from\s+)?(\d{1,2}(?::\d{2})?\s*(?:AM|PM))\s*(?:to\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM)))?',
            r'(?i)(?:on|from)\s+((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s*)?(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:from\s+)?(\d{1,2}(?::\d{2})?\s*(?:AM|PM))\s*(?:to\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM)))?',
            # Date only patterns
            r'(?i)(?:on|from)\s+((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s*)?(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?',
            # Numeric date patterns
            r'(?i)(?:on|from)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]
        
        # Location patterns
        location_patterns = [
            r'(?i)at\s+the\s+([A-Za-z\s]+(?:Center|Building|Hall|Room|Auditorium|Campus|Library)[^.]*?)(?:\.|,|\n|$)',
            r'(?i)at\s+([A-Za-z0-9\s,]+(?:Center|Building|Hall|Room|Auditorium|Campus|Library))',
            r'(?i)(?:location|venue|at):\s*([A-Za-z0-9\s,.-]+)',
        ]
        
        # Find event names
        event_names = []
        for pattern in event_name_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else (match[1] if len(match) > 1 else '')
                if match and len(match.strip()) > 3:
                    event_names.append(match.strip())
        
        # Find dates and times
        datetime_info = []
        for pattern in datetime_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                datetime_info.append(match)
        
        # Find locations
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, str) and len(match.strip()) > 3:
                    locations.append(match.strip())
        
        # Create events from extracted information
        for i, event_name in enumerate(event_names[:3]):  # Limit to 3 events
            event_data = {
                'title': event_name,
                'description': EventExtractionHelper._build_llm_description(response_text, event_name),
                'location': locations[0] if locations else '',
                'organizer': 'Northeastern University',
                'source_url': relevant_posts[0].get('post_url', '') if relevant_posts else '',
                'keywords': [event_name.lower().replace(' ', '_')],
                'confidence': 0.85,  # High confidence from LLM parsing
                'post_data': relevant_posts[0] if relevant_posts else {}
            }
            
            # Add datetime information
            if datetime_info:
                datetime_result = EventExtractionHelper._parse_llm_datetime(datetime_info[0])
                event_data.update(datetime_result)
            
            # Only add event if it has a valid date/time
            if event_data.get('start_date') or event_data.get('start_datetime'):
                events.append(event_data)
        
        return events
    
    @staticmethod
    def _build_llm_description(response_text: str, event_name: str) -> str:
        """Build description from LLM response"""
        lines = response_text.split('\n')
        relevant_lines = []
        
        for line in lines:
            if event_name.lower() in line.lower() or any(word in line.lower() for word in ['event', 'day', 'workshop']):
                relevant_lines.append(line.strip())
        
        description = '\n'.join(relevant_lines[:3])  # Take first 3 relevant lines
        if len(description) > 500:
            description = description[:500] + "..."
        
        return f" Event details extracted from Instagram analysis:\n{description}\n\n🤖 Auto-detected from LLM response on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    @staticmethod
    def _parse_llm_datetime(datetime_match) -> Dict[str, str]:
        """Parse datetime information from LLM response match"""
        result = {}
        
        if isinstance(datetime_match, tuple):
            # Handle tuple matches from regex groups
            date_part = ''
            time_part = ''
            
            # Extract date components
            for item in datetime_match:
                if isinstance(item, str) and item:
                    if any(month in item.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june',
                                                              'july', 'august', 'september', 'october', 'november', 'december']):
                        date_part = item
                    elif re.search(r'\d{1,2}(?::\d{2})?\s*(?:AM|PM)', item, re.IGNORECASE):
                        time_part = item
                    elif re.search(r'\d{1,2}[/-]\d{1,2}', item):
                        date_part = item
            
            # Try to parse the extracted components
            if date_part:
                try:
                    # Handle various date formats
                    current_year = datetime.now().year
                    
                    # Month + day format
                    month_day_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})'
                    month_day_match = re.search(month_day_pattern, date_part, re.IGNORECASE)
                    
                    if month_day_match:
                        month_name = month_day_match.group(1)
                        day = int(month_day_match.group(2))
                        
                        # Parse the date
                        date_str = f"{month_name} {day}, {current_year}"
                        parsed_date = datetime.strptime(date_str, '%B %d, %Y').date()
                        
                        # Handle time if present
                        if time_part:
                            time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM)', time_part, re.IGNORECASE)
                            if time_match:
                                hour = int(time_match.group(1))
                                minute = int(time_match.group(2)) if time_match.group(2) else 0
                                period = time_match.group(3).upper()
                                
                                if period == 'PM' and hour != 12:
                                    hour += 12
                                elif period == 'AM' and hour == 12:
                                    hour = 0
                                
                                event_time = datetime.min.time().replace(hour=hour, minute=minute)
                                event_datetime = datetime.combine(parsed_date, event_time)
                                result['start_datetime'] = event_datetime.isoformat()
                            else:
                                result['start_date'] = parsed_date.isoformat()
                        else:
                            result['start_date'] = parsed_date.isoformat()
                    
                    # Numeric date format
                    elif '/' in date_part or '-' in date_part:
                        for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%m/%d', '%m-%d']:
                            try:
                                if '%Y' not in fmt:
                                    date_part_with_year = f"{date_part}/{current_year}"
                                    fmt_with_year = f"{fmt}/%Y"
                                    parsed_date = datetime.strptime(date_part_with_year, fmt_with_year).date()
                                else:
                                    parsed_date = datetime.strptime(date_part, fmt).date()
                                result['start_date'] = parsed_date.isoformat()
                                break
                            except ValueError:
                                continue
                
                except Exception as e:
                    # If parsing fails, set a default date
                    result['start_date'] = (datetime.now() + timedelta(days=1)).date().isoformat()
        
        return result

    @staticmethod
    def add_source_links_to_response(response_text: str, relevant_posts: List[Dict]) -> str:
        """Add Instagram post links to the LLM response"""
        if not relevant_posts:
            return response_text
        
        # Get top 3 posts with valid URLs
        top_posts_with_urls = []
        for post in relevant_posts[:3]:
            post_url = post.get('post_url', '')
            if post_url and post_url != 'unknown' and post_url.startswith('http'):
                top_posts_with_urls.append({
                    'url': post_url,
                    'timestamp': post.get('timestamp', 'Unknown date'),
                    'similarity': post.get('similarity_score', 0)
                })
        
        if not top_posts_with_urls:
            return response_text
        
        # Add source links section to response
        links_section = "\n\n---\n**📱 Original Instagram Posts:**\n"
        for i, post_info in enumerate(top_posts_with_urls, 1):
            links_section += f"{i}. [View Post from {post_info['timestamp']}]({post_info['url']}) " \
                           f"(Relevance: {post_info['similarity']:.1%})\n"
        
        return response_text + links_section
    
    @staticmethod
    def _build_description(post: Dict, llm_response: str) -> str:
        """Build comprehensive event description"""
        parts = []
        
        # Add LLM context
        parts.append(" Event Details from Instagram Analysis:")
        parts.append(llm_response[:300] + "..." if len(llm_response) > 300 else llm_response)
        
        # Add original post content
        if post.get('caption'):
            parts.append(f"\n📱 Original Caption: {post['caption'][:200]}...")
        
        if post.get('transcribed_text'):
            parts.append(f"\nImage Text: {post['transcribed_text'][:200]}...")
        
        # Add metadata
        if post.get('keywords'):
            parts.append(f"\n Tags: {', '.join(post['keywords'][:5])}")
        
        if post.get('source_url'):
            parts.append(f"\n Source: {post['source_url']}")
        
        parts.append(f"\nAuto-detected from Instagram on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        return '\n'.join(parts)

class CalendarChatbot:
    """Main chatbot class with calendar integration"""
    
    def __init__(self):
        self.rag_system = None
        self.calendar_creator = None
        self.chat_history = []
        self.detected_events = []
        self.setup_logging()
        self.load_chat_history()
    
    def setup_logging(self):
        """Setup logging"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/calendar_chatbot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_chat_history(self):
        """Load chat history"""
        try:
            if os.path.exists(CHAT_HISTORY_FILE):
                with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                self.logger.info(f"📂 Loaded {len(self.chat_history)} chat messages")
            else:
                self.chat_history = []
        except Exception as e:
            self.logger.error(f" Error loading chat history: {e}")
            self.chat_history = []
    
    def save_chat_history(self):
        """Save chat history"""
        try:
            os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)
            if len(self.chat_history) > MAX_CHAT_HISTORY:
                self.chat_history = self.chat_history[-MAX_CHAT_HISTORY:]
            
            with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f" Error saving chat history: {e}")
    
    def initialize_systems(self) -> bool:
        """Initialize RAG and Calendar systems"""
        try:
            # Initialize RAG system
            if not self.rag_system:
                with st.spinner("🔧 Initializing RAG system..."):
                    self.rag_system = PineconeRAG(DATA_FILE, index_name="acm-hackathon")
                    self.rag_system.load_posts()
                st.success(" RAG system ready")
            
            # Initialize Calendar creator
            if not self.calendar_creator:
                self.calendar_creator = InstagramEventCreator(timezone="America/New_York")
                st.success(" Calendar system ready")
            
            return True
            
        except Exception as e:
            st.error(f" System initialization failed: {e}")
            self.logger.error(f"Initialization error: {e}")
            return False
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to chat history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.chat_history.append(message)
        self.save_chat_history()
    
    def search_and_extract_events(self, query: str) -> tuple[SearchResult, List[Dict]]:
        """Search for information and extract events"""
        # Perform RAG search
        result = self.rag_system.search(query, top_k=5)
        
        # Extract events from the search results
        events = EventExtractionHelper.extract_events_from_response(
            result.answer, 
            result.relevant_posts
        )
        
        return result, events
    
    def create_calendar_event(self, event_data: Dict) -> Dict:
        """Create a Google Calendar event"""
        try:
            # Authenticate if needed
            if not self.calendar_creator.authenticate():
                return {'success': False, 'error': 'Calendar authentication failed'}
            
            # Create the event
            result = self.calendar_creator.create_event(event_data)
            
            if result['success']:
                self.logger.info(f" Created calendar event: {event_data['title']}")
            else:
                self.logger.error(f" Failed to create event: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            error_msg = f"Calendar event creation failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def render_sidebar(self):
        """Render sidebar with system info"""
        with st.sidebar:
            
            # System Status
            st.subheader(" System Status")
            
            # RAG System Status
            if self.rag_system:
                try:
                    stats = self.rag_system.get_stats()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Posts", stats.get('local_posts_loaded', 0))
                        st.metric("Vectors", stats.get('pinecone_total_vectors', 0))
                    with col2:
                        avg_conf = stats.get('average_confidence', 0)
                        st.metric("Quality", f"{avg_conf:.1%}")
                        st.metric("Keywords", stats.get('unique_keywords', 0))
                    
                    st.success(" RAG System Online")
                except Exception as e:
                    st.error(f" RAG Status Error: {e}")
            else:
                st.warning(" RAG System Not Ready")
            
            # Calendar System Status
            st.subheader(" Calendar Status")
            token_status = get_token_status()
            
            if token_status['credentials_exists']:
                st.success(" Credentials Available")
            else:
                st.error(" Missing credentials.json")
            
            if token_status['token_exists'] and token_status['token_valid']:
                st.success(" Calendar Access Ready")
            elif token_status['token_exists'] and token_status['token_expired']:
                st.warning(" Token Expired (will refresh)")
            else:
                st.error(" No Calendar Access")
            
            st.divider()
            
            # Chat Management
            st.subheader(" Chat Management")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", len(self.chat_history))
            with col2:
                st.metric("Events Found", len(self.detected_events))
            
            if st.button("🗑️ Clear Chat"):
                self.chat_history = []
                self.detected_events = []
                self.save_chat_history()
                st.rerun()
            
            st.divider()
            
            # Quick Questions
            st.subheader(" Try These Questions")
            sample_questions = [
                
                "Tell me about music auditions",
                "Show me celebration events",
                "Any robotics event coming up ?",
                "Are there any workshops coming up?",
                "What cultural events are planned?",
                "What events are happening this week?"
            ]
            
            for question in sample_questions:
                if st.button(f" {question}", key=f"q_{hash(question)}"):
                    st.session_state.selected_question = question
            
            # Help
            with st.expander(" How It Works"):
                st.markdown("""
                **Features:**
                -  Search Instagram posts for events
                -  Automatically detect event details from top results
                - One-click calendar scheduling
                - Direct links to original Instagram posts
                - Persistent chat memory
                
                **Event Detection:**
                - Only creates schedulable events from top 3 most relevant posts
                - Shows original Instagram post links in responses
                - Extracts dates, times, locations automatically
                - Includes source attribution in calendar events
                
                **Usage:**
                1. Ask about events or activities
                2. Review AI findings with source links
                3. Click " Schedule Event" for detected events
                4. Events appear in your Google Calendar with full details!
                
                **Examples:**
                - "What events is ACM hosting?"
                - "Tell me about upcoming workshops"
                - "Any dance auditions this month?"
                """)
            
            # Recent events summary
            if self.detected_events:
                st.subheader(" Recent Events Scheduled")
                for event in self.detected_events[-3:]:  # Show last 3
                    st.write(f" {event.get('title', 'Event')}")
                if st.button("🗑️ Clear Event History"):
                    self.detected_events = []
                    st.rerun()
    
    def render_event_buttons(self, events: List[Dict]):
        """Render scheduling buttons for detected events"""
        if not events:
            return
        
        st.markdown("###  Detected Events - Click to Schedule")
        st.info(f" Found {len(events)} schedulable event(s) from top search results")
        
        for i, event in enumerate(events):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Show event details
                    event_date = event.get('start_date', event.get('start_datetime', 'Date TBD'))
                    if event_date != 'Date TBD' and 'T' in event_date:
                        # Format datetime for better display
                        try:
                            dt = datetime.fromisoformat(event_date)
                            event_date = dt.strftime('%B %d, %Y at %I:%M %p')
                        except:
                            pass
                    
                    st.markdown(f"""
                    ** {event['title']}**  
                     {event_date}  
                     {event.get('location', 'Location TBD')}  
                     {event.get('organizer', 'Unknown organizer')}
                    """)
                    
                    # Show source post link if available
                    source_url = event.get('source_url', '')
                    if source_url and source_url != 'unknown' and source_url.startswith('http'):
                        st.markdown(f"📱 [View Original Instagram Post]({source_url})")
                    
                    # Show confidence
                    confidence = event.get('confidence', 0.0)
                    if confidence > 0:
                        st.markdown(f" Detection confidence: {confidence:.1%}")
                
                with col2:
                    button_key = f"schedule_{i}_{hash(str(event))}"
                    if st.button(f" Schedule Event", key=button_key, type="primary"):
                        with st.spinner(f"Creating calendar event: {event['title']}..."):
                            result = self.create_calendar_event(event)
                        
                        if result['success']:
                            st.success(f" {result['message']}")
                            st.markdown(f"🔗 [View in Google Calendar]({result['event_link']})")
                            
                            # Add to detected events for tracking
                            if event not in self.detected_events:
                                self.detected_events.append(event)
                        else:
                            st.error(f" {result.get('message', 'Failed to create event')}")
                
                st.divider()
    
    def render_main_chat(self):
        """Render main chat interface"""
        st.title(" EventGram Bot")
        st.markdown("Ask me about events and I'll help you schedule them in Google Calendar!")
        
        # Initialize systems
        if not self.initialize_systems():
            st.error("Cannot proceed without proper system initialization")
            return
        
        # Display chat history
        for message in self.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show event buttons for assistant messages with events
                if message["role"] == "assistant" and message.get("metadata", {}).get("events"):
                    events = message["metadata"]["events"]
                    self.render_event_buttons(events)
        
        # Handle input
        if hasattr(st.session_state, 'selected_question'):
            user_input = st.session_state.selected_question
            delattr(st.session_state, 'selected_question')
        else:
            user_input = st.chat_input(" Ask me about events and I'll help schedule them...")
        
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Add to history
            self.add_message("user", user_input)
            
            # Generate response with event detection
            with st.chat_message("assistant"):
                with st.spinner(" Searching for events..."):
                    try:
                        # Search and extract events
                        search_result, detected_events = self.search_and_extract_events(user_input)
                        
                        # Display the response with source links
                        response_with_links = EventExtractionHelper.add_source_links_to_response(
                            search_result.answer, search_result.relevant_posts
                        )
                        st.markdown(response_with_links)
                        
                        # Show detailed source information
                        if search_result.relevant_posts:
                            with st.expander(f"📚 Sources ({len(search_result.relevant_posts)} posts)"):
                                for i, post in enumerate(search_result.relevant_posts[:5], 1):
                                    post_url = post.get('post_url', '')
                                    if post_url and post_url != 'unknown':
                                        st.markdown(f"{i}. Post from {post.get('timestamp', 'Unknown')} "
                                                 f"(Similarity: {post.get('similarity_score', 0):.1%}) "
                                                 f"- [🔗 View Original]({post_url})")
                                    else:
                                        st.write(f"{i}. Post from {post.get('timestamp', 'Unknown')} "
                                               f"(Similarity: {post.get('similarity_score', 0):.1%}) "
                                               f"- Original link not available")
                        
                        # Show confidence
                        confidence = search_result.confidence_score
                        confidence_emoji = "conf" if confidence > 0.7 else "avg" if confidence > 0.5 else "noconf"
                        st.info(f"{confidence_emoji} Answer confidence: {confidence:.1%}")
                        
                        # Render event scheduling buttons
                        if detected_events:
                            self.render_event_buttons(detected_events)
                        else:
                            st.info(" No specific events detected for scheduling in the top search results")
                        
                        # Save to history with events metadata
                        self.add_message("assistant", search_result.answer, {
                            "events": detected_events,
                            "confidence": confidence,
                            "sources": len(search_result.relevant_posts)
                        })
                        
                    except Exception as e:
                        error_msg = f" Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        self.logger.error(f"Chat error: {e}")
                        self.add_message("assistant", error_msg, {"error": str(e)})
    
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
    
    def run(self):
        """Main application runner"""
        # Check requirements
        issues = self.check_requirements()
        
        if issues:
            st.error(" Setup Issues Found:")
            for issue in issues:
                st.error(issue)
            
            st.info(" Setup Instructions:")
            st.markdown("""
            1. **Install dependencies**: `uv sync`
            2. **Prepare Instagram data**:
               - Run: `python pilots/APIScraper.py`
               - Run: `python pilots/image_transcibe.py --all`
            3. **Setup environment variables** (`.env` file):
               ```
               PINECONE_API_KEY=your_pinecone_key
               ANTHROPIC_API_KEY=your_anthropic_key
               ```
            4. **Setup Google Calendar**:
               - Download OAuth credentials as `credentials.json`
               - Place in project root directory
            5. **Restart**: `python streamlit_calendar_chatbot.py`
            """)
            return
        
        # Render the app
        self.render_sidebar()
        self.render_main_chat()

def main():
    """Application entry point"""
    try:
        chatbot = CalendarChatbot()
        chatbot.run()
    except Exception as e:
        st.error(f" Critical error: {e}")
        st.info("Check logs for details: logs/calendar_chatbot.log")

if __name__ == "__main__":
    main()