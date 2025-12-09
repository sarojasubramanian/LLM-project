#!/usr/bin/env python3
"""
Google Calendar Event Creator for Instagram Data
Automatically creates calendar events from Instagram posts with event information
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Import our centralized Google OAuth module
try:
    from .goog_oauth import get_service
except ImportError:
    # Fallback for when running as script directly
    from goog_oauth import get_service

class InstagramEventCreator:
    """Creates Google Calendar events from Instagram post data"""
    
    def __init__(self, timezone: str = "America/New_York"):
        self.timezone = timezone
        self.service = None
        
    def authenticate(self):
        """Authenticate with Google Calendar API"""
        try:
            self.service = get_service()
            print(" Successfully authenticated with Google Calendar")
            return True
        except Exception as e:
            print(f" Failed to authenticate: {e}")
            return False
    
    def create_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a single calendar event from structured event data.
        
        Args:
            event_data: Dict containing event information
            
        Returns:
            Dict with creation result and event details
        """
        if not self.service:
            return {'success': False, 'error': 'Not authenticated'}
        
        try:
            # Prepare the calendar event
            calendar_event = {
                'summary': event_data.get('title', 'Instagram Event'),
                'description': self._build_description(event_data),
                'location': event_data.get('location', ''),
            }
            
            # Handle start time
            if 'start_datetime' in event_data:
                calendar_event['start'] = {
                    'dateTime': event_data['start_datetime'],
                    'timeZone': self.timezone,
                }
            elif 'start_date' in event_data:
                calendar_event['start'] = {
                    'date': event_data['start_date'],
                }
            else:
                raise ValueError("Event must have either start_datetime or start_date")
            
            # Handle end time
            if 'end_datetime' in event_data:
                calendar_event['end'] = {
                    'dateTime': event_data['end_datetime'],
                    'timeZone': self.timezone,
                }
            elif 'end_date' in event_data:
                calendar_event['end'] = {
                    'date': event_data['end_date'],
                }
            else:
                # Default to 1 hour duration for datetime events
                if 'start_datetime' in event_data:
                    start_dt = datetime.fromisoformat(event_data['start_datetime'].replace('Z', '+00:00'))
                    end_dt = start_dt + timedelta(hours=1)
                    calendar_event['end'] = {
                        'dateTime': end_dt.isoformat(),
                        'timeZone': self.timezone,
                    }
                else:
                    # Same day for date-only events
                    calendar_event['end'] = calendar_event['start'].copy()
            
            # Create the event
            created_event = self.service.events().insert(
                calendarId=event_data.get('calendar_id', 'primary'), 
                body=calendar_event
            ).execute()
            
            return {
                'success': True,
                'event_id': created_event.get('id'),
                'event_link': created_event.get('htmlLink'),
                'summary': created_event.get('summary'),
                'start_time': created_event.get('start', {}).get('dateTime', created_event.get('start', {}).get('date')),
                'message': f" Created event: {created_event.get('summary')}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f" Failed to create event: {str(e)}"
            }
    
    def _build_description(self, event_data: Dict[str, Any]) -> str:
        """Build a comprehensive event description"""
        description_parts = []
        
        # Add original description
        if event_data.get('description'):
            description_parts.append(event_data['description'])
        
        # Add organizer
        if event_data.get('organizer'):
            description_parts.append(f"\n Organizer: {event_data['organizer']}")
        
        # Add keywords if available
        if event_data.get('keywords'):
            keywords = ', '.join(event_data['keywords'])
            description_parts.append(f"\n🏷️ Tags: {keywords}")
        
        # Add source information
        if event_data.get('source_url'):
            description_parts.append(f"\n📱 Source: {event_data['source_url']}")
        
        # Add confidence score
        if event_data.get('confidence'):
            confidence_pct = event_data['confidence'] * 100
            description_parts.append(f"\n AI Confidence: {confidence_pct:.1f}%")
        
        # Add creation timestamp
        description_parts.append(f"\n🤖 Auto-created from Instagram data on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return ''.join(description_parts)
    
    def process_instagram_data(self, data_source: str, dry_run: bool = True) -> Dict[str, Any]:
        """
        Process Instagram data and create calendar events.
        
        Args:
            data_source: Path to Instagram data file (scraped_provenance.json)
            dry_run: If True, only show what would be created without actually creating
            
        Returns:
            Dict with processing results and statistics
        """
        # Load Instagram data
        try:
            with open(data_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
                posts = data.get('posts', [])
        except Exception as e:
            return {'success': False, 'error': f"Failed to load Instagram data: {e}"}
        
        # Authenticate if not in dry run mode
        if not dry_run and not self.service:
            if not self.authenticate():
                return {'success': False, 'error': 'Authentication failed'}
        
        results = {
            'total_posts': len(posts),
            'events_detected': 0,
            'events_created': 0,
            'events_failed': 0,
            'created_events': [],
            'failed_events': [],
            'skipped_posts': []
        }
        
        print(f" Processing {len(posts)} Instagram posts for events...")
        
        # Process each post
        for i, post in enumerate(posts, 1):
            try:
                print(f"\n Processing post {i}/{len(posts)} (ID: {post.get('post_id', 'unknown')})")
                
                # Extract event data from post
                event_data = self.extract_event_from_post(post)
                
                if not event_data:
                    results['skipped_posts'].append({
                        'post_id': post.get('post_id', 'unknown'),
                        'reason': 'No event data detected'
                    })
                    print("    No event detected, skipping...")
                    continue
                
                results['events_detected'] += 1
                
                # Display event info
                print(f"    Event detected: {event_data.get('title', 'Untitled Event')}")
                print(f"    Date: {event_data.get('start_date', event_data.get('start_datetime', 'Unknown'))}")
                print(f"    Location: {event_data.get('location', 'No location')}")
                print(f"    Organizer: {event_data.get('organizer', 'Unknown')}")
                
                if dry_run:
                    print("    DRY RUN - Event would be created")
                    continue
                
                # Create the calendar event
                create_result = self.create_event(event_data)
                
                if create_result['success']:
                    results['events_created'] += 1
                    results['created_events'].append(create_result)
                    print(f"   {create_result['message']}")
                    print(f"    Link: {create_result.get('event_link', 'N/A')}")
                else:
                    results['events_failed'] += 1
                    results['failed_events'].append(create_result)
                    print(f"   {create_result['message']}")
                    
            except Exception as e:
                results['events_failed'] += 1
                error_info = {
                    'post_id': post.get('post_id', 'unknown'),
                    'error': str(e),
                    'message': f" Error processing post: {str(e)}"
                }
                results['failed_events'].append(error_info)
                print(f"   {error_info['message']}")
        
        # Print summary
        self._print_summary(results, dry_run)
        return results
    
    def extract_event_from_post(self, post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract event information from an Instagram post.
        
        Args:
            post: Instagram post data from scraped_provenance.json
            
        Returns:
            Dict with extracted event data or None if no event detected
        """
        # Get all text content from post
        text_content = []
        
        # From post caption
        post_data = post.get('post_data', {})
        if post_data.get('caption'):
            text_content.append(post_data['caption'])
        
        # From transcription
        transcription = post.get('transcription', {})
        if transcription.get('status') == 'successful':
            extracted_text = transcription.get('extracted_text', {})
            if extracted_text.get('main_text'):
                text_content.append(extracted_text['main_text'])
        
        # Combine all text
        full_text = '\n'.join(text_content).strip()
        if not full_text:
            return None
        
        # Check if this looks like an event
        if not self._is_event_post(full_text):
            return None
        
        # Extract event details
        event_data = {
            'title': self._extract_title(full_text),
            'description': full_text[:500],  # Truncate for calendar
            'location': self._extract_location(full_text),
            'organizer': self._extract_organizer(post, full_text),
            'source_url': post_data.get('post_url', ''),
            'keywords': transcription.get('extracted_text', {}).get('keywords', []),
            'confidence': transcription.get('extracted_text', {}).get('confidence', 0.0),
        }
        
        # Extract date and time
        datetime_info = self._extract_datetime(full_text)
        event_data.update(datetime_info)
        
        # Validate we have minimum required info
        if not event_data['title']:
            return None
        
        if not (datetime_info.get('start_date') or datetime_info.get('start_datetime')):
            return None
        
        return event_data
    
    def _is_event_post(self, text: str) -> bool:
        """Determine if text contains event information"""
        event_indicators = [
            # Event types
            r'\b(event|events)\b', r'\b(meeting|meetings)\b', r'\b(workshop|workshops)\b',
            r'\b(seminar|seminars)\b', r'\b(conference|conferences)\b', r'\b(symposium)\b',
            
            # Performance/entertainment
            r'\b(audition|auditions)\b', r'\b(performance|performances)\b', r'\b(show|shows)\b',
            r'\b(concert|concerts)\b', r'\b(festival|festivals)\b', r'\b(celebration|celebrations)\b',
            
            # Academic/professional
            r'\b(registration|deadline)\b', r'\b(ceremony|ceremonies)\b',
            r'\b(competition|competitions)\b', r'\b(tournament|tournaments)\b',
            r'\b(class|classes)\b', r'\b(session|sessions)\b', r'\b(training)\b',
            
            # Time indicators
            r'\b(today|tomorrow|tonight)\b', r'\b(this\s+(week|month|weekend))\b',
            r'\b(next\s+(week|month|weekend))\b', r'\b(save\s+the\s+date)\b',
        ]
        
        # Check for multiple indicators for higher confidence
        matches = sum(1 for indicator in event_indicators if re.search(indicator, text, re.IGNORECASE))
        return matches >= 2
    
    def _extract_title(self, text: str) -> str:
        """Extract event title from text"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Try first meaningful line as title
        for line in lines:
            if 5 <= len(line) <= 100:  # Reasonable title length
                # Clean up emojis and special characters
                title = re.sub(r'[]+', '', line).strip()
                title = re.sub(r'[!]{3,}', '!', title)  # Reduce excessive punctuation
                if title and not title.startswith('#'):  # Skip hashtag-only lines
                    return title
        
        # Fallback: look for specific event patterns
        title_patterns = [
            r'(\b[A-Z][A-Za-z\s]{5,50}(?:Event|Meeting|Workshop|Audition|Show|Festival)\b)',
            r'(\b(?:Event|Meeting|Workshop|Audition|Show|Festival)[\w\s]{5,50})',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Instagram Event"
    
    def _extract_location(self, text: str) -> str:
        """Extract location from text"""
        location_patterns = [
            # Explicit location markers
            r'(?:at|@|location:?)\s*([A-Za-z0-9\s,.-]{5,50})',
            # Building/venue types
            r'\b([A-Za-z\s]+(?:Hall|Room|Center|Centre|Building|Auditorium|Theater|Theatre|Stadium|Arena|Library))\b',
            # Address patterns
            r'\b(\d+\s+[A-Za-z\s]+(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|Boulevard|Blvd\.?))\b',
            # Campus locations
            r'\b([A-Z][A-Za-z\s]+(?:Campus|University|College))\b',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Clean up and validate
                location = re.sub(r'\s+', ' ', location)
                if 5 <= len(location) <= 100:
                    return location
        
        return ""
    
    def _extract_organizer(self, post: Dict[str, Any], text: str) -> str:
        """Extract organizer information"""
        # Try post metadata first
        post_data = post.get('post_data', {})
        username = post_data.get('username', '')
        if username and username != 'unknown':
            return username
        
        # Look for organizer patterns in text
        org_patterns = [
            r'(?:by|hosted\s+by|organized\s+by|presented\s+by):?\s*([A-Za-z\s&]{3,40})',
            r'\b([A-Za-z\s&]+(?:Club|Society|Organization|Committee|Team|Association|Group))\b',
            r'(?:contact|info):?\s*([A-Za-z\s]{3,30})',
        ]
        
        for pattern in org_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                organizer = match.group(1).strip()
                if 3 <= len(organizer) <= 50:
                    return organizer
        
        return ""
    
    def _extract_datetime(self, text: str) -> Dict[str, str]:
        """Extract date and time information from text"""
        result = {}
        
        # Date patterns (more comprehensive)
        date_patterns = [
            # MM/DD/YYYY, MM-DD-YYYY, MM.DD.YYYY
            (r'\b(\d{1,2}[/-\.]\d{1,2}[/-\.]\d{2,4})\b', ['%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y', '%m/%d/%y', '%m-%d-%y']),
            # Month DD, YYYY
            (r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', ['%B %d, %Y', '%B %d %Y']),
            # DD Month YYYY
            (r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b', ['%d %B %Y']),
            # Month DD (current year)
            (r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2})\b', ['%B %d']),
            # Abbreviated months
            (r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})\b', ['%b %d, %Y', '%b %d %Y']),
        ]
        
        # Time patterns
        time_patterns = [
            (r'\b(\d{1,2}:\d{2}\s*(?:AM|PM))\b', ['%I:%M %p']),
            (r'\b(\d{1,2}\s*(?:AM|PM))\b', ['%I %p']),
            (r'\b(\d{1,2}:\d{2})\b', ['%H:%M']),  # 24-hour format
        ]
        
        # Extract date
        extracted_date = None
        for pattern, formats in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                for fmt in formats:
                    try:
                        if '%Y' not in fmt:  # Add current year for formats without year
                            date_str_with_year = f"{date_str} {datetime.now().year}"
                            fmt_with_year = f"{fmt} %Y"
                            extracted_date = datetime.strptime(date_str_with_year, fmt_with_year).date()
                        else:
                            extracted_date = datetime.strptime(date_str, fmt).date()
                        break
                    except ValueError:
                        continue
                if extracted_date:
                    break
        
        # Extract time
        extracted_time = None
        for pattern, formats in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time_str = match.group(1)
                # Normalize time string
                time_str = re.sub(r'\s+', ' ', time_str).strip().upper()
                
                for fmt in formats:
                    try:
                        extracted_time = datetime.strptime(time_str, fmt).time()
                        break
                    except ValueError:
                        continue
                if extracted_time:
                    break
        
        # Combine date and time
        if extracted_date:
            if extracted_time:
                # Create full datetime
                event_datetime = datetime.combine(extracted_date, extracted_time)
                result['start_datetime'] = event_datetime.isoformat()
            else:
                # Date-only event
                result['start_date'] = extracted_date.isoformat()
        
        return result
    
    def _print_summary(self, results: Dict[str, Any], dry_run: bool):
        """Print processing summary"""
        print(f"\n{'='*60}")
        print(f" EVENT CREATION SUMMARY")
        print(f"{'='*60}")
        print(f" Total posts processed: {results['total_posts']}")
        print(f" Events detected: {results['events_detected']}")
        
        if not dry_run:
            print(f" Events created successfully: {results['events_created']}")
            print(f" Events failed to create: {results['events_failed']}")
            
            if results['created_events']:
                print(f"\n🎉 Successfully Created Events:")
                for i, event in enumerate(results['created_events'], 1):
                    print(f"   {i}. {event['summary']}")
                    print(f"       {event.get('start_time', 'Unknown time')}")
                    print(f"       {event.get('event_link', 'No link')}")
            
            if results['failed_events']:
                print(f"\n Failed Events:")
                for i, event in enumerate(results['failed_events'], 1):
                    print(f"   {i}. {event.get('message', 'Unknown error')}")
        else:
            print(f" DRY RUN MODE - No events were actually created")
        
        print(f" Posts skipped (no events): {len(results['skipped_posts'])}")
        print(f"{'='*60}")


def main():
    """Command-line interface for the calendar creator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Google Calendar events from Instagram data")
    parser.add_argument('data_file', help='Path to Instagram data file (scraped_provenance.json)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be created without actually creating')
    parser.add_argument('--timezone', default='America/New_York', help='Timezone for events (default: America/New_York)')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f" Data file not found: {args.data_file}")
        print("Please ensure the Instagram data file exists.")
        return
    
    # Create event creator
    creator = InstagramEventCreator(timezone=args.timezone)
    
    print(f" Instagram Calendar Event Creator")
    print(f"{'='*60}")
    print(f" Data file: {args.data_file}")
    print(f" Timezone: {args.timezone}")
    print(f" Mode: {'DRY RUN' if args.dry_run else 'LIVE CREATION'}")
    
    if not args.dry_run:
        print(f"\n  LIVE MODE: Events will be created in your Google Calendar!")
        response = input("Continue? (y/N): ").lower().strip()
        if response != 'y':
            print("Cancelled.")
            return
    
    # Process the Instagram data
    results = creator.process_instagram_data(args.data_file, dry_run=args.dry_run)
    
    if not results.get('success', True):
        print(f" Failed: {results.get('error', 'Unknown error')}")
        return
    
    # Suggest next steps
    if args.dry_run and results['events_detected'] > 0:
        print(f"\n Next steps:")
        print(f"   1. Review the detected events above")
        print(f"   2. Run without --dry-run to create events:")
        print(f"      python src/calendar_creator.py {args.data_file}")


if __name__ == "__main__":
    main()