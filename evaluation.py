"""
EventGram Evaluation Suite
Comprehensive evaluation metrics for event extraction and RAG system performance
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from pathlib import Path

# For embeddings and RAG evaluation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EventGramEvaluator:
    """Main evaluation class for EventGram system"""
    
    def __init__(self, events_file: str, instagram_data_file: str = None):
        """
        Initialize evaluator with data files
        
        Args:
            events_file: Path to scraped events JSON file
            instagram_data_file: Optional path to Instagram extraction results
        """
        self.events = self.load_events(events_file)
        self.instagram_data = self.load_instagram_data(instagram_data_file) if instagram_data_file else None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.evaluation_results = {}
        
    def load_events(self, filepath: str) -> List[Dict]:
        """Load events from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading events: {e}")
            return []
    
    def load_instagram_data(self, filepath: str) -> Dict:
        """Load Instagram extraction results"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    # ================== 1. DATA QUALITY EVALUATION ==================
    
    def evaluate_data_quality(self) -> Dict[str, Any]:
        """Evaluate the quality and completeness of extracted event data"""
        
        quality_metrics = {
            'total_events': len(self.events),
            'completeness_scores': [],
            'missing_fields': defaultdict(int),
            'data_issues': []
        }
        
        required_fields = ['title', 'start', 'venue', 'description']
        optional_fields = ['end', 'image', 'url']
        
        for event in self.events:
            # Calculate completeness score
            filled_required = sum(1 for field in required_fields 
                                if event.get(field) and event[field] != "null")
            completeness = filled_required / len(required_fields)
            quality_metrics['completeness_scores'].append(completeness)
            
            # Track missing fields
            for field in required_fields + optional_fields:
                if not event.get(field) or event[field] == "null":
                    quality_metrics['missing_fields'][field] += 1
            
            # Check for data quality issues
            if event.get('title'):
                if len(event['title']) < 3:
                    quality_metrics['data_issues'].append(f"Very short title: {event['title']}")
                if len(event['title']) > 200:
                    quality_metrics['data_issues'].append(f"Unusually long title: {event['title'][:50]}...")
            
            # Validate date format
            if event.get('start'):
                try:
                    datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
                except:
                    quality_metrics['data_issues'].append(f"Invalid date format: {event['start']}")
        
        # Calculate summary statistics
        quality_metrics['average_completeness'] = np.mean(quality_metrics['completeness_scores'])
        quality_metrics['min_completeness'] = np.min(quality_metrics['completeness_scores'])
        quality_metrics['fully_complete_events'] = sum(1 for score in quality_metrics['completeness_scores'] if score == 1.0)
        
        # Venue analysis
        venues = [e.get('venue') for e in self.events if e.get('venue') and e['venue'] != "null"]
        quality_metrics['unique_venues'] = len(set(venues))
        quality_metrics['events_with_venue'] = len(venues)
        quality_metrics['venue_coverage'] = len(venues) / len(self.events) if self.events else 0
        
        return quality_metrics
    
    # ================== 2. EXTRACTION EVALUATION ==================
    
    def evaluate_extraction_pipeline(self) -> Dict[str, Any]:
        """Evaluate the extraction performance from different sources"""
        
        extraction_metrics = {
            'neu_events': {
                'total_scraped': len(self.events),
                'complete_records': 0,
                'partial_records': 0,
                'failed_records': 0,
                'field_extraction_rates': {}
            }
        }
        
        # Evaluate NEU events extraction
        fields_to_check = ['title', 'start', 'venue', 'description', 'url']
        field_counts = {field: 0 for field in fields_to_check}
        
        for event in self.events:
            fields_present = sum(1 for field in fields_to_check 
                               if event.get(field) and event[field] != "null")
            
            if fields_present == len(fields_to_check):
                extraction_metrics['neu_events']['complete_records'] += 1
            elif fields_present > 0:
                extraction_metrics['neu_events']['partial_records'] += 1
            else:
                extraction_metrics['neu_events']['failed_records'] += 1
            
            for field in fields_to_check:
                if event.get(field) and event[field] != "null":
                    field_counts[field] += 1
        
        # Calculate extraction rates
        for field, count in field_counts.items():
            extraction_metrics['neu_events']['field_extraction_rates'][field] = count / len(self.events) if self.events else 0
        
        # If Instagram data is available
        if self.instagram_data:
            extraction_metrics['instagram'] = self.evaluate_instagram_extraction()
        
        return extraction_metrics
    
    def evaluate_instagram_extraction(self) -> Dict[str, Any]:
        """Evaluate Instagram-specific extraction if data is available"""
        
        instagram_metrics = {
            'total_posts': len(self.instagram_data.get('posts', [])),
            'events_identified': 0,
            'non_events': 0,
            'extraction_success_rate': 0
        }
        
        # This would analyze Instagram-specific extraction
        # Add your Instagram evaluation logic here based on your data structure
        
        return instagram_metrics
    
    # ================== 3. TEMPORAL UNDERSTANDING EVALUATION ==================
    
    def evaluate_temporal_understanding(self) -> Dict[str, Any]:
        """Evaluate system's ability to handle time-based queries"""
        
        temporal_metrics = {
            'date_parsing_accuracy': 0,
            'relative_date_tests': [],
            'date_range_coverage': {}
        }
        
        # Test date parsing
        valid_dates = 0
        date_errors = []
        
        for event in self.events:
            if event.get('start'):
                try:
                    parsed_date = datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
                    valid_dates += 1
                except Exception as e:
                    date_errors.append({
                        'event': event.get('title', 'Unknown'),
                        'date_string': event['start'],
                        'error': str(e)
                    })
        
        temporal_metrics['date_parsing_accuracy'] = valid_dates / len(self.events) if self.events else 0
        temporal_metrics['date_parsing_errors'] = date_errors
        
        # Analyze date distribution
        dates = []
        for event in self.events:
            try:
                date = datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
                dates.append(date)
            except:
                continue
        
        if dates:
            temporal_metrics['date_range_coverage'] = {
                'earliest_event': min(dates).isoformat(),
                'latest_event': max(dates).isoformat(),
                'total_days_covered': (max(dates) - min(dates)).days,
                'events_per_day': len(dates) / ((max(dates) - min(dates)).days + 1)
            }
        
        # Test relative date understanding (simulated)
        test_date = datetime(2025, 10, 11)  # Example reference date
        relative_tests = [
            ('this_weekend', self.get_weekend_events(test_date)),
            ('next_week', self.get_next_week_events(test_date)),
            ('today', self.get_events_on_date(test_date))
        ]
        
        for test_name, events in relative_tests:
            temporal_metrics['relative_date_tests'].append({
                'test': test_name,
                'events_found': len(events),
                'success': len(events) > 0
            })
        
        return temporal_metrics
    
    def get_weekend_events(self, reference_date: datetime) -> List[Dict]:
        """Get events for the upcoming weekend"""
        days_ahead = 5 - reference_date.weekday()  # Saturday
        if days_ahead <= 0:
            days_ahead += 7
        saturday = reference_date + timedelta(days=days_ahead)
        sunday = saturday + timedelta(days=1)
        
        weekend_events = []
        for event in self.events:
            try:
                event_date = datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
                if saturday.date() <= event_date.date() <= sunday.date():
                    weekend_events.append(event)
            except:
                continue
        
        return weekend_events
    
    def get_next_week_events(self, reference_date: datetime) -> List[Dict]:
        """Get events for next week"""
        next_week_start = reference_date + timedelta(days=7)
        next_week_end = next_week_start + timedelta(days=7)
        
        next_week_events = []
        for event in self.events:
            try:
                event_date = datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
                if next_week_start.date() <= event_date.date() < next_week_end.date():
                    next_week_events.append(event)
            except:
                continue
        
        return next_week_events
    
    def get_events_on_date(self, target_date: datetime) -> List[Dict]:
        """Get events on a specific date"""
        events_on_date = []
        for event in self.events:
            try:
                event_date = datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
                if event_date.date() == target_date.date():
                    events_on_date.append(event)
            except:
                continue
        
        return events_on_date
    
    # ================== 4. RAG SYSTEM EVALUATION ==================
    
    def evaluate_rag_system(self) -> Dict[str, Any]:
        """Evaluate RAG system with test queries"""
        
        rag_metrics = {
            'embedding_quality': {},
            'retrieval_tests': [],
            'query_coverage': {}
        }
        
        # Test embedding quality
        if self.events:
            rag_metrics['embedding_quality'] = self.evaluate_embedding_quality()
        
        # Define test queries
        test_queries = [
            {
                'query': 'meditation and yoga events',
                'expected_keywords': ['meditation', 'yoga', 'daily'],
                'type': 'keyword_based'
            },
            {
                'query': 'sports events at Matthews Arena',
                'expected_venue': 'Matthews Arena',
                'type': 'venue_based'
            },
            {
                'query': 'AI and technology workshops',
                'expected_keywords': ['AI', 'digital', 'technology'],
                'type': 'topic_based'
            }
        ]
        
        # Evaluate each test query
        for test in test_queries:
            result = self.evaluate_single_query(test)
            rag_metrics['retrieval_tests'].append(result)
        
        # Calculate query coverage
        rag_metrics['query_coverage'] = {
            'venue_queries': self.calculate_venue_coverage(),
            'temporal_queries': self.calculate_temporal_coverage(),
            'topic_queries': self.calculate_topic_coverage()
        }
        
        return rag_metrics
    
    def evaluate_embedding_quality(self) -> Dict[str, float]:
        """Evaluate the quality of event embeddings"""
        
        # Create embeddings for a sample of events
        sample_size = min(20, len(self.events))
        sample_events = self.events[:sample_size]
        
        embeddings = []
        for event in sample_events:
            text = f"{event.get('title', '')} {event.get('description', '')}"
            embedding = self.model.encode([text])[0]
            embeddings.append(embedding)
        
        # Calculate average cosine similarity
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
        
        return {
            'average_similarity': np.mean(similarities) if similarities else 0,
            'min_similarity': np.min(similarities) if similarities else 0,
            'max_similarity': np.max(similarities) if similarities else 0,
            'std_similarity': np.std(similarities) if similarities else 0
        }
    
    def evaluate_single_query(self, test_query: Dict) -> Dict[str, Any]:
        """Evaluate a single test query"""
        
        query = test_query['query']
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarity with all events
        results = []
        for event in self.events:
            event_text = f"{event.get('title', '')} {event.get('description', '')} {event.get('venue', '')}"
            event_embedding = self.model.encode([event_text])[0]
            similarity = cosine_similarity([query_embedding], [event_embedding])[0][0]
            
            results.append({
                'event': event,
                'similarity': similarity
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = results[:5]  # Top 5 results
        
        # Evaluate based on query type
        evaluation = {
            'query': query,
            'type': test_query['type'],
            'top_5_results': [r['event'].get('title', '') for r in top_results],
            'top_5_scores': [r['similarity'] for r in top_results]
        }
        
        if test_query['type'] == 'keyword_based':
            keywords = test_query.get('expected_keywords', [])
            keyword_hits = 0
            for result in top_results:
                event_text = f"{result['event'].get('title', '')} {result['event'].get('description', '')}".lower()
                if any(keyword.lower() in event_text for keyword in keywords):
                    keyword_hits += 1
            evaluation['precision'] = keyword_hits / len(top_results) if top_results else 0
            
        elif test_query['type'] == 'venue_based':
            expected_venue = test_query.get('expected_venue', '')
            venue_hits = sum(1 for r in top_results if r['event'].get('venue') == expected_venue)
            evaluation['precision'] = venue_hits / len(top_results) if top_results else 0
        
        return evaluation
    
    def calculate_venue_coverage(self) -> Dict[str, float]:
        """Calculate how well venues are covered"""
        venues = [e.get('venue') for e in self.events if e.get('venue') and e['venue'] != 'null']
        unique_venues = list(set(venues))
        
        return {
            'total_unique_venues': len(unique_venues),
            'events_with_venue': len(venues),
            'venue_coverage_rate': len(venues) / len(self.events) if self.events else 0,
            'average_events_per_venue': len(venues) / len(unique_venues) if unique_venues else 0
        }
    
    def calculate_temporal_coverage(self) -> Dict[str, Any]:
        """Calculate temporal coverage of events"""
        dates = []
        for event in self.events:
            try:
                date = datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
                dates.append(date.date())
            except:
                continue
        
        if not dates:
            return {'error': 'No valid dates found'}
        
        unique_dates = set(dates)
        date_range = (max(dates) - min(dates)).days + 1
        
        return {
            'unique_dates': len(unique_dates),
            'date_range_days': date_range,
            'date_coverage_rate': len(unique_dates) / date_range if date_range > 0 else 0,
            'events_per_day': len(dates) / len(unique_dates) if unique_dates else 0
        }
    
    def calculate_topic_coverage(self) -> Dict[str, int]:
        """Calculate coverage of different event topics"""
        topics = {
            'sports': ['soccer', 'hockey', 'basketball', 'volleyball', 'field hockey'],
            'wellness': ['meditation', 'yoga', 'wellness', 'health'],
            'academic': ['workshop', 'seminar', 'lecture', 'talk', 'presentation'],
            'career': ['career', 'job', 'recruitment', 'fair'],
            'arts': ['music', 'art', 'gallery', 'performance'],
            'technology': ['AI', 'digital', 'technology', 'data']
        }
        
        topic_counts = {topic: 0 for topic in topics}
        
        for event in self.events:
            event_text = f"{event.get('title', '')} {event.get('description', '')}".lower()
            for topic, keywords in topics.items():
                if any(keyword.lower() in event_text for keyword in keywords):
                    topic_counts[topic] += 1
        
        return topic_counts
    
    # ================== 5. PERFORMANCE METRICS ==================
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate overall system performance metrics"""
        
        performance = {
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'mrr_score': 0,  # Mean Reciprocal Rank
            'ndcg_score': 0  # Normalized Discounted Cumulative Gain
        }
        
        # Simulate retrieval tests
        test_cases = [
            {
                'query': 'yoga classes',
                'relevant_events': ['Daily Yoga'],
                'retrieved_events': self.search_events('yoga')
            },
            {
                'query': 'hockey games',
                'relevant_events': ["Men's Ice Hockey", "Women's Ice Hockey"],
                'retrieved_events': self.search_events('hockey')
            }
        ]
        
        for test in test_cases:
            retrieved_titles = [e.get('title', '') for e in test['retrieved_events'][:5]]
            relevant = set(test['relevant_events'])
            retrieved = set(retrieved_titles)
            
            # Calculate precision and recall
            if retrieved:
                precision = len(relevant & retrieved) / len(retrieved)
                recall = len(relevant & retrieved) / len(relevant) if relevant else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                performance['precision_scores'].append(precision)
                performance['recall_scores'].append(recall)
                performance['f1_scores'].append(f1)
                
                # Calculate MRR
                for i, event in enumerate(retrieved_titles, 1):
                    if event in relevant:
                        performance['mrr_score'] += 1.0 / i
                        break
        
        # Calculate averages
        performance['avg_precision'] = np.mean(performance['precision_scores']) if performance['precision_scores'] else 0
        performance['avg_recall'] = np.mean(performance['recall_scores']) if performance['recall_scores'] else 0
        performance['avg_f1'] = np.mean(performance['f1_scores']) if performance['f1_scores'] else 0
        performance['mrr_score'] = performance['mrr_score'] / len(test_cases) if test_cases else 0
        
        return performance
    
    def search_events(self, query: str) -> List[Dict]:
        """Simple search function for events"""
        results = []
        query_lower = query.lower()
        
        for event in self.events:
            event_text = f"{event.get('title', '')} {event.get('description', '')}".lower()
            if query_lower in event_text:
                results.append(event)
        
        return results
    
    # ================== 6. GENERATE COMPREHENSIVE REPORT ==================
    
    def generate_full_report(self, output_file: str = 'evaluation_report.json') -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        logger.info("Starting comprehensive evaluation...")
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_events': len(self.events),
                'data_source': 'NEU Events + Instagram' if self.instagram_data else 'NEU Events'
            },
            'evaluations': {}
        }
        
        # Run all evaluations
        logger.info("Evaluating data quality...")
        report['evaluations']['data_quality'] = self.evaluate_data_quality()
        
        logger.info("Evaluating extraction pipeline...")
        report['evaluations']['extraction'] = self.evaluate_extraction_pipeline()
        
        logger.info("Evaluating temporal understanding...")
        report['evaluations']['temporal'] = self.evaluate_temporal_understanding()
        
        logger.info("Evaluating RAG system...")
        report['evaluations']['rag_system'] = self.evaluate_rag_system()
        
        logger.info("Calculating performance metrics...")
        report['evaluations']['performance'] = self.calculate_performance_metrics()
        
        # Calculate overall scores
        report['overall_scores'] = self.calculate_overall_scores(report['evaluations'])
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_file}")
        
        # Print summary
        self.print_summary(report)
        
        return report
    
    def calculate_overall_scores(self, evaluations: Dict) -> Dict[str, float]:
        """Calculate overall system scores"""
        
        scores = {
            'data_quality_score': evaluations['data_quality']['average_completeness'],
            'extraction_score': evaluations['extraction']['neu_events']['field_extraction_rates'].get('title', 0),
            'temporal_accuracy': evaluations['temporal']['date_parsing_accuracy'],
            'rag_precision': np.mean([test.get('precision', 0) for test in evaluations['rag_system']['retrieval_tests']]),
            'overall_f1': evaluations['performance']['avg_f1']
        }
        
        # Calculate weighted overall score
        weights = {
            'data_quality_score': 0.2,
            'extraction_score': 0.2,
            'temporal_accuracy': 0.2,
            'rag_precision': 0.25,
            'overall_f1': 0.15
        }
        
        scores['weighted_overall'] = sum(scores[metric] * weight for metric, weight in weights.items())
        
        return scores
    
    def print_summary(self, report: Dict):
        """Print a summary of the evaluation results"""
        
        print("\n" + "="*60)
        print("EVENTGRAM EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\n DATA OVERVIEW:")
        print(f"  Total Events: {report['data_summary']['total_events']}")
        print(f"  Data Sources: {report['data_summary']['data_source']}")
        
        print(f"\n QUALITY METRICS:")
        quality = report['evaluations']['data_quality']
        print(f"  Average Completeness: {quality['average_completeness']:.2%}")
        print(f"  Events with Venue: {quality['events_with_venue']}/{quality['total_events']}")
        print(f"  Unique Venues: {quality['unique_venues']}")
        
        print(f"\n EXTRACTION PERFORMANCE:")
        extraction = report['evaluations']['extraction']['neu_events']
        print(f"  Complete Records: {extraction['complete_records']}/{extraction['total_scraped']}")
        print(f"  Title Extraction Rate: {extraction['field_extraction_rates'].get('title', 0):.2%}")
        print(f"  Venue Extraction Rate: {extraction['field_extraction_rates'].get('venue', 0):.2%}")
        
        print(f"\n TEMPORAL UNDERSTANDING:")
        temporal = report['evaluations']['temporal']
        print(f"  Date Parsing Accuracy: {temporal['date_parsing_accuracy']:.2%}")
        if temporal.get('date_range_coverage'):
            print(f"  Date Range: {temporal['date_range_coverage'].get('total_days_covered', 0)} days")
        
        print(f"\n RAG SYSTEM PERFORMANCE:")
        performance = report['evaluations']['performance']
        print(f"  Average Precision: {performance['avg_precision']:.2%}")
        print(f"  Average Recall: {performance['avg_recall']:.2%}")
        print(f"  Average F1 Score: {performance['avg_f1']:.2%}")
        print(f"  MRR Score: {performance['mrr_score']:.3f}")
        
        print(f"\n OVERALL SCORES:")
        overall = report['overall_scores']
        print(f"  Weighted Overall Score: {overall['weighted_overall']:.2%}")
        
        print("\n" + "="*60)


# ================== MAIN EXECUTION ==================

def main():
    """Main execution function"""
    
    # Configuration
    EVENTS_FILE = "events.json"  # Your scraped events file
    INSTAGRAM_FILE = "instagram_extractions.json"  # Optional: Instagram data
    OUTPUT_FILE = "eventgram_evaluation_report.json"
    
    # Check if events file exists
    if not os.path.exists(EVENTS_FILE):
        print(f" Events file not found: {EVENTS_FILE}")
        print("Please ensure your events.json file is in the same directory as this script.")
        return
    
    # Initialize evaluator
    print(" Initializing EventGram Evaluator...")
    evaluator = EventGramEvaluator(
        events_file=EVENTS_FILE,
        instagram_data_file=INSTAGRAM_FILE if os.path.exists(INSTAGRAM_FILE) else None
    )
    
    # Run evaluation
    print(" Running comprehensive evaluation...")
    report = evaluator.generate_full_report(OUTPUT_FILE)
    
    print(f"\n Evaluation complete! Full report saved to: {OUTPUT_FILE}")
    
    # Optional: Export key metrics to CSV for further analysis
    try:
        overall_scores = pd.DataFrame([report['overall_scores']])
        overall_scores.to_csv('eventgram_scores.csv', index=False)
        print(f" Scores exported to: eventgram_scores.csv")
    except Exception as e:
        print(f"Warning: Could not export CSV: {e}")


if __name__ == "__main__":
    main()