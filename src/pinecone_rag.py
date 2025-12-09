#!/usr/bin/env python3
"""
Pinecone RAG System for Instagram Data
Uses Pinecone vector database for persistent, scalable storage
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import hashlib

# Import existing dependencies
from anthropic import Anthropic
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI for native 1536-dimension embeddings (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Pinecone for vector storage
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print(" Pinecone not available. Install with: pip install pinecone")
    sys.exit(1)

# Embeddings using sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print(" SentenceTransformer not available. Install with: pip install sentence-transformers")
    sys.exit(1)

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pinecone_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """Pydantic model for search results"""
    query: str
    answer: str
    relevant_posts: List[Dict[str, Any]]
    confidence_score: float = Field(ge=0.0, le=1.0)
    timestamp: str
    search_method: str = "pinecone_semantic"
    metadata: Dict[str, Any] = Field(default_factory=dict)

@dataclass
class InstagramPost:
    """Structured Instagram post data"""
    post_id: str
    caption: str
    transcribed_text: str
    keywords: List[str]
    timestamp: str
    post_url: str
    confidence: float
    image_description: str = ""
    
    def get_combined_text(self) -> str:
        """Get all text content combined for search"""
        parts = []
        if self.caption: 
            parts.append(f"Caption: {self.caption}")
        if self.transcribed_text: 
            parts.append(f"Image Text: {self.transcribed_text}")
        if self.image_description: 
            parts.append(f"Image Description: {self.image_description}")
        if self.keywords: 
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        
        # Add temporal context
        if self.timestamp:
            parts.append(f"Posted: {self.timestamp}")
            
        return " | ".join(parts)
    
    def to_pinecone_vector(self, embedding: List[float]) -> Dict[str, Any]:
        """Convert to Pinecone vector format"""
        return {
            "id": f"post_{self.post_id}",
            "values": embedding,
            "metadata": {
                "post_id": self.post_id,
                "caption": self.caption[:500],  # Pinecone metadata size limit
                "transcribed_text": self.transcribed_text[:500],
                "keywords": self.keywords[:10],  # Limit keywords
                "timestamp": self.timestamp,
                "post_url": self.post_url,
                "confidence": self.confidence,
                "image_description": self.image_description[:200],
                "has_caption": bool(self.caption.strip()),
                "has_transcription": bool(self.transcribed_text.strip()),
                "combined_text": self.get_combined_text()[:800]  # For display
            }
        }

class PineconeRAG:
    """RAG implementation using Pinecone vector database"""
    
    def __init__(self, data_file: str, index_name: str = "acm-hackathon"):
        self.data_file = data_file
        self.index_name = index_name
        self.posts: List[InstagramPost] = []
        self.embeddings_model = None
        self.pinecone_client = None
        self.index = None
        self.anthropic_client = Anthropic()  # Uses ANTHROPIC_API_KEY from env
        
        # Initialize Pinecone
        self._initialize_pinecone()
        
        # Initialize embeddings model with multiple options
        self.embedding_model_type = os.getenv("EMBEDDING_MODEL", "sentence_transformer")  # Options: openai, sentence_transformer
        self._initialize_embeddings_model()
    
    def _initialize_embeddings_model(self):
        """Initialize the appropriate embeddings model based on configuration"""
        if self.embedding_model_type == "openai" and OPENAI_AVAILABLE:
            # Use OpenAI embeddings (native 1536 dimensions)
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error(" OPENAI_API_KEY required for OpenAI embeddings")
                logger.info(" Falling back to SentenceTransformer")
                self.embedding_model_type = "sentence_transformer"
            else:
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                
                # Set dimensions based on model
                if self.openai_model == "text-embedding-3-large":
                    self.embedding_dimension = 3072  # Can be reduced to 1536 if needed
                else:  # text-embedding-3-small or text-embedding-ada-002
                    self.embedding_dimension = 1536
                
                logger.info(f" Initialized OpenAI embeddings ({self.openai_model}, native {self.embedding_dimension} dimensions)")
                return
        
        # Fallback to SentenceTransformer (requires padding)
        if EMBEDDINGS_AVAILABLE:
            self.embeddings_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self.embedding_dimension = 1536  # Will be padded from 768
            logger.info(" Initialized SentenceTransformer embeddings (768→1536 dimensions with padding)")
        else:
            logger.error(" No embedding model available. Install: pip install sentence-transformers openai")
            sys.exit(1)
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using the configured model"""
        if self.embedding_model_type == "openai" and hasattr(self, 'openai_client'):
            # Use OpenAI API (native 1536 dimensions)
            try:
                response = self.openai_client.embeddings.create(
                    model=self.openai_model,
                    input=text,
                    encoding_format="float"
                )
                embedding = response.data[0].embedding
                
                # Handle text-embedding-3-large (3072D → 1536D)
                if len(embedding) > 1536:
                    embedding = embedding[:1536]  # Truncate to 1536D
                
                return embedding
            except Exception as e:
                logger.error(f" OpenAI embedding failed: {e}. Falling back to SentenceTransformer")
                # Fallback to sentence transformer
                if hasattr(self, 'embeddings_model'):
                    raw_embedding = self.embeddings_model.encode([text])[0].tolist()
                    return self._pad_embedding(raw_embedding)
                else:
                    raise Exception("No fallback embedding model available")
        else:
            # Use SentenceTransformer (requires padding to 1536)
            raw_embedding = self.embeddings_model.encode([text])[0].tolist()
            return self._pad_embedding(raw_embedding)
    
    def _pad_embedding(self, embedding: List[float]) -> List[float]:
        """Pad embedding to 1536 dimensions if needed"""
        current_dim = len(embedding)
        target_dim = 1536
        
        if current_dim == target_dim:
            return embedding
        elif current_dim < target_dim:
            # Pad with zeros
            padding = [0.0] * (target_dim - current_dim)
            return embedding + padding
        else:
            # Truncate if somehow larger
            return embedding[:target_dim]
    
    def _initialize_pinecone(self) -> None:
        """Initialize Pinecone client and index"""
        try:
            # Get Pinecone API key from environment
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                logger.error(" PINECONE_API_KEY not found in environment variables")
                print("Please set PINECONE_API_KEY in your .env file")
                sys.exit(1)
            
            # Initialize Pinecone client
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            
            # Check if index exists, create if not
            existing_indexes = [idx.name for idx in self.pinecone_client.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f" Creating Pinecone index: {self.index_name}")
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=1536,  # Match your configured Pinecone index dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Match your existing setup
                    )
                )
                logger.info(f" Created Pinecone index: {self.index_name}")
            else:
                logger.info(f" Using existing Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = self.pinecone_client.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f" Pinecone index stats: {stats.total_vector_count} vectors")
            
        except Exception as e:
            logger.error(f" Failed to initialize Pinecone: {str(e)}")
            raise
    
    def load_posts(self) -> None:
        """Load Instagram posts from scraped_provenance.json"""
        logger.info(f" Loading posts from {self.data_file}")
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            posts_data = data.get("posts", [])
            successful_posts = 0
            
            for post_data in posts_data:
                transcription = post_data.get("transcription", {})
                if transcription.get("status") != "successful":
                    continue
                    
                extracted_text = transcription.get("extracted_text", {})
                image_analysis = transcription.get("image_analysis", {})
                post_info = post_data.get("post_data", {})
                
                post = InstagramPost(
                    post_id=post_data.get("post_id", ""),
                    caption=post_info.get("caption", ""),
                    transcribed_text=extracted_text.get("main_text", ""),
                    keywords=extracted_text.get("keywords", []),
                    timestamp=post_info.get("timestamp", ""),
                    post_url=post_info.get("post_url", ""),
                    confidence=extracted_text.get("confidence", 0.0),
                    image_description=image_analysis.get("description", "")
                )
                
                self.posts.append(post)
                successful_posts += 1
            
            logger.info(f" Loaded {successful_posts} posts with successful transcriptions")
            
        except Exception as e:
            logger.error(f" Error loading posts: {str(e)}")
            raise
    
    def _load_insertion_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata about previously inserted vectors"""
        metadata_file = "data/metadata/pinecone_insertions.json"
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f" Could not load insertion metadata: {e}")
        
        return {}
    
    def _save_insertion_metadata(self, metadata: Dict[str, Dict[str, Any]]) -> None:
        """Save metadata about inserted vectors"""
        metadata_file = "data/metadata/pinecone_insertions.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.debug(f" Saved insertion metadata to {metadata_file}")
        except Exception as e:
            logger.error(f" Failed to save insertion metadata: {e}")

    def embed_and_store(self, force_reindex: bool = False) -> None:
        """Create embeddings and store in Pinecone"""
        if not self.posts:
            logger.error(" No posts loaded. Call load_posts() first.")
            return
        
        logger.info(f" Processing embeddings for {len(self.posts)} posts...")
        
        # Load insertion metadata
        insertion_metadata = self._load_insertion_metadata()
        
        # Check existing vectors (unless forcing reindex)
        existing_ids = set()
        if not force_reindex:
            try:
                # First check local metadata (much faster)
                existing_ids = set(insertion_metadata.keys())
                logger.info(f" Found {len(existing_ids)} vectors in local metadata")
                
                # Verify with Pinecone stats
                stats = self.index.describe_index_stats()
                logger.info(f" Pinecone reports {stats.total_vector_count} total vectors")
                
                # If there's a mismatch, do a full check (but limit it)
                if len(existing_ids) != stats.total_vector_count and stats.total_vector_count > 0:
                    logger.info(" Metadata mismatch detected, checking Pinecone directly...")
                    
                    # Use multiple queries to get all IDs if needed
                    all_pinecone_ids = set()
                    batch_size = 1000
                    
                    # Get a sample to verify
                    query_result = self.index.query(
                        vector=[0.0] * 1536,
                        top_k=min(batch_size, stats.total_vector_count),
                        include_metadata=True
                    )
                    all_pinecone_ids.update(match.id for match in query_result.matches)
                    
                    # Update existing_ids with verified data
                    existing_ids = all_pinecone_ids
                    logger.info(f" Verified {len(existing_ids)} existing vectors in Pinecone")
                    
            except Exception as e:
                logger.warning(f" Could not check existing vectors: {e}")
                existing_ids = set()
        
        # Prepare vectors to upsert
        vectors_to_upsert = []
        new_insertions = {}
        new_vectors = 0
        skipped_vectors = 0
        
        for post in self.posts:
            vector_id = f"post_{post.post_id}"
            
            # Skip if already exists (unless forcing reindex)
            if not force_reindex and vector_id in existing_ids:
                skipped_vectors += 1
                logger.debug(f" Skipping existing post {post.post_id}")
                continue
            
            # Create embedding using configured model
            combined_text = post.get_combined_text()
            embedding = self._get_embedding(combined_text)
            
            # Create Pinecone vector
            vector_data = post.to_pinecone_vector(embedding)
            vectors_to_upsert.append(vector_data)
            
            # Prepare metadata for this insertion
            model_info = f"openai-text-embedding-3-small" if self.embedding_model_type == "openai" else "sentence-transformers/all-mpnet-base-v2"
            new_insertions[vector_id] = {
                "post_id": post.post_id,
                "insertion_timestamp": datetime.now().isoformat(),
                "embedding_model": model_info,
                "embedding_dimension": len(embedding),
                "embedding_type": self.embedding_model_type,
                "text_length": len(combined_text),
                "has_caption": bool(post.caption.strip()),
                "has_transcription": bool(post.transcribed_text.strip()),
                "transcription_confidence": post.confidence,
                "force_reindex": force_reindex
            }
            
            new_vectors += 1
            logger.debug(f" Prepared embedding for post {post.post_id}")
        
        logger.info(f" Processing summary: {new_vectors} new vectors, {skipped_vectors} skipped (already exist)")
        
        # Upsert vectors to Pinecone
        if vectors_to_upsert:
            logger.info(f" Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
            
            # Upsert in batches to avoid API limits
            batch_size = 100
            successful_inserts = 0
            
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                try:
                    upsert_response = self.index.upsert(vectors=batch)
                    batch_count = len(batch)
                    successful_inserts += batch_count
                    
                    logger.info(f" Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1} ({batch_count} vectors)")
                    logger.debug(f" Upsert response: {upsert_response}")
                    
                except Exception as e:
                    logger.error(f" Failed to upsert batch {i//batch_size + 1}: {e}")
                    # Remove failed insertions from metadata
                    for vector in batch:
                        vector_id = vector['id']
                        if vector_id in new_insertions:
                            del new_insertions[vector_id]
                    raise
            
            # Update insertion metadata with successful inserts
            insertion_metadata.update(new_insertions)
            self._save_insertion_metadata(insertion_metadata)
            
            logger.info(f" Successfully stored {successful_inserts} new vectors in Pinecone")
            logger.info(f" Updated insertion metadata with {len(new_insertions)} records")
        else:
            logger.info(" All posts already exist in Pinecone (use force_reindex=True to reprocess)")
        
        # Final stats
        final_stats = self.index.describe_index_stats()
        logger.info(f" Pinecone index now contains {final_stats.total_vector_count} total vectors")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search using Pinecone"""
        logger.info(f" Searching Pinecone for: '{query}'")
        
        try:
            # Create query embedding using configured model
            query_embedding = self._get_embedding(query)
            
            # Search Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"has_transcription": True}  # Only posts with successful transcriptions
            )
            
            # Process results
            relevant_posts = []
            for match in search_results.matches:
                if match.score > 0.3:  # Minimum similarity threshold
                    relevant_posts.append({
                        "post_id": match.metadata.get("post_id", ""),
                        "post_url": match.metadata.get("post_url", ""),
                        "timestamp": match.metadata.get("timestamp", ""),
                        "confidence": match.metadata.get("confidence", 0.0),
                        "similarity_score": match.score,
                        "caption": match.metadata.get("caption", ""),
                        "transcribed_text": match.metadata.get("transcribed_text", ""),
                        "keywords": match.metadata.get("keywords", []),
                        "image_description": match.metadata.get("image_description", ""),
                        "combined_text": match.metadata.get("combined_text", "")
                    })
            
            logger.info(f" Found {len(relevant_posts)} relevant posts (similarity > 0.3)")
            return relevant_posts
            
        except Exception as e:
            logger.error(f" Pinecone search failed: {str(e)}")
            return []
    
    def generate_answer(self, query: str, relevant_posts: List[Dict[str, Any]]) -> str:
        """Generate answer using Claude based on relevant posts"""
        if not relevant_posts:
            return "I couldn't find any relevant information about that in the Instagram posts stored in Pinecone."
        
        # Prepare context from relevant posts
        context_parts = []
        for i, post in enumerate(relevant_posts[:3], 1):  # Use top 3 posts
            post_context = f"Post {i} (ID: {post['post_id']}, Similarity: {post['similarity_score']:.2f}):\n"
            
            if post.get('caption'):
                post_context += f"Caption: {post['caption']}\n"
            if post.get('transcribed_text'):
                post_context += f"Image text: {post['transcribed_text']}\n"
            if post.get('keywords'):
                post_context += f"Keywords: {', '.join(post['keywords'])}\n"
            if post.get('post_url'):
                post_context += f"URL: {post['post_url']}\n"
            if post.get('timestamp'):
                post_context += f"Posted: {post['timestamp']}\n"
            
            post_context += f"Transcription confidence: {post.get('confidence', 0):.1%}\n"
            context_parts.append(post_context)
        
        context = "\n" + "="*50 + "\n".join(context_parts)
        
        # Create prompt for Claude
        prompt = f"""You are helping users find information from Instagram posts stored in a Pinecone vector database. Based on the following Instagram post data retrieved through semantic search, please answer the user's question.

Question: {query}

Relevant Instagram Posts (retrieved via Pinecone semantic search):
{context}

Instructions:
- Provide a helpful and accurate answer based only on the information in these posts
- Include specific details like dates, times, locations when mentioned
- If the posts contain event information, highlight important details (dates, venues, times)
- Include relevant post URLs when helpful for verification
- Mention the confidence scores when relevant
- If the information isn't sufficient to fully answer the question, say so
- Be conversational and helpful
- Prioritize posts with higher similarity scores and transcription confidence

Answer:"""

        try:
            # Call Claude API
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f" Error calling Claude API: {e}")
            return "Sorry, I encountered an error while generating the response. Please try again."
    
    def search(self, query: str, top_k: int = 5) -> SearchResult:
        """Main search function using Pinecone"""
        logger.info(f" Processing search query: '{query}'")
        
        # Find relevant posts using Pinecone
        relevant_posts = self.semantic_search(query, top_k=top_k)
        
        # Generate answer
        answer = self.generate_answer(query, relevant_posts)
        
        # Calculate confidence score
        if relevant_posts:
            # Weight by similarity score and transcription confidence
            total_weight = 0
            weighted_confidence = 0
            
            for post in relevant_posts:
                similarity_weight = post.get('similarity_score', 0.0)
                transcription_confidence = post.get('confidence', 0.0)
                combined_weight = similarity_weight * transcription_confidence
                
                weighted_confidence += combined_weight
                total_weight += similarity_weight
            
            confidence_score = weighted_confidence / total_weight if total_weight > 0 else 0.0
        else:
            confidence_score = 0.0
        
        # Prepare result
        result = SearchResult(
            query=query,
            answer=answer,
            relevant_posts=relevant_posts,
            confidence_score=confidence_score,
            timestamp=datetime.now().isoformat(),
            search_method="pinecone_semantic",
            metadata={}
        )
        
        logger.info(f" Search completed. Found {len(relevant_posts)} relevant posts, confidence: {confidence_score:.1%}")
        return result
    
    def get_insertion_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about Pinecone insertions"""
        insertion_metadata = self._load_insertion_metadata()
        
        if not insertion_metadata:
            return {"total_insertions": 0, "message": "No insertion metadata found"}
        
        # Analyze insertion data
        total_insertions = len(insertion_metadata)
        with_captions = sum(1 for meta in insertion_metadata.values() if meta.get("has_caption", False))
        with_transcriptions = sum(1 for meta in insertion_metadata.values() if meta.get("has_transcription", False))
        
        confidence_scores = [meta.get("transcription_confidence", 0) for meta in insertion_metadata.values()]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Get insertion timeline
        timestamps = [meta.get("insertion_timestamp", "") for meta in insertion_metadata.values()]
        timestamps = [ts for ts in timestamps if ts]  # Filter empty timestamps
        
        first_insertion = min(timestamps) if timestamps else "unknown"
        last_insertion = max(timestamps) if timestamps else "unknown"
        
        return {
            "total_insertions": total_insertions,
            "posts_with_captions": with_captions,
            "posts_with_transcriptions": with_transcriptions,
            "average_confidence": avg_confidence,
            "first_insertion": first_insertion,
            "last_insertion": last_insertion,
            "embedding_model": insertion_metadata[list(insertion_metadata.keys())[0]].get("embedding_model", "unknown") if insertion_metadata else "unknown",
            "metadata_file": "data/metadata/pinecone_insertions.json"
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded data and Pinecone index"""
        local_stats = {
            "local_posts_loaded": len(self.posts),
            "posts_with_captions": len([p for p in self.posts if p.caption.strip()]),
            "posts_with_transcriptions": len([p for p in self.posts if p.transcribed_text.strip()]),
            "average_confidence": sum(p.confidence for p in self.posts) / len(self.posts) if self.posts else 0,
            "total_keywords": sum(len(p.keywords) for p in self.posts),
            "unique_keywords": len(set(kw for p in self.posts for kw in p.keywords)),
        }
        
        # Get Pinecone stats
        pinecone_stats = {}
        try:
            index_stats = self.index.describe_index_stats()
            pinecone_stats = {
                "pinecone_total_vectors": index_stats.total_vector_count,
                "pinecone_dimension": 1536,
                "pinecone_index_name": self.index_name,
                "pinecone_status": "connected"
            }
        except Exception as e:
            pinecone_stats = {
                "pinecone_status": f"error: {str(e)}"
            }
        
        return {**local_stats, **pinecone_stats, "search_method": "pinecone_semantic"}
    
    def delete_index(self) -> None:
        """Delete the Pinecone index (use with caution!)"""
        try:
            self.pinecone_client.delete_index(self.index_name)
            logger.info(f"🗑️ Deleted Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f" Failed to delete index: {e}")

def main():
    """Interactive demo of the Pinecone RAG system"""
    data_file = "data/metadata/scraped_provenance.json"
    
    if not os.path.exists(data_file):
        print(f" Data file not found: {data_file}")
        print("Please run Instagram scraping and transcription first:")
        print("1. python pilots/APIScraper.py")
        print("2. python pilots/image_transcibe.py --all")
        return
    
    print(" Pinecone Instagram RAG System")
    print("=" * 50)
    
    # Check environment variables
    if not os.getenv("PINECONE_API_KEY"):
        print(" PINECONE_API_KEY not found in environment")
        print("Please add PINECONE_API_KEY to your .env file")
        return
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(" ANTHROPIC_API_KEY not found in environment")
        print("Please add ANTHROPIC_API_KEY to your .env file")
        return
    
    try:
        # Initialize RAG
        print(" Initializing Pinecone RAG system...")
        rag = PineconeRAG(data_file, index_name="acm-hackathon")
        
        # Load posts
        print(" Loading Instagram posts...")
        rag.load_posts()
        
        # Create/update embeddings
        print(" Creating embeddings and storing in Pinecone...")
        rag.embed_and_store()
        
        # Show stats
        stats = rag.get_stats()
        print(f"\n System Statistics:")
        print(f"   Local posts loaded: {stats['local_posts_loaded']}")
        print(f"   Pinecone vectors: {stats.get('pinecone_total_vectors', 'unknown')}")
        print(f"   Average confidence: {stats['average_confidence']:.1%}")
        print(f"   Unique keywords: {stats['unique_keywords']}")
        print(f"   Search method: {stats['search_method']}")
        
        print(f"\n Pinecone RAG system ready!")
        
        # Interactive query loop
        print("\n" + "="*50)
        print("ASK QUESTIONS ABOUT INSTAGRAM POSTS")
        print("="*50)
        print("Examples:")
        print("- What events is NU Sanskriti hosting?")
        print("- Tell me about Navratri celebrations")
        print("- What dance auditions are happening?")
        print("- When is move-in week?")
        print("\nType 'quit' to exit, 'stats' for statistics, 'insertions' for insertion details\n")
        
        while True:
            try:
                print("\n" + "-" * 40)
                query = input(" Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print(" Goodbye!")
                    break
                
                if query.lower() == 'stats':
                    current_stats = rag.get_stats()
                    insertion_stats = rag.get_insertion_stats()
                    
                    print(f"\n Current Statistics:")
                    for key, value in current_stats.items():
                        print(f"   {key}: {value}")
                    
                    print(f"\n📈 Insertion Statistics:")
                    for key, value in insertion_stats.items():
                        if key == "average_confidence":
                            print(f"   {key}: {value:.1%}")
                        else:
                            print(f"   {key}: {value}")
                    continue
                
                if query.lower() == 'insertions':
                    insertion_stats = rag.get_insertion_stats()
                    print(f"\n Detailed Insertion Statistics:")
                    print(f"    Total vectors inserted: {insertion_stats.get('total_insertions', 0)}")
                    print(f"    Posts with captions: {insertion_stats.get('posts_with_captions', 0)}")
                    print(f"     Posts with transcriptions: {insertion_stats.get('posts_with_transcriptions', 0)}")
                    print(f"    Average confidence: {insertion_stats.get('average_confidence', 0):.1%}")
                    print(f"    First insertion: {insertion_stats.get('first_insertion', 'unknown')}")
                    print(f"    Last insertion: {insertion_stats.get('last_insertion', 'unknown')}")
                    print(f"    Embedding model: {insertion_stats.get('embedding_model', 'unknown')}")
                    print(f"    Metadata file: {insertion_stats.get('metadata_file', 'unknown')}")
                    continue
                
                if not query:
                    continue
                
                print(" Searching Pinecone...")
                result = rag.search(query)
                
                print(f"\n💬 Answer:")
                print(result.answer)
                
                if result.relevant_posts:
                    print(f"\n📚 Based on {len(result.relevant_posts)} post(s) from Pinecone:")
                    for i, post in enumerate(result.relevant_posts[:2], 1):  # Show top 2
                        similarity = post.get('similarity_score', 0)
                        confidence = post.get('confidence', 0)
                        print(f"   {i}. Post {post['post_id']} (similarity: {similarity:.2f}, confidence: {confidence:.1%})")
                        if post.get('post_url'):
                            print(f"      {post['post_url']}")
                
                print(f"\n Overall Confidence: {result.confidence_score:.1%}")
                
            except KeyboardInterrupt:
                print("\n Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f" Error: {str(e)}")
                
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        print(f" System initialization failed: {str(e)}")

if __name__ == "__main__":
    main()