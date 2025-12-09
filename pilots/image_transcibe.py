# Import necessary libraries
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from typing import List, Optional
import base64
import os
import json
import glob
import logging
import re
import argparse
from datetime import datetime, timezone

# Load environment variables from .env file
load_dotenv()

# Setup logging configuration
def setup_logging():
    """Setup logging configuration for image transcription process"""
    # Create logs directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, "image_transcribe.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Define Pydantic models for structured output
class ExtractedText(BaseModel):
    """Model for structured text extraction from images"""
    main_text: str = Field(description="The primary text content found in the image")
    title: Optional[str] = Field(description="Title or heading text if present")
    keywords: List[str] = Field(description="Important keywords or phrases extracted from the text")
    text_confidence: float = Field(description="Confidence score for text extraction (0-1)", ge=0, le=1)
    language: Optional[str] = Field(description="Detected language of the text")
    
class ImageAnalysis(BaseModel):
    """Model for complete image analysis with extracted text"""
    extracted_text: ExtractedText = Field(description="Structured text data from the image")
    image_description: str = Field(description="Brief description of what's in the image")
    text_locations: List[str] = Field(description="Locations where text appears (e.g., 'top-left', 'center', 'bottom')")

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    logger.info(f"Starting base64 encoding for image: {image_path}")
    try:
        file_size = os.path.getsize(image_path)
        logger.info(f"Image file size: {file_size} bytes")
        
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        encoded_size = len(encoded_string)
        logger.info(f"Base64 encoding completed. Encoded size: {encoded_size} characters")
        return encoded_string
        
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        raise Exception(f"Error encoding image: {str(e)}")

def get_image_media_type(image_path: str) -> str:
    """Determine the correct media type based on file extension"""
    _, ext = os.path.splitext(image_path.lower())
    
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    
    return media_type_map.get(ext, 'image/jpeg')  # Default to jpeg

def extract_text_from_image_with_claude(image_path: str) -> ImageAnalysis:
    """
    Extract text from image using Claude Vision and return structured Pydantic object
    
    Args:
        image_path: Path to the image file
        
    Returns:
        ImageAnalysis object with structured text data
    """
    logger.info(f"Starting Claude Vision analysis for image: {os.path.basename(image_path)}")
    
    try:
        # Encode the image to base64
        image_base64 = encode_image_to_base64(image_path)
        
        # Determine correct media type
        media_type = get_image_media_type(image_path)
        logger.info(f"Detected media type: {media_type}")
        
        # Create the LLM without structured output first
        logger.info("Initializing Claude model (claude-3-haiku-20240307)")
        llm = ChatAnthropic(
            temperature=0,
            model="claude-3-haiku-20240307"
        )
        
        # Create the prompt template for image analysis
        logger.info("Creating prompt template for image analysis")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert OCR and image analysis assistant. 
            Analyze the provided image and extract all visible text content.
            
            Please respond with a JSON object in this exact format:
            {{
                "extracted_text": {{
                    "main_text": "all the text you can see in the image",
                    "title": "title or heading if present, or null",
                    "keywords": ["keyword1", "keyword2", "keyword3"],
                    "text_confidence": 0.95,
                    "language": "english"
                }},
                "image_description": "brief description of what you see in the image",
                "text_locations": ["top-left", "center", "bottom-right"]
            }}
            
            Be thorough and accurate in your text extraction. Provide a confidence score between 0 and 1."""),
            ("user", [
                {"type": "text", "text": "Please analyze this image and extract all text content. Return only the JSON object, no other text:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_base64}"
                    }
                }
            ])
        ])
        
        # Create the chain and invoke it
        chain = prompt | llm
        
        logger.info("Sending request to Claude Vision API...")
        start_time = datetime.now()
        response = chain.invoke({})
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"Claude Vision API response received in {processing_time:.2f} seconds")
        
        # Parse the response content
        response_content = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Raw response content: {response_content[:200]}...")
        
        # Try to parse JSON from response
        try:
            import re
            # Extract JSON from response if it's wrapped in other text
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                response_data = json.loads(json_str)
                logger.info("Successfully parsed JSON from response")
                
                # Create ImageAnalysis object from parsed data
                extracted_text = ExtractedText(
                    main_text=response_data.get("extracted_text", {}).get("main_text", ""),
                    title=response_data.get("extracted_text", {}).get("title"),
                    keywords=response_data.get("extracted_text", {}).get("keywords", []),
                    text_confidence=float(response_data.get("extracted_text", {}).get("text_confidence", 0.0)),
                    language=response_data.get("extracted_text", {}).get("language")
                )
                
                result = ImageAnalysis(
                    extracted_text=extracted_text,
                    image_description=response_data.get("image_description", ""),
                    text_locations=response_data.get("text_locations", [])
                )
                
                # Log analysis results
                logger.info(f"Text extraction completed successfully")
                logger.info(f"Confidence score: {result.extracted_text.text_confidence:.2%}")
                logger.info(f"Detected language: {result.extracted_text.language}")
                logger.info(f"Keywords found: {len(result.extracted_text.keywords)}")
                logger.info(f"Text locations: {len(result.text_locations)}")
                
                return result
                
            else:
                logger.error("Could not find JSON in response")
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Error parsing response as JSON: {str(e)}")
            # If JSON parsing fails, create a fallback response with the raw text
            result = ImageAnalysis(
                extracted_text=ExtractedText(
                    main_text=response_content[:1000],  # Truncate if too long
                    title=None,
                    keywords=[],
                    text_confidence=0.5,
                    language="unknown"
                ),
                image_description=f"Raw response (parsing failed): {response_content[:200]}...",
                text_locations=[]
            )
            return result
        
    except Exception as e:
        logger.error(f"Error during Claude Vision analysis: {str(e)}")
        logger.error(f"Returning error response structure")
        
        # Return a default structure if analysis fails
        return ImageAnalysis(
            extracted_text=ExtractedText(
                main_text=f"Error extracting text: {str(e)}",
                title=None,
                keywords=[],
                text_confidence=0.0,
                language=None
            ),
            image_description="Could not analyze image due to error",
            text_locations=[]
        )

def get_post_id_from_filename(filename: str) -> str:
    """Extract post_id from image filename (remove extension)"""
    post_id = os.path.splitext(filename)[0]
    logger.info(f"Extracted post_id '{post_id}' from filename '{filename}'")
    return post_id

def check_transcription_status(post_id: str) -> dict:
    """
    Check if transcription already exists for a post_id
    
    Args:
        post_id: The post ID to check
        
    Returns:
        dict with status info: {'exists': bool, 'successful': bool, 'data': dict or None}
    """
    # Get the correct path to metadata file from script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_file = os.path.join(script_dir, "..", "data", "metadata", "scraped_provenance.json")
    metadata_file = os.path.normpath(metadata_file)
    
    logger.info(f"Checking transcription status for post_id: {post_id}")
    
    try:
        # Read existing metadata
        if not os.path.exists(metadata_file):
            logger.warning(f"Metadata file not found at {metadata_file}")
            return {'exists': False, 'successful': False, 'data': None}
            
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Find the post with matching post_id
        for post in metadata.get("posts", []):
            if post.get("post_id") == post_id:
                transcription = post.get("transcription", {})
                
                if transcription:
                    status = transcription.get("status", "")
                    is_successful = status == "successful"
                    
                    logger.info(f"Found existing transcription for post_id '{post_id}':")
                    logger.info(f"  - Status: {status}")
                    logger.info(f"  - Successful: {is_successful}")
                    
                    if is_successful:
                        logger.info(f"  - Confidence: {transcription.get('extracted_text', {}).get('confidence', 0):.2%}")
                        logger.info(f"  - Language: {transcription.get('extracted_text', {}).get('language', 'N/A')}")
                        logger.info(f"  - Transcribed at: {transcription.get('transcribed_at', 'N/A')}")
                    
                    return {
                        'exists': True, 
                        'successful': is_successful, 
                        'data': transcription
                    }
                else:
                    logger.info(f"No transcription data found for post_id '{post_id}'")
                    return {'exists': False, 'successful': False, 'data': None}
        
        logger.info(f"Post_id '{post_id}' not found in metadata")
        return {'exists': False, 'successful': False, 'data': None}
        
    except Exception as e:
        logger.error(f"Error checking transcription status: {str(e)}")
        return {'exists': False, 'successful': False, 'data': None}

def update_scraped_provenance(post_id: str, transcription_result: ImageAnalysis) -> bool:
    """Update scraped_provenance.json with transcription results"""
    # Get the correct path to metadata file from script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_file = os.path.join(script_dir, "..", "data", "metadata", "scraped_provenance.json")
    metadata_file = os.path.normpath(metadata_file)
    logger.info(f"Updating metadata for post_id: {post_id}")
    logger.info(f"Metadata file path: {os.path.abspath(metadata_file)}")
    
    try:
        # Read existing metadata
        if os.path.exists(metadata_file):
            logger.info(f"Reading existing metadata from {metadata_file}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata with {len(metadata.get('posts', []))} posts")
        else:
            logger.warning(f"Metadata file not found at {metadata_file}")
            print(f"Warning: Metadata file not found at {metadata_file}")
            return False
        
        # Find the post with matching post_id and update it
        updated = False
        total_posts = len(metadata.get("posts", []))
        logger.info(f"Searching for post_id '{post_id}' among {total_posts} posts")
        
        for idx, post in enumerate(metadata.get("posts", [])):
            if post.get("post_id") == post_id:
                logger.info(f"Found matching post at index {idx}")
                
                # Add transcription data to the post
                transcription_data = {
                    "status": "successful",
                    "transcribed_at": datetime.now(timezone.utc).isoformat(),
                    "extracted_text": {
                        "main_text": transcription_result.extracted_text.main_text,
                        "title": transcription_result.extracted_text.title,
                        "keywords": transcription_result.extracted_text.keywords,
                        "confidence": transcription_result.extracted_text.text_confidence,
                        "language": transcription_result.extracted_text.language
                    },
                    "image_analysis": {
                        "description": transcription_result.image_description,
                        "text_locations": transcription_result.text_locations
                    },
                    "method": "claude_vision_api"
                }
                
                post["transcription"] = transcription_data
                updated = True
                
                logger.info(f"Added transcription data to post:")
                logger.info(f"  - Status: successful")
                logger.info(f"  - Confidence: {transcription_result.extracted_text.text_confidence:.2%}")
                logger.info(f"  - Language: {transcription_result.extracted_text.language}")
                logger.info(f"  - Keywords count: {len(transcription_result.extracted_text.keywords)}")
                break
        
        if updated:
            # Save updated metadata back to file
            logger.info(f"Saving updated metadata to {metadata_file}")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully updated metadata for post_id: {post_id}")
            print(f" Updated metadata for post_id: {post_id}")
            return True
        else:
            logger.warning(f"Post with ID '{post_id}' not found in metadata")
            print(f" Post with ID '{post_id}' not found in metadata")
            return False
            
    except Exception as e:
        logger.error(f"Error updating metadata: {str(e)}", exc_info=True)
        print(f" Error updating metadata: {str(e)}")
        return False

def find_images_to_process(images_dir: str, limit: int = None) -> List[str]:
    """Find image files to process from the images directory"""
    logger.info(f"Searching for images in directory: {images_dir}")
    
    # Look for common image extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(images_dir, ext)
        found_files = glob.glob(pattern)
        if found_files:
            logger.info(f"Found {len(found_files)} files with extension {ext}")
        image_files.extend(found_files)
    
    total_found = len(image_files)
    logger.info(f"Total images found: {total_found}")
    
    # Return based on limit
    if limit is None or limit == 0:
        logger.info(f"Processing all {total_found} image(s)")
        return image_files
    else:
        limited_files = image_files[:limit]
        logger.info(f"Limiting to {limit} image(s) for processing")
        return limited_files

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract text from images using Claude Vision API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_transcibe.py                    # Process 1 image (default)
  python image_transcibe.py --limit 5         # Process up to 5 images
  python image_transcibe.py --all             # Process all images
  python image_transcibe.py --limit 0         # Process all images (same as --all)
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--limit", 
        type=int, 
        default=1,
        help="Number of images to process (default: 1, 0 = all images)"
    )
    group.add_argument(
        "--all", 
        action="store_true",
        help="Process all images in the directory"
    )
    
    parser.add_argument(
        "--skip-successful",
        action="store_true",
        default=True,
        help="Skip images that already have successful transcriptions (default: True)"
    )
    
    return parser.parse_args()

def main():
    """Main function to process images with command line controls"""
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("="*60)
    logger.info("Starting Claude Image Text Extraction Process")
    logger.info(f"Arguments: limit={args.limit}, all={args.all}, skip_successful={args.skip_successful}")
    logger.info("="*60)
    
    print(" Claude Image Text Extraction with Batch Processing!")
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "..", "images")
    images_dir = os.path.normpath(images_dir)
    
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Images directory exists: {os.path.exists(images_dir)}")
    
    print(f" Looking for images in: {images_dir}")
    
    # Determine processing limit
    limit = None if args.all else args.limit
    if args.limit == 0:
        limit = None  # Process all images
    
    # Find images to process
    image_files = find_images_to_process(images_dir, limit)
    
    if not image_files:
        logger.error(f"No images found in {images_dir}")
        print(f" No images found in {images_dir}")
        print("Please ensure there are image files in the images directory.")
        return
    
    # Display processing information
    total_images = len(image_files)
    if args.all or limit is None:
        print(f" Processing all {total_images} images found")
        logger.info(f"Processing all {total_images} images")
    else:
        print(f" Processing {min(args.limit, total_images)} out of {total_images} images found")
        logger.info(f"Processing {min(args.limit, total_images)} out of {total_images} images")
    
    processed_count = 0
    skipped_count = 0
    success_count = 0
    error_count = 0
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        image_filename = os.path.basename(image_path)
        post_id = get_post_id_from_filename(image_filename)
        
        logger.info(f"Processing image {i}/{total_images}: {image_filename} (post_id: {post_id})")
        print(f"\n[{i}/{total_images}]  Processing: {image_filename}")
        print(f"    Post ID: {post_id}")
        
        try:
            # Check if transcription already exists and is successful
            logger.info("Checking existing transcription status")
            transcription_status = check_transcription_status(post_id)
            
            if args.skip_successful and transcription_status['exists'] and transcription_status['successful']:
                logger.info(" Transcription already exists and is successful - skipping Claude API call")
                print(f"     Already transcribed successfully - skipping to save costs")
                skipped_count += 1
                continue
            
            processed_count += 1
            
            if transcription_status['exists'] and transcription_status['successful']:
                logger.info("Using existing successful transcription")
                # Create ImageAnalysis object from existing data
                existing_data = transcription_status['data']
                extracted_text_data = existing_data.get('extracted_text', {})
                
                analysis_result = ImageAnalysis(
                    extracted_text=ExtractedText(
                        main_text=extracted_text_data.get('main_text', ''),
                        title=extracted_text_data.get('title'),
                        keywords=extracted_text_data.get('keywords', []),
                        text_confidence=float(extracted_text_data.get('confidence', 0.0)),
                        language=extracted_text_data.get('language')
                    ),
                    image_description=existing_data.get('image_analysis', {}).get('description', ''),
                    text_locations=existing_data.get('image_analysis', {}).get('text_locations', [])
                )
                
                print(f"    Using existing transcription (no API call needed)")
                success_count += 1
                
            else:
                # Need to perform transcription
                if transcription_status['exists']:
                    logger.info(" Existing transcription found but status is not successful - retrying")
                    print(f"     Previous attempt failed - retrying with Claude API...")
                else:
                    logger.info(" No existing transcription found - performing new transcription")
                    print(f"    New transcription with Claude API...")
                
                # Extract text from image
                logger.info("Starting text extraction process")
                start_time = datetime.now()
                
                analysis_result = extract_text_from_image_with_claude(image_path)
                
                end_time = datetime.now()
                total_time = (end_time - start_time).total_seconds()
                logger.info(f"Text extraction completed in {total_time:.2f} seconds")
                
                if analysis_result:
                    # Update metadata with transcription results
                    logger.info("Starting metadata update process")
                    success = update_scraped_provenance(post_id, analysis_result)
                    
                    if success:
                        logger.info("Metadata update completed successfully")
                        print(f"    Successfully transcribed and updated metadata")
                        success_count += 1
                    else:
                        logger.error("Metadata update failed")
                        print(f"    Transcribed but failed to update metadata")
                        error_count += 1
                else:
                    logger.error("Text extraction failed")
                    print(f"    Failed to extract text from image")
                    error_count += 1
            
            # Show brief results for batch processing
            if analysis_result and analysis_result.extracted_text:
                text_preview = analysis_result.extracted_text.main_text[:60]
                if len(analysis_result.extracted_text.main_text) > 60:
                    text_preview += "..."
                print(f"    Text preview: {text_preview}")
                print(f"    Confidence: {analysis_result.extracted_text.text_confidence:.1%}")
                
        except Exception as e:
            logger.error(f"Error processing {image_filename}: {str(e)}", exc_info=True)
            print(f"    Error processing image: {str(e)}")
            error_count += 1
            processed_count += 1
            continue
    
    # Display final summary
    print(f"\n" + "="*50)
    print(" BATCH PROCESSING SUMMARY")
    print("="*50)
    print(f"Total images found: {total_images}")
    print(f"Images processed: {processed_count}")
    print(f"Images skipped: {skipped_count}")
    print(f"Successful transcriptions: {success_count}")
    print(f"Errors encountered: {error_count}")
    
    if processed_count > 0:
        success_rate = (success_count / processed_count) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    logger.info("Batch processing completed")
    logger.info(f"Summary - Total: {total_images}, Processed: {processed_count}, Skipped: {skipped_count}, Success: {success_count}, Errors: {error_count}")
    logger.info("="*60)

if __name__ == "__main__":
    main()