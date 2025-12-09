import requests
import json
import os
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables from .env file
load_dotenv()

# Configuration
ACCESS_TOKEN = os.getenv("META_DEV_ACCESS_TOKEN")
USER_ID = os.getenv("META_DEV_USER_ID") 
API_VERSION = os.getenv("API_VERSION")

print("ACCESS_TOKEN : ", ACCESS_TOKEN)
print("USER_ID : ", USER_ID)
print("API_VERSION : ", API_VERSION)

# Validate token before running
if not ACCESS_TOKEN or ACCESS_TOKEN == "":
    print("ERROR: INSTAGRAM_ACCESS_TOKEN is not set in .env file!")
    print("\nTo fix this:")
    print("1. Create a .env file in your project root")
    print("2. Add the following line to .env:")
    print("   META_DEV_ACCESS_TOKEN=your_access_token_here")
    print("   META_DEV_USER_ID=your_instagram_business_account_id")
    print("\nTo get your access token:")
    print("1. Go to: https://developers.facebook.com/tools/explorer/")
    print("2. Select your app")
    print("3. Generate an access token with 'instagram_basic' permissions")
    exit(1)

# Create required folders if they don't exist
os.makedirs("images", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)

def fetch_instagram_posts():
    """Fetch posts from Instagram Graph API"""
    url = f"https://graph.facebook.com/{API_VERSION}/{USER_ID}/media"
    params = {
        "fields": "id,caption,media_type,media_url,permalink,timestamp",
        "access_token": ACCESS_TOKEN
    }
    
    print(f"Making request to: {url}")
    print(f"Access token (first 20 chars): {ACCESS_TOKEN[:20]}...")
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        if response.status_code == 400:
            print("\n  Common causes:")
            print("1. Access token is missing or empty")
            print("2. Access token has expired")
            print("3. Access token doesn't have correct permissions")
            print("\nTo fix:")
            print("- Generate a new token at: https://developers.facebook.com/tools/explorer/")
            print("- Make sure to select 'instagram_basic' and 'pages_read_engagement' permissions")
        return None

def download_image(image_url, image_name):
    """Download image and save to images folder"""
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            filepath = os.path.join("images", image_name)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {image_name}")
            return True
        else:
            print(f"Failed to download: {image_name}")
            return False
    except Exception as e:
        print(f"Error downloading {image_name}: {e}")
        return False

def process_posts(data):
    """Process posts and create JSON with required format plus metadata"""
    posts_data = []
    metadata = []
    
    if not data or "data" not in data:
        print("No data found")
        return posts_data, metadata
    
    scrape_timestamp = datetime.now(timezone.utc).isoformat()
    
    for idx, post in enumerate(data["data"]):
        # Generate image filename
        post_id = post.get("id", f"post_{idx}")
        image_name = f"{post_id}.jpg"
        
        # Download image if available
        download_success = False
        if post.get("media_url"):
            download_success = download_image(post["media_url"], image_name)
        
        # Create post entry (same as before)
        post_entry = {
            "post_url": post.get("permalink", ""),
            "caption": post.get("caption", ""),
            "image_name": image_name
        }
        
        # Create detailed metadata entry
        metadata_entry = {
            "post_id": post_id,
            "source": "instagram",
            "source_type": "instagram_graph_api",
            "scraped_at": scrape_timestamp,
            "scraper_version": "1.0",
            "post_data": {
                "post_url": post.get("permalink", ""),
                "caption": post.get("caption", ""),
                "media_type": post.get("media_type", ""),
                "media_url": post.get("media_url", ""),
                "timestamp": post.get("timestamp", ""),
                "instagram_id": post.get("id", "")
            },
            "file_info": {
                "image_name": image_name,
                "image_path": f"images/{image_name}",
                "download_success": download_success,
                "file_exists": os.path.exists(f"images/{image_name}")
            },
            "processing_info": {
                "processed_at": scrape_timestamp,
                "index_in_batch": idx,
                "api_response_received": True
            }
        }
        
        posts_data.append(post_entry)
        metadata.append(metadata_entry)
    
    return posts_data, metadata

def save_to_json(posts_data, filename="posts_data.json"):
    """Save posts data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(posts_data, f, ensure_ascii=False, indent=2)
    print(f"\nData saved to {filename}")

def save_metadata(metadata, filename="data/metadata/scraped_provenance.json"):
    """Save metadata to JSON file"""
    # Create metadata summary
    metadata_summary = {
        "scraping_session": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "instagram",
            "method": "instagram_graph_api",
            "total_posts_processed": len(metadata),
            "successful_downloads": sum(1 for m in metadata if m["file_info"]["download_success"]),
            "api_version": API_VERSION,
            "user_id": USER_ID
        },
        "posts": metadata
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metadata_summary, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to {filename}")

def main():
    print("Fetching Instagram posts...")
    data = fetch_instagram_posts()
    
    if data:
        print(f"Found {len(data.get('data', []))} posts\n")
        posts_data, metadata = process_posts(data)
        
        # Save both posts data and metadata
        save_to_json(posts_data)
        save_metadata(metadata)
        
        print(f"\nTotal posts processed: {len(posts_data)}")
        print(f"Images saved in: ./images/")
        print(f"Metadata saved in: ./data/metadata/scraped_provenance.json")
    else:
        print("Failed to fetch posts")

if __name__ == "__main__":
    main()
