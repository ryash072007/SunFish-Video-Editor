from LLMChain import *
import os
from dotenv import load_dotenv
from groq import Groq
import json
import re
import requests

load_dotenv()
logging.basicConfig(level=logging.INFO)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

ai_chain = LLMChain(
    "SunFish Editor Groq",
    [
        JSONGroqLink(
            groq_client,
            "deepseek-r1-distill-llama-70b", # "llama-3.3-70b-versatile",
            open("groq_pp_sys_prompt.txt").read(),
            False,
        )
    ],
)

result = ai_chain.forward(open("shotstack_hp.json").read() + "\n\nVideo Size: (width: 720, height: 1280)")
video_url: str = "{{IDK}}"
def replace_audio_url(audio_identifier):
    """
    Replace audio identifier with URL from an open source audio service
    
    Args:
        audio_identifier: The identifier for the audio file
        
    Returns:
        str: URL for the audio file from an open source service
        
    Raises:
        ValueError: If no audio is found for the given identifier
    """
    
    try:
        # Use Freesound API to search for audio based on the identifier
        # You would need to register for an API key at https://freesound.org/apiv2/apply/
        FREESOUND_API_KEY = os.environ.get("FREESOUND_API_KEY", "")
        if not FREESOUND_API_KEY:
            logging.warning("FREESOUND_API_KEY not found in environment variables")
            raise ValueError(f"No API key available to search for '{audio_identifier}'")
            
        # Search for sounds matching the identifier
        search_url = f"https://freesound.org/apiv2/search/text/"
        params = {
            "query": audio_identifier,
            "fields": "id,name,previews,download",
            "filter": "license:(\"Creative Commons 0\")",  # Filter for CC0 licensed content
            "page_size": 1
        }
        headers = {"Authorization": f"Token {FREESOUND_API_KEY}"}
        
        response = requests.get(search_url, params=params, headers=headers)
        data = response.json()
        
        if response.status_code == 200 and data.get("results") and len(data["results"]) > 0:
            sound_id = data["results"][0]["id"]
            
            # Get the sound details to access the download link
            sound_url = f"https://freesound.org/apiv2/sounds/{sound_id}/"
            sound_response = requests.get(sound_url, headers=headers)
            sound_data = sound_response.json()
            
            if sound_response.status_code == 200 and sound_data.get("download"):
                # Get the full audio file (requires authentication)
                download_url = sound_data["download"]
                download_response = requests.get(download_url, headers=headers)
                
                if download_response.status_code == 200:
                    return download_url
                else:
                    # Fall back to the HQ preview if full download fails
                    return data["results"][0]["previews"]["preview-hq-mp3"]
            else:
                # Fall back to the HQ preview
                return data["results"][0]["previews"]["preview-hq-mp3"]
        else:
            logging.warning(f"No audio found for '{audio_identifier}'")
            raise ValueError(f"No audio found matching '{audio_identifier}'")
            
    except Exception as e:
        logging.error(f"Error fetching audio URL: {e}")
        raise ValueError(f"Failed to retrieve audio for '{audio_identifier}': {str(e)}")

# New function to fetch font URLs from Google Fonts
def replace_font_url(font_family):
    """
    Replace font identifier with URL to the actual font file from Google Fonts
    
    Args:
        font_family: The font family name (e.g., "Open Sans")
        
    Returns:
        str: URL for the font file (.ttf/.otf)
        
    Raises:
        ValueError: If no font is found for the given family name
    """
    try:
        # Map of common fonts to their direct TTF URLs
        # This is a simple approach to avoid complex CSS parsing
        font_map = {
            "open sans": "https://fonts.gstatic.com/s/opensans/v27/memSYaGs126MiZpBA-UvWbX2vVnXBbObj2OVZyOOSr4dVJWUgsgH1x4gaVc.ttf",
            "roboto": "https://fonts.gstatic.com/s/roboto/v29/KFOlCnqEu92Fr1MmEU9fBBc9.ttf",
            "montserrat": "https://fonts.gstatic.com/s/montserrat/v21/JTUHjIg1_i6t8kCHKm4532VJOt5-QNFgpCtr6Hw5aX8.ttf",
            "lato": "https://fonts.gstatic.com/s/lato/v20/S6u9w4BMUTPHh7USSwaPHA.ttf",
            "clear sans": "https://shotstack-assets.s3-ap-southeast-2.amazonaws.com/fonts/ClearSans-Regular.ttf",
            "montserrat extrabold": "https://shotstack-assets.s3.amazonaws.com/fonts/Montserrat-ExtraBold.ttf",
            "montserrat semibold": "https://shotstack-assets.s3.amazonaws.com/fonts/Montserrat-SemiBold.ttf",
            "opensans bold": "https://shotstack-assets.s3.amazonaws.com/fonts/OpenSans-Bold.ttf",
            "didact gothic": "https://shotstack-assets.s3.amazonaws.com/fonts/DidactGothic-Regular.ttf"
        }
        
        normalized_font = font_family.lower().strip()
        
        # Check if we have a direct mapping
        if normalized_font in font_map:
            return font_map[normalized_font]
        
        # If not in our map, try to fetch from Shotstack's default fonts
        return f"https://shotstack-assets.s3.amazonaws.com/fonts/{font_family.replace(' ', '')}.ttf"
            
    except Exception as e:
        logging.error(f"Error fetching font URL: {e}")
        # Fallback to Open Sans as a safe default
        return "https://fonts.gstatic.com/s/opensans/v27/memSYaGs126MiZpBA-UvWbX2vVnXBbObj2OVZyOOSr4dVJWUgsgH1x4gaVc.ttf"

# Find and replace all {{AUDIO_URL:identifier}} patterns in the result
def replace_audio_urls_in_text(text):
    pattern = r'\{\{AUDIO_URL:(.*?)\}\}'
    return re.sub(pattern, lambda match: replace_audio_url(match.group(1)), text)

# Find and replace all {{FONT_URL:font_family}} patterns in the result
def replace_font_urls_in_text(text):
    pattern = r'\{\{FONT_URL:(.*?)\}\}'
    return re.sub(pattern, lambda match: replace_font_url(match.group(1)), text)

# Apply the replacements to the result
result = replace_audio_urls_in_text(result)
result = replace_font_urls_in_text(result)

# Fix any missing font src attributes in the fonts array
try:
    result_json = json.loads(result)
    
    # Check if we need to add src to fonts
    if "timeline" in result_json and "fonts" in result_json["timeline"]:
        for i, font in enumerate(result_json["timeline"]["fonts"]):
            # Store the font family if it exists (for using in replacement function)
            family = font.get("family", "Open Sans")
            
            # Create a new font object with only the src attribute
            if "src" not in font:
                result_json["timeline"]["fonts"][i] = {"src": replace_font_url(family)}
            else:
                # Keep the src but remove any other properties
                result_json["timeline"]["fonts"][i] = {"src": font["src"]}
    
    # Ensure output doesn't have both resolution and size
    if "output" in result_json:
        if "resolution" in result_json["output"] and "size" in result_json["output"]:
            # Prefer size over resolution if both exist
            del result_json["output"]["resolution"]
    
    # Ensure track 0 and track 1 are both audio tracks
    if "timeline" in result_json and "tracks" in result_json["timeline"]:
        tracks = result_json["timeline"]["tracks"]
        
        # If we have fewer than 2 tracks, add empty audio tracks
        while len(tracks) < 2:
            tracks.append({
                "clips": [{
                    "asset": {
                        "type": "audio",
                        "src": "https://cdn.freesound.org/previews/345/345284_6026280-hq.mp3",
                        "volume": 0
                    },
                    "start": 0,
                    "length": 1
                }]
            })
        
        # If track 1 doesn't have audio assets, insert an empty audio track at index 1
        if len(tracks) >= 2:
            if "clips" not in tracks[1] or not all(clip.get("asset", {}).get("type") == "audio" for clip in tracks[1]["clips"]):
                # Insert an empty audio track at position 1
                tracks.insert(1, {
                    "clips": [{
                        "asset": {
                            "type": "audio",
                            "src": "https://cdn.freesound.org/previews/345/345284_6026280-hq.mp3",
                            "volume": 0
                        },
                        "start": 0,
                        "length": 1
                    }]
                })
    
    # Save the updated JSON
    result = json.dumps(result_json)
    
except json.JSONDecodeError:
    # If not valid JSON, we'll just continue with the text-based replacement
    pass

try:
    result = json.loads(result)
    json.dump(result, open("shotstack_full.json", "w"))
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    print(result)
