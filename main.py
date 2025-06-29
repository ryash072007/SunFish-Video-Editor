from LLMChain import *
import os
from dotenv import load_dotenv
from google import genai
from groq import Groq
import json
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
gemini_client = genai.Client()


class VideoGeminiResponse(BaseModel):
    edit_type: str
    start_time: str
    duration: str
    parameters: list
    reason: str
    envision: str


ai_chain = LLMChain(
    "SunFish Editor",
    [
        VideoGeminiLink(
            gemini_client,
            "gemini-2.5-flash",
            open("video_sys_prompt.txt").read(),
            True,
            VideoGeminiResponse,
        )
    ],
)

result = ai_chain.forward(
    "User: I want a faster, much shorter, paced video with even more cuts", "video.mp4"
)

try:
    result_dict = json.loads(result)
    json.dump(result_dict, open("shotstack_hp.json", "w"))
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    print(result)
