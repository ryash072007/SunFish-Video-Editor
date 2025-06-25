from moviepy import VideoClip, VideoFileClip
import numpy as np
from PIL import Image
from LLMChain import *
import base64
import os
from dotenv import load_dotenv
import io

debug = False

def frame_to_base64(frame: np.ndarray) -> str:
    """
    Convert a numpy array frame to base64 encoded string.
    
    Args:
        frame: Numpy array representing an image frame
        
    Returns:
        Base64 encoded string of the image
    """
    image = Image.fromarray(frame)
    
    buffer = io.BytesIO()
    
    image.save(buffer, format="JPEG")
    
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    
    return img_base64

def extract_every_x_frames(x: int, video: VideoClip) -> list:
    frames = []
    i = 0
    for frame in video.iter_frames():
        i += 1
        if i % x != 0:
            continue
        frames.append(frame)
        if debug:
            save_image = Image.fromarray(frame)
            save_image.save(f"temp/image_{i*x}.jpg")
    return frames


video_path = "video.mp4"
clip = VideoFileClip(video_path)

frames = extract_every_x_frames(int(clip.fps // 2), clip)

load_dotenv()
logging.basicConfig(level=logging.INFO)


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

sys_prompt = open("image_sys_prompt.txt").read()

bot = LLMChain(
    "sunfish-v0",
    [
        ImageGroqLink(client, sys_prompt, _use_memory=True, _memory_size = 3)
    ]
)


file = open("descriptions.txt", "w")

for frame in frames:
    result: str = bot.forward(frame_to_base64(frame))
    file.write(result.replace("\n", ""))
    file.write("\n\n")

file.close()
