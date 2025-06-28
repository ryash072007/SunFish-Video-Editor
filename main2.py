from LLMChain import *
import os
from dotenv import load_dotenv
from groq import Groq
import json

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

try:
    # result = result.replace("{{VIDEO_URL}}", "fuckus")
    result = json.loads(result)
    json.dump(result, open("shotstack_full.json", "w"))
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    print(result)
