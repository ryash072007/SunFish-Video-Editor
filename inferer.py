from LLMChain import *
import os
from dotenv import load_dotenv

desc_file = open("descriptions.txt")
descs = desc_file.read()

descs = descs.split("\n\n")

desc_file.close()

load_dotenv()
logging.basicConfig(level=logging.INFO)


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

bot = LLMChain(
    "sunfish-v0.1",
    [
        TextGroqLink(client, "llama3-70b-8192", "Relate the given frame description with the previous 2 frame desciptions given", _memory_size = 3)
    ]

)

file = open("result.txt", "w")

for idx, desc in enumerate(descs):
    result = bot.forward(desc)
    file.write(f"{idx} - {result}\n")

file.close()
