import time
from groq import Groq
from google import genai
import logging
import types
import json
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BaseLLMLink:
    def __init__(self):
        pass


class ImageGroqLink(BaseLLMLink):
    def __init__(
        self,
        _client: Groq,
        _system_prompt: str = None,
        _model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        _use_memory: bool = False,
        _memory_size: int = None,
    ):
        self.client: Groq = _client
        self.model: str = _model
        self.system_prompt = _system_prompt

        self.use_memory: bool = _use_memory
        self.memory_size: int = _memory_size

        self.memory: list = []
        if self.system_prompt:
            self.memory.append({"role": "system", "content": self.system_prompt})

        if not self.use_memory and _memory_size is not None:
            logger.warning(
                "You should not specify memory_size when use_memory is False."
            )
        elif self.use_memory and _memory_size is None:
            logger.warning(
                "Using memory but memory_size is not specified. Memory will increase and indefinetely and could hit rate limits fast."
            )

    def forward(self, _image_b64: str) -> str:
        if _image_b64 is None:
            return None

        message_to_send: dict = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_image_b64}",
                    },
                },
            ],
        }

        send_messages: list = self.memory + [message_to_send]

        chat_completion = self.client.chat.completions.create(
            messages=send_messages, model=self.model, stream=False
        )

        response_message = chat_completion.choices[0].message

        if self.use_memory:
            self.memory.extend([message_to_send, response_message])
            if self.memory_size is not None:
                size_diff: int = len(self.memory) - self.memory_size
                if size_diff > 0:
                    del self.memory[1 : 1 + size_diff]

        return response_message.content


class AudioGroqLink(BaseLLMLink):
    def __init__(
        self,
        _client: Groq,
        _model: str,
    ):
        self.client: Groq = _client
        self.model: str = _model

    def forward(self, _wav_file: str) -> str:

        if _wav_file is None:
            return None

        with open(_wav_file, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
                file=file, model=self.model, language="en", temperature=0.3
            )
            return transcription.text


class LLMChain(BaseLLMLink):
    def __init__(self, _name: str, _llm_links: list):
        self.llm_links: list = _llm_links
        self.name = _name
        logger.debug(f"LLM classes provided: {self.llm_links}")

        super().__init__()

    def forward(self, prompt, vid_path=None):
        logger.info(f"Starting generation for LLM Chain: {self.name}")
        self.next_layer_input = prompt
        for idx, link in enumerate(self.llm_links):
            logger.info(f"Processing link at idx: {idx}")
            if isinstance(link, VideoGeminiLink):
                if vid_path is None:
                    logger.error(f"No video passed but LLMChain has VideoGeminiLink")
                self.next_layer_input = link.forward(self.next_layer_input, vid_path)
            elif isinstance(link, BaseLLMLink):
                self.next_layer_input = link.forward(self.next_layer_input)
            elif isinstance(link, types.FunctionType):
                self.next_layer_input = link(self.next_layer_input)
            else:
                logger.error(f"Undefined type of link found in LLMChain: {type(link)}")
        return self.next_layer_input


class JSON2StrLink(BaseLLMLink):
    def __init__(self, _key: str):
        self.key = _key
        super().__init__()

    def forward(self, _prompt: dict):
        logger.info(f"JSON Prompt: {_prompt}")
        return str(_prompt[self.key])


class SplitterLink(BaseLLMLink):
    def __init__(self, _splits: dict):
        self.splits = _splits
        super().__init__()

    def forward(self, _prompts: dict) -> dict:
        return_prompt = {}
        if len(self.splits) != len(_prompts):
            logger.warning(
                f"Split size of {len(self.splits)} does not match that of prompts: {len(_prompts)}"
            )
        for key, link in self.splits.items():
            prompt = _prompts[key]
            output = None
            if isinstance(link, BaseLLMLink):
                output = link.forward(prompt)
            elif isinstance(link, types.FunctionType):
                output = link(prompt)
            else:
                logger.error(
                    f"Undefined type of link found in SplitterLink: {type(link)}"
                )
            return_prompt[key] = output
        return return_prompt


class TextGroqLink(BaseLLMLink):
    def __init__(
        self,
        _client: Groq,
        _model: str,
        _system_prompt: str = None,
        _use_memory: bool = True,
        _memory_size: int = None,
    ):
        self.client: Groq = _client
        self.model: str = _model
        self.system_prompt: str = _system_prompt

        logger.info(f"TextGroqLink System prompt: {_system_prompt}")

        self.use_memory: bool = _use_memory
        self.memory_size: int = _memory_size

        self.memory: list = []
        if self.system_prompt:
            self.memory.append({"role": "system", "content": self.system_prompt})

        if not self.use_memory and _memory_size is not None:
            logger.warning(
                "You should not specify memory_size when use_memory is False."
            )
        elif self.use_memory and _memory_size is None:
            logger.warning(
                "Using memory but memory_size is not specified. Memory will increase and indefinetely and could hit rate limits fast."
            )

    def forward(self, _prompt: str) -> str:

        if _prompt is None:
            return ""

        message_to_send: dict = {
            "role": "user",
            "content": _prompt,
        }

        logger.info(f"TextGroqLink PROMPT: {_prompt}")

        send_messages: list = self.memory + [message_to_send]

        chat_completion = self.client.chat.completions.create(
            messages=send_messages,
            model=self.model,
            stream=False,
        )

        response_message = chat_completion.choices[0].message

        if self.use_memory:
            self.memory.extend([message_to_send, response_message])
            if self.memory_size is not None:
                size_diff: int = len(self.memory) - self.memory_size
                if size_diff > 0:
                    del self.memory[1 : 1 + size_diff]

        return response_message.content


class JSONGroqLink(TextGroqLink):

    # Overiding the forward function of JSONGroqLink
    def forward(self, _prompt: str) -> dict:

        message_to_send: dict = {
            "role": "user",
            "content": _prompt,
        }

        logger.info(f"JSONGroqLink PROMPT: {_prompt}")

        send_messages: list = self.memory + [message_to_send]

        chat_completion = self.client.chat.completions.create(
            messages=send_messages,
            model=self.model,
            stream=False,
            response_format={"type": "json_object"},
        )

        response_message = chat_completion.choices[0].message

        if self.use_memory:
            self.memory.extend([message_to_send, response_message])
            if self.memory_size is not None:
                size_diff: int = len(self.memory) - self.memory_size
                if size_diff > 0:
                    del self.memory[1 : 1 + size_diff]

        return json.loads(response_message.content)


class VideoGeminiLink(BaseLLMLink):
    def __init__(
        self,
        _client: genai.Client,
        _model: str,
        _system_prompt: str = None,
        _as_json: bool = False,
        _json_schema: BaseModel = None
    ):
        self.client: genai.Client = _client
        self.model: str = _model
        self.system_prompt: str = _system_prompt
        self.as_json = _as_json
        self.json_schema = _json_schema
        self.config = genai.types.GenerateContentConfig()

        logger.info(f"VideoGeminiLink System prompt: {_system_prompt}")

        if self.system_prompt:
            self.config.system_instruction = self.system_prompt
        if self.as_json:
            self.config.response_mime_type = "application/json"
            if self.json_schema:
                self.config.response_schema = list[self.json_schema]
            else:
                logger.error("as_json is true but no schema is provided, output may not be completely valid")


    def forward(self, _prompt: str, _video_path: str):
        logger.info(f"VideoGeminiLink PROMPT: {_prompt}")

        vid_file = self.client.files.upload(file=_video_path)
        while vid_file.state.name == "PROCESSING":
            logger.info("Waiting for video processing…")
            time.sleep(5)
            vid_file = self.client.files.get(name=vid_file.name)
        if vid_file.state.name != "ACTIVE":
            logger.error(f"Video file failed with state {vid_file.state.name}")

        response = self.client.models.generate_content(
            model=self.model,
            contents=[vid_file, _prompt],
            config=self.config,
        )

        self.client.files.delete(name=vid_file.name)

        return response.text


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    llm_chain = LLMChain(
        "main",
        [
            TextGroqLink(
                client, "gemma2-9b-it", "You are a question generation agent", True, 3
            ),
            TextGroqLink(
                client,
                "llama3-70b-8192",
                "You are an question answering agent that answers with only 10 tokens",
                False,
            ),
        ],
    )

    result = llm_chain.forward("Generate a question about UAE")
    print(result)

    result = llm_chain.forward("Generate the same question again")
    print(result)
