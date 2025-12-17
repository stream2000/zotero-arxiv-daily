# from llama_cpp import Llama # Removed direct import
from openai import OpenAI
import google.generativeai as genai
from loguru import logger
from time import sleep

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English", provider: str = "openai"):
        self.is_api_llm = False
        self.llm = None # Initialize to None
        self.provider = provider

        if api_key:
            if self.provider == "openai":
                self.llm = OpenAI(api_key=api_key, base_url=base_url)
                self.is_api_llm = True
                logger.info("Using OpenAI API for LLM features.")
            elif self.provider == "gemini":
                genai.configure(api_key=api_key)
                self.llm = genai.GenerativeModel(model)
                self.is_api_llm = True
                logger.info(f"Using Google Gemini API ({model}) for LLM features.")
            else:
                logger.error(f"Unknown provider: {self.provider}")
        else:
            logger.warning("Local LLM (llama-cpp-python) not configured or installed. LLM features will be disabled. Set USE_LLM_API=1 and OPENAI_API_KEY to enable API-based LLM.")
            # Note: If a user manually installs llama-cpp-python, they could re-enable this path if desired.
            # For now, it is explicitly disabled when api_key is not provided and llama-cpp-python is removed.
        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        if self.llm is None:
            logger.warning("LLM features are disabled, returning placeholder for TLDR.")
            return "TLDR: LLM generation skipped."

        if self.is_api_llm:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if self.provider == "openai":
                        response = self.llm.chat.completions.create(messages=messages, temperature=0, model=self.model)
                        return response.choices[0].message.content
                    elif self.provider == "gemini":
                        # Convert messages to a single prompt for Gemini
                        prompt = ""
                        for msg in messages:
                            prompt += f"{msg['role']}: {msg['content']}\n\n"
                        response = self.llm.generate_content(prompt)
                        return response.text
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} to call {self.provider} API failed: {e}")
                    if attempt == max_retries - 1:
                        raise # Re-raise if all retries fail
                    sleep(3)
        else:
            # This branch for local llama-cpp-python should not be reached if self.llm is None.
            # If self.llm was initialized, it implies llama_cpp was present.
            # However, since it's removed, this path should not execute.
            logger.error("Attempted to use local LLM, but it's not initialized. This code path should not be reached.")
            return "TLDR: Local LLM failed to initialize."


def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English", provider: str = "openai"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang, provider=provider)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM