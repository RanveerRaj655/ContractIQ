"""
generation/llm_client.py
------------------------
Wrapper around the OpenRouter API for grounded RAG generation.
"""

import os

from openai import OpenAI

from contractiq.chunking import Chunk
from contractiq.config import settings
from contractiq.generation.prompts import RAG_SYSTEM_PROMPT, build_user_prompt


class LLMClient:
    def __init__(self):
        api_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing. Please set it in your .env file.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = settings.openrouter_model
        self.max_tokens = settings.generation_max_tokens
        self.temperature = settings.generation_temperature

    def generate_answer(self, query: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
        """
        Generate an answer to a query grounded strictly in the provided chunks.
        """
        user_content = build_user_prompt(query, retrieved_chunks)
        
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        )
        return response.choices[0].message.content
