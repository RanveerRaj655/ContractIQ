"""
generation/prompts.py
---------------------
Prompt templates for grounded generation.
"""

from contractiq.chunking import Chunk

# Strict prompt requiring exact adherence to the provided context.
RAG_SYSTEM_PROMPT = """You are an expert legal AI assistant. Your task is to answer questions about legal contracts using ONLY the provided numbered context chunks.

STRICT RULES:
1. You must base your answer strictly and exclusively on the provided context. Do NOT use outside knowledge.
2. If the context does not contain enough information to fully and accurately answer the question, you must explicitly output the exact phrase: "I don't have enough information in the provided contracts to answer this." Do NOT guess or hallucinate.
3. If you can answer the question based on the context, you must cite the source chunks you used by appending their number in brackets, like this: [Chunk 1].

Answer clearly, concisely, and professionally."""


def format_context(chunks: list[tuple[Chunk, float]]) -> str:
    """Format the retrieved chunks into a numbered list for the prompt."""
    if not chunks:
        return "No context provided."
        
    formatted = []
    for i, (chunk, _score) in enumerate(chunks, 1):
        formatted.append(f"--- [Chunk {i}] Document: {chunk.doc_id} ---\n{chunk.text.strip()}\n")
    return "\n".join(formatted)

def build_user_prompt(query: str, chunks: list[tuple[Chunk, float]]) -> str:
    """Combine the user's query and the formatted context."""
    context_str = format_context(chunks)
    return f"Here is the context extracted from the contracts:\n\n{context_str}\n\nQuestion: {query}"
