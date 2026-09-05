"""
guardrails/hallucination.py
---------------------------
Uses an LLM judge to verify if an answer is supported by the context.
"""

from contractiq.generation.llm_client import LLMClient
from contractiq.generation.prompts import format_context

JUDGE_PROMPT = """You are a strict Hallucination Judge.
You will be provided with retrieved context and an AI's answer.
Your task is to determine if the AI's answer is strictly supported by the context.

Respond in exactly this format on the first line:
[VERDICT]

Followed by a brief one-line reason.

Valid verdicts are ONLY:
SUPPORTED (The answer is fully backed by the context)
PARTIALLY_SUPPORTED (The answer mixes context with outside knowledge or hallucinations)
UNSUPPORTED (The answer contradicts the context or states facts not found in the context)

If the AI's answer is exactly "I don't have enough information in the provided contracts to answer this.", you must return SUPPORTED.
"""

def check_hallucination(llm_client: LLMClient, query: str, answer: str, chunks: list) -> dict:
    """
    Calls the LLM as a judge to evaluate the generated answer.
    """
    # Fast path if we already gracefully abstained during generation
    if "don't have enough information" in answer.lower():
        return {"verdict": "SUPPORTED", "reason": "Graceful abstention."}

    context_str = format_context(chunks)
    
    user_content = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer: {answer}"
    
    # We call the OpenAI client directly since our LLMClient has a hardcoded RAG_SYSTEM_PROMPT
    response = llm_client.client.chat.completions.create(
        model=llm_client.model,
        max_tokens=150,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_content}
        ]
    )
    
    judge_text = response.choices[0].message.content.strip()
    lines = judge_text.split('\n')
    verdict = lines[0].strip() if len(lines) > 0 else "UNKNOWN"
    reason = " ".join(lines[1:]).strip() if len(lines) > 1 else ""
    
    # Standardize the verdict string
    if "SUPPORTED" in verdict and "PARTIALLY" not in verdict and "UN" not in verdict:
        verdict = "SUPPORTED"
    elif "PARTIALLY_SUPPORTED" in verdict:
        verdict = "PARTIALLY_SUPPORTED"
    elif "UNSUPPORTED" in verdict:
        verdict = "UNSUPPORTED"
    else:
        verdict = "UNKNOWN"
        reason = judge_text
        
    return {
        "verdict": verdict,
        "reason": reason
    }
