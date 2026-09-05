"""
api/main.py
-----------
FastAPI web service exposing the ContractIQ Guarded RAG Pipeline.
"""

import json
import time
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel

# Add src/ to the path so we can import contractiq
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contractiq.chunking import structure_aware_chunk
from contractiq.config import settings
from contractiq.pipeline import RetrievalPipeline
from contractiq.generation import LLMClient
from contractiq.guardrails import scan_input, should_abstain, check_hallucination

# Global state
pipeline = None
llm = None
rate_limit_records = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup event: Load the pipeline and LLM once.
    We load only the 40 benchmark contracts to utilize the instantly-loading FAISS cache.
    """
    global pipeline, llm
    print("--- Starting up ContractIQ API ---")
    
    print(f"Loading 40 benchmark contracts from {settings.corpus_dir}...")
    with open(settings.benchmark_mini, encoding="utf-8") as f:
        bench = json.load(f)
    doc_ids = sorted({snip["file_path"] for t in bench["tests"] for snip in t["snippets"]})
    corpus_files = [settings.corpus_dir / doc_id for doc_id in doc_ids]

    all_chunks = []
    for filepath in corpus_files:
        text = filepath.read_text(encoding="utf-8")
        all_chunks.extend(structure_aware_chunk(filepath.name, text))

    print(f"Initializing RetrievalPipeline (strategy: {settings.active_retrieval_strategy})...")
    pipeline = RetrievalPipeline(all_chunks)
    
    print("Initializing LLMClient...")
    llm = LLMClient()
    
    print("--- Startup Complete ---")
    yield
    # Shutdown logic (none needed)


app = FastAPI(
    title="ContractIQ API",
    description="Production-style guarded RAG pipeline for legal contracts.",
    version="1.0.0",
    lifespan=lifespan
)


# --- Rate Limiting (In-Memory) ---
def rate_limiter(request: Request):
    """
    A basic in-memory rate limiter restricting clients to 20 requests per minute.
    """
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    if client_ip not in rate_limit_records:
        rate_limit_records[client_ip] = []
        
    # Remove records older than 60 seconds
    rate_limit_records[client_ip] = [
        t for t in rate_limit_records[client_ip] 
        if current_time - t < 60
    ]
    
    if len(rate_limit_records[client_ip]) >= 20:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Maximum 20 requests per minute.")
        
    rate_limit_records[client_ip].append(current_time)
    return client_ip


# --- Models ---
class QueryRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    doc_id: str
    score: float
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    abstained: bool
    hallucination_verdict: str
    latency_ms: int


# --- Logging ---
def log_query(log_entry: dict):
    out_dir = settings.eval_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "query_log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/eval-report")
def eval_report():
    """
    Returns the pre-computed evaluation benchmark comparison JSON.
    """
    eval_file = settings.eval_results_dir / "retrieval_comparison.json"
    if not eval_file.exists():
        raise HTTPException(status_code=404, detail="Eval report not found.")
    
    with open(eval_file, encoding="utf-8") as f:
        return json.load(f)


@app.post("/query", response_model=QueryResponse)
def query_pipeline(req: QueryRequest, client_ip: str = Depends(rate_limiter)):
    """
    Executes the guarded RAG pipeline on a given question.
    """
    start_time = time.time()
    question = req.question
    
    # 1. Input Filter
    input_flags = scan_input(question)
    
    # 2. Retrieve
    retrieved_chunks_scores = pipeline.retrieve(question, k=settings.top_k)
    
    # 3. Abstention Check
    top_scores = [score for chunk, score in retrieved_chunks_scores]
    top_score = max(top_scores) if top_scores else 0.0
    abstained = should_abstain(top_scores)
    
    actual_answer = ""
    hallucination_verdict = "N/A"
    hallucination_reason = ""
    
    if abstained:
        actual_answer = f"Insufficient context (Score {top_score:.4f} < Threshold {settings.abstention_threshold}). Abstained."
    elif not retrieved_chunks_scores:
        actual_answer = "No chunks retrieved."
    else:
        # 4. Generate
        actual_answer = llm.generate_answer(question, retrieved_chunks_scores)
        # 5. Hallucination Check
        halluc_res = check_hallucination(llm, question, actual_answer, retrieved_chunks_scores)
        hallucination_verdict = halluc_res["verdict"]
        hallucination_reason = halluc_res["reason"]

    latency_ms = int((time.time() - start_time) * 1000)

    # Format sources for response
    sources = [
        SourceChunk(doc_id=c.doc_id, score=s, text=c.text)
        for c, s in retrieved_chunks_scores
    ]

    # Log observability data
    log_query({
        "query": question,
        "input_flags": input_flags,
        "top_retrieval_score": top_score,
        "abstained": abstained,
        "latency_ms": latency_ms,
        "hallucination_verdict": hallucination_verdict,
        "hallucination_reason": hallucination_reason,
        "answer": actual_answer,
        "ip": client_ip
    })

    return QueryResponse(
        answer=actual_answer,
        sources=sources,
        abstained=abstained,
        hallucination_verdict=hallucination_verdict,
        latency_ms=latency_ms
    )
