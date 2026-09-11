"""
frontend/app.py
---------------
Streamlit Chat UI that calls the FastAPI backend for Guarded RAG on contracts.
"""


import requests
import streamlit as st

# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ContractIQ", page_icon="⚖️", layout="wide")

st.title("⚖️ ContractIQ Chat")
st.markdown("Ask questions about your legal contracts. Protected by RAG Guardrails.")

# --- Sidebar ---
with st.sidebar:
    st.header("📊 System Evaluation")
    st.markdown("Retrieval Quality Benchmarks (Recall / Precision / F1)")
    
    try:
        res = requests.get(f"{API_URL}/eval-report")
        if res.status_code == 200:
            eval_data = res.json()
            # Convert dictionary into a flat table for Streamlit
            table_data = []
            for strategy, metrics in eval_data.items():
                table_data.append({
                    "Strategy": strategy,
                    "F1 Score": f"{metrics['f1']:.3f}",
                    "Precision": f"{metrics['precision']:.3f}",
                    "Recall": f"{metrics['recall']:.3f}",
                    "Latency (s)": f"{metrics['avg_latency_seconds']:.2f}"
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)
        else:
            st.error("Could not load evaluation report.")
    except requests.RequestException as e:
        st.error(f"Backend not reachable: {e}")

    st.markdown("---")
    st.markdown("Powered by **FastAPI** + **Streamlit**")

# --- Main Chat ---
# We keep it simple: just show the current interaction.

query = st.chat_input("Ask a question about the contracts (e.g. What is the governing law?)")

if query:
    st.chat_message("user").write(query)
    
    with st.chat_message("assistant"), st.spinner("Analyzing contracts..."):
        try:
                response = requests.post(f"{API_URL}/query", json={"question": query})
                
                if response.status_code == 429:
                    st.error("Rate limit exceeded. Please wait a moment.")
                elif response.status_code == 200:
                    data = response.json()
                    
                    # 1. Check Abstention
                    if data.get("abstained"):
                        st.warning("⚠️ **Abstained:** " + data.get("answer", "Insufficient context found in the provided contracts to answer this query."))
                    else:
                        # 2. Show Hallucination Verdict
                        verdict = data.get("hallucination_verdict", "UNKNOWN")
                        if verdict == "SUPPORTED":
                            st.success("✅ **Supported by Context**")
                        elif verdict == "PARTIALLY_SUPPORTED":
                            st.warning("⚠️ **Partially Supported:** The answer may contain external knowledge.")
                        else:
                            st.error("❌ **Unsupported:** The judge flagged this answer as a potential hallucination.")
                            
                        # 3. Show Answer
                        st.write(data.get("answer"))
                        
                        # 4. Show Sources
                        st.markdown(f"**Latency:** `{data.get('latency_ms')} ms`")
                        sources = data.get("sources", [])
                        if sources:
                            with st.expander(f"View {len(sources)} Source Chunks"):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(f"**Chunk {i}** (Score: `{src['score']:.4f}`)")
                                    st.caption(f"📄 `{src['doc_id']}`")
                                    st.info(src['text'])
                else:
                    st.error(f"Backend Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Is FastAPI running on port 8000?")
