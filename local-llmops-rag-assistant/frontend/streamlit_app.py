import os
from uuid import uuid4

import httpx
import streamlit as st

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Local LLMOps RAG Assistant", layout="wide")
st.title("Local LLMOps RAG Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = "local-user"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace_id" not in st.session_state:
    st.session_state.last_trace_id = None

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or Markdown", type=["pdf", "txt", "md"])
    if uploaded_file and st.button("Index Document", use_container_width=True):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        with httpx.Client(timeout=90) as client:
            response = client.post(f"{BACKEND_BASE_URL}/upload", files=files)
        if response.status_code == 200:
            payload = response.json()
            st.success(f"Indexed {payload['chunk_count']} chunks from {payload['source_file']}")
        else:
            st.error(f"Upload failed: {response.text}")

    st.header("Incident Simulation")
    incident = st.selectbox(
        "Incident Type",
        ["openai_timeout", "vector_db_failure", "low_retrieval_score", "prompt_injection", "high_token_usage"],
    )
    if st.button("Simulate Incident", use_container_width=True):
        with httpx.Client(timeout=30) as client:
            response = client.post(f"{BACKEND_BASE_URL}/simulate-incident", json={"incident_type": incident})
        if response.status_code == 200:
            st.info(response.json())
        else:
            st.error(response.text)

st.subheader("Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            if message.get("citations"):
                st.caption("Citations")
                for item in message["citations"]:
                    st.write(f"- {item['source_file']} | chunk={item['chunk_id']} | score={item['score']:.3f}")
            if message.get("latency_ms") is not None:
                st.caption(f"Latency: {message['latency_ms']} ms | Guardrail: {message.get('guardrail_status', 'unknown')}")
            if message.get("trace_url"):
                st.markdown(f"LangSmith Trace: {message['trace_url']}")

question = st.chat_input("Ask a question about uploaded documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    payload = {
        "user_id": st.session_state.user_id,
        "session_id": st.session_state.session_id,
        "question": question,
    }
    with httpx.Client(timeout=90) as client:
        response = client.post(f"{BACKEND_BASE_URL}/chat", json=payload)

    if response.status_code == 200:
        body = response.json()
        st.session_state.last_trace_id = body.get("trace_id")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": body["answer"],
                "citations": body.get("citations", []),
                "latency_ms": body.get("latency_ms"),
                "guardrail_status": body.get("guardrail_status"),
                "trace_url": body.get("trace_url"),
            }
        )
    else:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Request failed: {response.text}",
                "citations": [],
            }
        )
    st.rerun()

st.subheader("Feedback")
col1, col2 = st.columns(2)
if col1.button("Thumbs Up", use_container_width=True):
    if st.session_state.messages and len(st.session_state.messages) >= 2:
        with httpx.Client(timeout=30) as client:
            client.post(
                f"{BACKEND_BASE_URL}/feedback",
                json={
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                    "question": st.session_state.messages[-2]["content"],
                    "answer": st.session_state.messages[-1]["content"],
                    "feedback": "thumbs_up",
                    "trace_id": st.session_state.last_trace_id,
                },
            )
        st.success("Feedback submitted")

if col2.button("Thumbs Down", use_container_width=True):
    if st.session_state.messages and len(st.session_state.messages) >= 2:
        with httpx.Client(timeout=30) as client:
            client.post(
                f"{BACKEND_BASE_URL}/feedback",
                json={
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                    "question": st.session_state.messages[-2]["content"],
                    "answer": st.session_state.messages[-1]["content"],
                    "feedback": "thumbs_down",
                    "trace_id": st.session_state.last_trace_id,
                },
            )
        st.success("Feedback submitted")
