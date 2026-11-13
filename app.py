import json
import os

import streamlit as st

from chatbot import VIDEO_URL, build_store, create_chain, fetch_transcript

st.set_page_config(page_title="YouTube video chatbot", page_icon="▶️")
st.title("Chat with the video")
st.markdown(f"Ask questions about [this YouTube video]({VIDEO_URL}). Answers cite transcript passages.")
with st.sidebar:
    st.header("Setup")
    key = st.text_input("OpenAI API key", type="password", help="Used only for this session.") or os.getenv("OPENAI_API_KEY", "")
    uploaded = st.file_uploader("Optional transcript JSON", type="json", help="List of {text, start, duration} caption objects, for the linked video.")
    load = st.button("Load video transcript", type="primary")
    st.caption("Loading sends transcript text to OpenAI for embeddings. Questions and retrieved excerpts are sent for answers. Embeddings are cached locally.")
if load:
    if not key:
        st.error("Enter an OpenAI API key or set OPENAI_API_KEY.")
    else:
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            with st.spinner("Loading captions and building the search index…"):
                segments = json.load(uploaded) if uploaded is not None else fetch_transcript()
                store = build_store(segments, embeddings=OpenAIEmbeddings(model="text-embedding-3-small", api_key=key))
                st.session_state.chain = create_chain(store, ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=key))
                st.session_state.messages = []
            st.success("Transcript indexed. Ask a question below.")
        except Exception as exc:
            st.error(f"Could not load the video ({type(exc).__name__}). Check your API key, network access, and caption availability, or upload transcript JSON.")
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
question = st.chat_input("Ask a standalone question about the video", disabled="chain" not in st.session_state)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching the transcript…"):
                result = st.session_state.chain.invoke(question)
            sources = "\n\n" + " · ".join(f"[{i}: {int(d.metadata['start'])}s]({d.metadata['source']})" for i, d in enumerate(result["documents"], 1))
            answer = result["answer"] + sources
            st.markdown(answer)
        st.session_state.messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    except Exception as exc:
        st.error(f"Answer request failed ({type(exc).__name__}). Check API access and try again.")
