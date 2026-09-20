"""Streamlit chat interface for the FDP RAG chatbot.

Run with:  streamlit run app.py
"""

import streamlit as st

from rag import answer, get_llm, load_retriever

st.set_page_config(page_title="FDP AI Assistant", page_icon="🎓")
st.title("🎓 FDP AI Assistant")
st.caption("Ask anything about the Faculty Development Programme on AI.")


@st.cache_resource(show_spinner="Loading knowledge base...")
def init():
    return load_retriever(), get_llm()


try:
    retriever, llm = init()
except (FileNotFoundError, EnvironmentError) as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

if question := st.chat_input("Ask a question about the FDP..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, docs = answer(question, retriever, llm)
        st.markdown(response)

        sources = sorted({d.metadata.get("source", "unknown") for d in docs})
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
