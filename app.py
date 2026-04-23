import streamlit as st

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource(show_spinner=False)
def load_rag():
    import rag
    return rag

if "rag_loaded" not in st.session_state:
    with st.status("Starting up...", expanded=True) as status:
        st.write("Loading vector store...")
        rag_module = load_rag()
        status.update(label="Ready", state="complete", expanded=False)
    st.session_state.rag_loaded = True
else:
    rag_module = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(rag_module.ask(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()