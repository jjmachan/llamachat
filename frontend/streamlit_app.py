import streamlit as st
from llama_index import StorageContext, load_index_from_storage

st.set_page_config(page_title="LlamaChat")
st.title("LlamaChat")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you?"},
    ]


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(
        message["role"], avatar="👨‍💻" if message["role"] == "user" else "🦙"
    ):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👨‍💻"):
        st.write(prompt)

# load llama index
storage_context = StorageContext.from_defaults(persist_dir="../notebooks/storage")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="condense_question")

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="🦙"):
        msg_placeholder = st.empty()
        full_response = ""
        streaming_response = chat_engine.stream_chat(prompt)
        for token in streaming_response.response_gen:
            full_response += token
            msg_placeholder.markdown(full_response + "▌")
        msg_placeholder.markdown(full_response)
        context_txts = [n.text for n in streaming_response.source_nodes]
        for i, c in enumerate(context_txts):
            expander = st.expander(f"Ref [{i}]")
            expander.markdown(c)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
