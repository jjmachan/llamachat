import streamlit as st
from llama_index import StorageContext, load_index_from_storage

st.set_page_config(page_title="LlamaChat")
st.title("LlamaChat")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you?"},
    ]

# load llama index

storage_context = StorageContext.from_defaults(persist_dir="../notebooks/storage")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine()


def generate_response(prompt):
    response = chat_engine.chat(prompt)
    return response.response


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
