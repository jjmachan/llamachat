from llama_index import StorageContext, load_index_from_storage
from llama_index.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from ragas.metrics import answer_relevancy, faithfulness

import streamlit as st

RAGAS_EVAL_MSG = "**Faithfulness:** {faithfulness:0.2f} **Answer Relevancy:** {answer_relevancy:0.2f} *(powered by [ragas](https://github.com/explodinggradients/ragas))*"  # noqa
LLAMAINDEX_EVAL_MSG = (
    "**Faithfulness:** {faithfulness} **Answer Relevancy:** {relevancy}"
)

st.set_page_config(page_title="LlamaChat")
st.title("LlamaChat")
with st.sidebar:
    use_ragas = st.toggle("ragas eval")
    use_llamaindex_eval = st.toggle("llamaindex eval")
    use_feedback = st.toggle("feedback", value=True)

# load llama index
storage_context = StorageContext.from_defaults(persist_dir="../notebooks/storage")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="condense_question")
service_context = index.service_context

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
        if use_ragas:
            # since -1 != assistant
            query = st.session_state.messages[-1]["content"]
            faithfulness = faithfulness.score_single(
                {"question": query, "contexts": context_txts, "answer": full_response}
            )
            relevancy = answer_relevancy.score_single(
                {"question": query, "answer": full_response}
            )
            st.success(
                RAGAS_EVAL_MSG.format(
                    faithfulness=faithfulness, answer_relevancy=relevancy
                )
            )
        if use_llamaindex_eval:
            # since -1 != assistant
            query = st.session_state.messages[-1]["content"]
            faithfulness = FaithfulnessEvaluator(service_context=service_context)
            relevancy = RelevancyEvaluator(service_context=service_context)
            faithfulness_resp = faithfulness.evaluate_response(
                response=streaming_response
            )
            relevancy_resp = relevancy.evaluate_response(
                query_str=query, contexts=context_txts, response=streaming_response
            )

            st.success(
                LLAMAINDEX_EVAL_MSG.format(
                    faithfulness="Pass" if faithfulness_resp.passing else "Fail",
                    relevancy="Pass" if relevancy_resp.passing else "Fail",
                )
            )
        if use_feedback:
            # using columns to center the buttons
            cols = st.columns([1, 1, 2, 1, 1])
            cols[1].button("👍")
            cols[3].button("👎")
        for i, c in enumerate(context_txts):
            expander = st.expander(f"Reference {i+1}")
            expander.markdown(c)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
