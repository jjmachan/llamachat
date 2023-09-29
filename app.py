import asyncio

import streamlit as st
from llama_index import StorageContext, load_index_from_storage
from llama_index.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from ragas.metrics import answer_relevancy, faithfulness

RAGAS_EVAL_MSG = "**Faithfulness:** {faithfulness:0.2f} **Answer Relevancy:** {answer_relevancy:0.2f} *(powered by [ragas](https://github.com/explodinggradients/ragas))*"  # noqa
LLAMAINDEX_EVAL_MSG = (
    "**Faithfulness:** {faithfulness} **Answer Relevancy:** {relevancy}"
)
LLAMA_INTRO_MSG = """\
hey there 👋

I'm llama, your go to for everything at this year's [PyCon India Conference](https://in.pycon.org/2023/). You can ask me everything about the sessions happening this weekend. I'm still learning so please be patient with me. 

Here are some questions I have helped others with:
- What is the session on llamaindex about?
- Is there a session on mojo?
- list the sessions where I can learn about telemetry
"""


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


loop = get_or_create_eventloop()
asyncio.set_event_loop(loop)

st.set_page_config(page_title="LlamaChat")
st.title("LlamaChat")

use_langsmith = True
with st.sidebar:
    use_ragas = st.toggle("ragas eval")
    use_llamaindex_eval = st.toggle("llamaindex eval")
    use_feedback = st.toggle("feedback", value=True)

    # info
    st.markdown(
        "Facing some problems? [Open an issue](https://github.com/jjmachan/llamachat/issues/new)"
    )
    st.markdown(
        "Want to learn more or contribute? [Check out the repo](https://github.com/jjmachan/llamachat)"
    )


@st.cache_resource
def load_index(path: str):
    storage_context = StorageContext.from_defaults(persist_dir=path)
    index = load_index_from_storage(storage_context)
    chat_engine = index.as_chat_engine(chat_mode="condense_question")
    service_context = index.service_context

    return chat_engine, service_context


# load llama index
chat_engine, service_context = load_index("./notebooks/pycon23/storage")


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": LLAMA_INTRO_MSG}]


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
    assert prompt is not None

    # now start generating response
    with st.chat_message("assistant", avatar="🦙"):
        msg_placeholder = st.empty()
        full_response = ""

        # stream response
        streaming_response = chat_engine.stream_chat(prompt)
        for token in streaming_response.response_gen:
            full_response += token
            msg_placeholder.markdown(full_response + "▌")
        msg_placeholder.markdown(full_response)

        # retriever_stop
        context_txts = [n.text for n in streaming_response.source_nodes]

        # feedback and evaluation
        if use_ragas:
            with st.status("calculating ragas scores"):
                faithfulness = faithfulness.score_single(
                    {
                        "question": prompt,
                        "contexts": context_txts,
                        "answer": full_response,
                    }
                )
                relevancy = answer_relevancy.score_single(
                    {"question": prompt, "answer": full_response}
                )
            st.success(
                RAGAS_EVAL_MSG.format(
                    faithfulness=faithfulness, answer_relevancy=relevancy
                )
            )

        if use_llamaindex_eval:
            with st.status("running llamaindex evaluation"):
                faithfulness = FaithfulnessEvaluator(service_context=service_context)
                relevancy = RelevancyEvaluator(service_context=service_context)
                faithfulness_resp = faithfulness.evaluate_response(
                    response=streaming_response
                )

            st.success(
                LLAMAINDEX_EVAL_MSG.format(
                    faithfulness="Pass" if faithfulness_resp.passing else "Fail",
                    relevancy="Pass",
                )
            )

        # get user feedback
        if use_feedback:
            # using columns to center the buttons
            cols = st.columns([1, 1, 2, 1, 1])
            cols[1].button("👍")
            cols[3].button("👎")

        # show references
        for i, c in enumerate(context_txts):
            expander = st.expander(f"Reference {i+1}")
            expander.markdown(c)

    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
