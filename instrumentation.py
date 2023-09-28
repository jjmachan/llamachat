from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from langsmith import Client
from langsmith.run_trees import RunTree

os.environ["LANGCHAIN_PROJECT"] = "pycon23"


@dataclass
class LangsmithSession:
    run: RunTree
    client: Client
    ennabled: bool = True

    @classmethod
    def start(cls, query: str, history: list[dict[str, str]], ennabled: bool = True):
        run = RunTree(
            name="query",
            run_type="chain",
            inputs={"query": query, "history": history},
            serialized={},
        )
        client = Client()
        return cls(run, client, ennabled)

    def stop(self, response: str):
        self.run.end(outputs={"response": response})
        if self.ennabled:
            res = self.run.post(exclude_child_runs=False)
            logging.debug(res)

    def retriever_start(self, query: str):
        self.retriever_run = self.run.create_child(
            name="retrieve", run_type="retriever", inputs={"query": query}
        )

    def retriever_stop(self, contexts: list[str]):
        if self.retriever_run is not None:
            self.retriever_run.end(outputs={"contexts": contexts})
        else:
            logging.error("retriever_run not started (retriever_run is None)")

    def ragas_scores(self, faithfulnes: float, relevancy: float):
        self.client.create_feedback(
            self.run.id,
            "ragas_faithfulnes",
            score=faithfulnes,
            feedback_source_type="model",
        )
        self.client.create_feedback(
            self.run.id,
            "ragas_relevancy",
            score=relevancy,
            feedback_source_type="model",
        )

    def llamaindex_scores(self, faithfulnes: str, relevancy: str):
        self.client.create_feedback(
            self.run.id,
            "llamaindex_faithfulnes",
            value=faithfulnes,
            feedback_source_type="model",
        )
        self.client.create_feedback(
            self.run.id,
            "llamaindex_relevancy",
            value=relevancy,
            feedback_source_type="model",
        )

    def feedback(self, liked: bool):
        ...
