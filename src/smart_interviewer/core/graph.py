from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from smart_interviewer.core.transcriber import WhisperTranscriber
from smart_interviewer.core.types import InterviewState
from smart_interviewer.core.nodes import (
    node_wait_start,
    node_ask_question,
    node_wait_answer,
    node_transcribe,
    node_evaluate,
    node_wait_next,
    node_prepare_finish,
    node_wait_finish,
    node_finalize,
    route_after_transcribe,
    route_after_eval,
    route_after_next,
)


def build_interview_graph(*, transcriber: WhisperTranscriber):
    g = StateGraph(InterviewState)

    g.add_node("wait_start", node_wait_start)
    g.add_node("ask_question", node_ask_question)
    g.add_node("wait_answer", node_wait_answer)

    async def transcribe_node(state: InterviewState) -> InterviewState:
        return await node_transcribe(state, transcriber=transcriber)

    g.add_node("transcribe", transcribe_node)
    g.add_node("evaluate", node_evaluate)
    g.add_node("wait_next", node_wait_next)

    g.add_node("finish_prepare", node_prepare_finish)
    g.add_node("finish_wait", node_wait_finish)
    g.add_node("finish_finalize", node_finalize)

    g.add_edge(START, "wait_start")
    g.add_edge("wait_start", "ask_question")
    g.add_edge("ask_question", "wait_answer")
    g.add_edge("wait_answer", "transcribe")

    g.add_conditional_edges("transcribe", route_after_transcribe, {"wait_again": "wait_answer", "evaluate": "evaluate"})
    g.add_conditional_edges("evaluate", route_after_eval, {"wait_next": "wait_next", "wait_again": "wait_answer"})
    g.add_conditional_edges("wait_next", route_after_next, {"ask": "ask_question", "finish": "finish_prepare"})

    g.add_edge("finish_prepare", "finish_wait")
    g.add_edge("finish_wait", "finish_finalize")
    g.add_edge("finish_finalize", END)

    return g.compile(checkpointer=InMemorySaver())
