from operator import add
from typing import TypedDict, Annotated, List
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.config import get_stream_writer
from langgraph.graph import START, END
from langgraph.checkpoint.memory import InMemorySaver

from src.llm.llm_factory import LLMFactory

nodes = ["supervisor", "travel", "joke", "couplet", "other"]

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add]
    type: str
llm = LLMFactory.create_llm()

def supervisor_node(state: State) -> State:
    print(">>> Supervisor Node")
    writer = get_stream_writer()
    writer({"node": ">>> supervisor_node"})
    # 根据用户的问题对问题进行分类，结果保存到type中
    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将任务分给其他Agent执行。
如果用户的问题是和旅游路线规划相关的，那就返回 travel 。
如果用户的问题是希望讲一个笑话，那就返回 joke 。
如果用户的问题是希望对一个对联，那就返回 couplet 。
如果是其他的问题，返回 other 。
除了这几个选项外，不要返回任何其他的内容。
"""
    prompts = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["messages"][0]}
    ]
    # 表示问题已经交由其他节点处理完成
    if 'type' in state:
        writer({"supervisor_step": "问题已经交由其他节点处理完成"})
        return {"type": END}
    else:
        response = llm.invoke(prompts)
        typeRes = response.content
        writer({"supervisor_step": f"问题分类结果: {typeRes}"})
        if typeRes in nodes:
            return {"type": typeRes}
        else:
            raise ValueError("type must be one of " + str(nodes) + ", but got " + typeRes)
    return state


def joke_node(state: State) -> State:
    print(">>> joke Node")
    writer = get_stream_writer()
    writer({"node": ">>> joke_node"})
    return {"messages": [HumanMessage(content="Joke Node")], "type": "joke"}


def couplet_node(state: State) -> State:
    print(">>> couplet Node")
    writer = get_stream_writer()
    writer({"node": ">>>couplet_node"})
    return {"messages": [HumanMessage(content="couplet Node")], "type": "couplet"}


def travel_node(state: State) -> State:
    print(">>> travel Node")
    writer = get_stream_writer()
    writer({"node": ">>>travel_node"})
    return {"messages": [HumanMessage(content="travel Node")], "type": "travel"}


def other_node(state: State) -> State:
    print(">>> other Node")
    writer = get_stream_writer()
    writer({"node": ">>> other_node"})
    return {"messages": [HumanMessage(content="我暂时无法回答这个问题")], "type": "other"}


builder = StateGraph(State)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("other_node", other_node)


def routing_fun(state: State):
    if state['type'] == "travel":
        return "travel_node"
    elif state['type'] == "joke":
        return "joke_node"
    elif state['type'] == "couplet":
        return "couplet_node"
    elif state['type'] == "other":
        return "other_node"
    elif state['type'] == END:
        return END
    return "other_node"

builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", routing_fun, ["travel_node", "joke_node", "couplet_node", "other_node", END])
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("couplet_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")
check_point = InMemorySaver()

graph = builder.compile(checkpointer=check_point)

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    for chunk in graph.stream({
        "messages": ["给我讲一个郭德纲的笑话"]
    }, config, stream_mode="custom"):
        print(chunk)
