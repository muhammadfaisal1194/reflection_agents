from typing import List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from chains import generation_chain, reflection_chain

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"

# Define the state schema
class GraphState(dict):
    messages: List[BaseMessage]

graph = StateGraph(GraphState)

def generate_node(state: GraphState):
    """Generate tweet"""
    response = generation_chain.invoke({
        "messages": state["messages"]
    })
    return {
        "messages": state["messages"] + [response]
    }

def reflect_node(state: GraphState):
    """Reflect and critique tweet"""
    response = reflection_chain.invoke({
        "messages": state["messages"]
    })
    return {
        "messages": state["messages"] + [HumanMessage(content=response.content)]
    }

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)
graph.set_entry_point(GENERATE)

def should_continue(state: GraphState):
    # Example: stop after more than 4 messages
    if len(state["messages"]) > 4:
        return END
    return REFLECT

graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

# Run the graph
initial_state = {"messages": [HumanMessage(content="Write a tweet about the latest tech trends.")]}
response = app.invoke(initial_state)

print("Final Response:", response["messages"][-1].content)
