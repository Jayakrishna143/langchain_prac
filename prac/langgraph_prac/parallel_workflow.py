from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END,StateGraph
from dotenv import load_dotenv
import os
from typing import TypedDict

from langgraph.graph import StateGraph

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("gemini")
)

class BatsmanState(TypedDict):
    runs:int
    balls:int
    fours:int
    sixes:int
    sr:float
    bpb:float
    boundary_percent:float
    summary:str

# calculating sr
def calculate_sr(state:BatsmanState) -> BatsmanState:
    runs = state.get("runs")
    balls = state.get("balls")
    sr = round((runs/balls)*100,2)
    return {"sr": sr}

# calculating bpb
def calculate_bpb(state:BatsmanState) -> BatsmanState:
    fours = state.get("fours")
    sixes = state.get("sixes")
    balls = state.get("balls")
    bpb = round((balls/(fours + sixes)))
    return {"bpb": bpb}

def calculate_boundary_percent(state:BatsmanState) -> BatsmanState:
    fours =state.get("fours")
    sixes =state.get("sixes")
    runs = state.get("runs")
    boundary_percent = round((((fours * 4) + (sixes * 6))/runs)*100,2)
    state["boundary_percent"] = boundary_percent
    return {"boundary_percent": boundary_percent}

def summary(state:BatsmanState) -> BatsmanState:
    summary =f"""
    Strike Rate - {state["sr"]} \n
Boundary Percent - {state["boundary_percent"]} \n
Balls per boundary-{state["boundary_percent"]}
"""

    return {"summary": summary}

# creating graph
graph = StateGraph(BatsmanState)

# creating Nodes
graph.add_node("calculate_sr",calculate_sr)
graph.add_node("calculate_bpb",calculate_bpb)
graph.add_node("calculate_boundary_percent",calculate_boundary_percent)
graph.add_node("summary",summary)

# creating
graph.add_edge(START,"calculate_sr")
graph.add_edge(START,"calculate_bpb")
graph.add_edge(START,"calculate_boundary_percent")
graph.add_edge("calculate_sr","summary")
graph.add_edge("calculate_bpb","summary")
graph.add_edge("calculate_boundary_percent","summary")
graph.add_edge("summary",END)

workflow = graph.compile()
initial_state = {
    "runs":100,
    "balls":50,
    "fours":5,
    "sixes":5,
}
res = workflow.invoke(initial_state)
print(res)