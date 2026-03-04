from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BMIstate(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    categroy: str

def calculate_bmi(state:BMIstate) -> BMIstate:
    weight_kg = state["weight_kg"]
    height_m = state["height_m"]
    bmi = weight_kg / (height_m ** 2)
    state["bmi"] = round(bmi, 2)
    return state
def label_bmi(state:BMIstate) -> BMIstate:
    bmi = state["bmi"]
    if bmi > 0 and bmi <= 18.5:
        state["categroy"] = "Underweight"
    elif 18.5 <= bmi < 25:
        state["categroy"] = "Normal"
    elif 25 <= bmi < 30:
        state["categroy"] = "Overweight"
    else:
        state["categroy"] = "Obese"
    return state

graph = StateGraph(BMIstate)

# add node to the graph
graph.add_node("calculate_bmi",calculate_bmi)
graph.add_node("label_bmi",label_bmi )
# add edges to the graph
graph.add_edge(START,"calculate_bmi")
graph.add_edge("calculate_bmi","label_bmi")
graph.add_edge("label_bmi",END)
#compile the grap
workflow  = graph.compile()
res = workflow.invoke({"weight_kg":63.0,"height_m":1.64})
print(res)

