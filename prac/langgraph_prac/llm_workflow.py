import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("gemini")
)
# creating a state
class LLMstate(TypedDict):
    question: str
    answer: str

# create llm_qa function
def llm_qa(state: LLMstate) -> LLMstate:
    question = state["question"]
    prompt = (f"Answer the following question: {question}")
    answer = model.invoke(prompt)
    state["answer"] = answer.content
    return state


# creating graph
graph = StateGraph(LLMstate)

# add nodes
graph.add_node("llm_qa",llm_qa)

# add edges
graph.add_edge(START,"llm_qa")
graph.add_edge("llm_qa",END)

# COMPILE
work_flow = graph.compile()

res = work_flow.invoke({"question":"what is the capital of India"})
print(res.get("answer"))
