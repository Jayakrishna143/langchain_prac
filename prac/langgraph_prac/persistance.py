from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import os
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("gemini")
)

class JokeState(TypedDict):
    topic:str
    joke:str
    explanation:str

def generate_joke(state: JokeState):
    prompt = f"generate a joke on the topic {state["topic"]}"
    response = llm.invoke(prompt).content
    return {"joke":response}
def generate_explanation(state: JokeState):
    prompt =f"write a explanation for the joke -{state["joke"]}"
    response = llm.invoke(prompt).content
    return {"explanation":response}

graph = StateGraph(JokeState)
graph.add_node("generate_joke",generate_joke)
graph.add_node("generate_explanation",generate_explanation)

graph.add_edge(START,"generate_joke")
graph.add_edge("generate_joke","generate_explanation")
graph.add_edge("generate_explanation",END)

checkpoint = InMemorySaver()
workflow = graph.compile(checkpointer = checkpoint)
config1 = {"configurable":{"thread_id":"1"}}
answer = workflow.invoke({"topic":"pizza"},config = config1)
print(answer)
status = list(workflow.get_state_history(config1))
print(status)