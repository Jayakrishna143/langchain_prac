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
class BlogState(TypedDict):
    title: str
    outline: str
    content :str

def create_outline(state: BlogState)-> BlogState:
    #fetch title
    title = state["title"]
    prompt = f"generate a detailed outline for the blog on topic {title}"
    res = model.invoke(prompt)
    outline = res.content
    state["outline"] = outline
    return state

def create_blog(state: BlogState)-> BlogState:
    title = state["title"]
    outline = state["outline"]
    prompt = f"write a detailed blog on the title {title}, using the following \n {outline}"
    content = model.invoke(prompt).content
    state["content"] = content
    return state

# creating graph
graph = StateGraph(BlogState)

# creatin nodes
graph.add_node("create_outline",create_outline)
graph.add_node("create_blog",create_blog)

# creating edges
graph.add_edge(START,"create_outline")
graph.add_edge("create_outline","create_blog")
graph.add_edge("create_blog",END)

workflow = graph.compile()

res = workflow.invoke({"title": "Cricket"})
print(res.get("content"))

