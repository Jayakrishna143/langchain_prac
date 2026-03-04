from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="Conversational AI API")


# Setup LLM
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("gemini"))

prompt = ChatPromptTemplate.from_messages([
    ("system" , "you are a helpful and candid AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
])
chain = prompt |model| StrOutputParser()

store = {}
def get_session_history(session_id : str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key = "input",
    history_messages_key = "history",
)
class ChatRequest(BaseModel):
    session_id: str
    message: str
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = conversational_chain.invoke(
        {"input": request.message},
        config = {"configurable" : {"session_id" : request.session_id}}
    )
    return {"reply":response}