from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal,Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
import os
from dotenv import load_dotenv
from pydantic import BaseModel,Field
load_dotenv()

generator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # Or gemini-1.5-flash depending on your API access
    temperature=1,
    google_api_key=os.getenv("gemini")
)
evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # Or gemini-1.5-flash depending on your API access
    temperature=0,
    google_api_key=os.getenv("gemini")
)
optimizer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # Or gemini-1.5-flash depending on your API access
    temperature=0,
    google_api_key=os.getenv("gemini")
)
#
class TweetEvaluation(BaseModel):
    evaluation:Literal["approved","needs_improvement"] = Field(description="Final evaluation result.")
    feedback: str = Field(description="Feedback for the tweet.")

structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluation)

# state
class TweetState(TypedDict):
    topic:str
    tweet:str
    evaluation:Literal["approved","needs_improvement"]
    feedback:str
    iteration:int
    max_iteration:int

# generate tweet
def generate_tweet(state:TweetState):
    #prompt
    message=[
        SystemMessage(content = "you are a funny and clever Twitter/x influencer."),
        HumanMessage(content = """
        write a short, original, and hilarious tweet on the topic: "{state['topic']}".
        rules:
        - Do NOT use question-answer format.
        - max 289 characters.
        - use observational humor, irony, sarcasm, or cultural references.
        - use simple ,day to day english
        - this is version{state["itration"] +1}
        """)]
    #model
    response=generator_llm.invoke(message).content
    return {"tweet":response}

# evaluation node
def evaluate_tweet(state:TweetState):
    #prompt
    message=[
        SystemMessage(content = "you are a ruthless,no_laugh_given Twitter critic.You evaluate tweets based on humor, originality, virality, and tweet format"),
        HumanMessage(content = f"""
        evaluate the following tweets:
        Tweet :"{state['tweet']}"
        use the criteria below to evaluate the tweet:
        *important*: The tweet should be regarding the topic given "{state['topic']}".
        1.originality - is this fresh, or have you seen it a hundred times before?
        2.Humor - Did it genuinely make you smile,laugh,or chuckle?
        3.Punchiness -Is it short,sharp,and scroll-stopping?
        4.Virality Potential - Would people retweet or share it?
        5.Format - Is it a well- formed tweet (not a setup-punchline joke,not a Q&A joke,and under 28 characters)?
        
        Auto-reject if :
        It's written in question-answer format(w.g, "why did .." or "what happens when...")
        -It exceeds 280 characters
        - It reads like a traditional setup-punchline joke
        - dont end with generic, throwaway,or deflating lines that weaken the humor(eg.,"masterpieces)
        ### respond only in structured format:
        -evaluation:"approved" or "needs_improvement"
        - feedback:one paragraph explaining the strengths and weaknesses """)]

    response = structured_evaluator_llm.invoke(message)
    return {"evaluation":response.evaluation,"feedback":response.feedback}

def optimize_tweet(state:TweetState):
    #prompt
    messages = [
        SystemMessage(content = "You punch up tweets for virality and humor based on given feedback"),
        HumanMessage(content = f"""
        Improve the tweet based on ths feedback:
        "{state['feedback']}"
        Topic :"{state['topic']}"
        Original tweet:{state['tweet']} 
          Re-write it as  a short,viral _worthy tweet. Avoid Q&A style and stay under 280 characters""")]
    response = optimizer_llm.invoke(messages).content
    iteration = state["iteration"] +1
    return {"tweet":response,"iteration":iteration}
# decision code
def route_evaluation(state:TweetState):
    if state['evaluation'] == "approved" or state["iteration"] > state["max_iteration"]:
        return "approved"
    else:
        return "needs_improvement"


graph = StateGraph(TweetState)

#nodes
graph.add_node("generate",generate_tweet)
graph.add_node("evaluate",evaluate_tweet)
graph.add_node("optimize",optimize_tweet)

#edges
graph.add_edge(START,"generate")
graph.add_edge("generate","evaluate")
graph.add_conditional_edges("evaluate",route_evaluation,{"approved":END,"needs_improvement":"optimize"})
graph.add_edge("optimize","evaluate")

workflow = graph.compile()

initial_state = {
    "topic" :"Indian Railways",
    "iteration" :1,
    "max_iteration":5
}
result = workflow.invoke(initial_state)
print(result)