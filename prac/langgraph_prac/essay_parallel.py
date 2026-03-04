import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START,END,StateGraph
import operator
from pydantic import BaseModel, Field
from typing import TypedDict,Annotated
load_dotenv()
model = ChatGoogleGenerativeAI(
    model = "gemini-3-flash-preview",
    temperature=0,
    google_api_key=os.getenv("gemini")
)
class EvaluationSchema(BaseModel):
    feedback: str = Field(description=" detailed Feedback for the essay")
    score:int = Field(description=" score out of 10",ge=0,le=10)

structured_model = model.with_structured_output(EvaluationSchema)


class EssayState(TypedDict):
    essay: str
    cot_feedback:str
    doa_feedback:str
    language_feedback:str
    overall_feedback:str
    individual_scores:Annotated[list[int],operator.add]
    average_score:float

def evaluate_language(state: EssayState):
    essay = state.get("essay")
    prompt = f"evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {essay}"
    output = structured_model.invoke(prompt)
    return {"language_feedback": output.feedback,"individual_scores": [output.score]}

def evaluate_analysis(state: EssayState):
    essay = state.get("essay")
    prompt = f"evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {essay}"
    output = structured_model.invoke(prompt)
    return {"doa_feedback": output.feedback,"individual_scores": [output.score]}

def evaluate_thought(state: EssayState):
    essay = state.get("essay")
    prompt = f"evaluate the content of thought of the following essay and provide a feedback and assign a score out of 10 \n {essay}"
    output = structured_model.invoke(prompt)
    return {"cot_feedback": output.feedback,"individual_scores": [output.score]}

def final_evaluation(state: EssayState):
    prompt = f"based on the following feedbacks create a summarized feedback \n language feedback - {state['language_feedback']} \n clarity of thought feedback -{state['cot_feedback']} \n depth of analysis feedback -{state["doa_feedback"]}"
    overall_feedback = model.invoke(prompt).content
    average_score = sum(state["individual_scores"])/len(state["individual_scores"])
    return {"overall_feedback": overall_feedback,"average_score": average_score}


# creating the graph
graph = StateGraph(EssayState)


# adding nodes to the graph
graph.add_node("evaluate_language",evaluate_language)
graph.add_node("evaluate_analysis",evaluate_analysis)
graph.add_node("evaluate_thought",evaluate_thought)
graph.add_node("final_evaluation",final_evaluation)


# adding edges to the graph

graph.add_edge(START,"evaluate_language")
graph.add_edge(START,"evaluate_analysis")
graph.add_edge(START,"evaluate_thought")

graph.add_edge("evaluate_language","final_evaluation")
graph.add_edge("evaluate_analysis","final_evaluation")
graph.add_edge("evaluate_thought","final_evaluation")

graph.add_edge("final_evaluation",END)

workflow = graph.compile()

essay_samples = ["""The integration of artificial intelligence into medical diagnostics represents a paradigm shift in modern healthcare, particularly in the analysis of complex imaging such as DICOM X-rays. Advanced deep learning architectures, such as DenseNet models, have demonstrated remarkable accuracy in detecting subtle pathologies that might evade the human eye. However, the true efficacy of AI in a clinical setting hinges not just on raw accuracy, but on interpretability. Physicians require transparent decision-making processes to trust AI-generated diagnoses. Techniques like Gradient-weighted Class Activation Mapping (Grad-CAM) are essential in bridging this gap. By visually highlighting the specific regions of a chest X-ray that led to a particular pathology prediction, visual mapping allows medical professionals to verify the model's reasoning. Ultimately, AI should not be viewed as a replacement for radiologists, but as a highly capable diagnostic assistant. The future of healthcare lies in this collaborative human-in-the-loop approach, where the processing power of AI is balanced with the contextual judgment and empathy of human doctors.""","""Artificial intelligence is becoming very important in hospitals today. Doctors are using AI to help them look at things like X-rays to figure out what is wrong with patients. This is really good because AI can look at pictures much faster than a human can. When a doctor has a lot of patients, the AI can save them a lot of time by pointing out the bad spots on the X-ray. It is also very accurate, which means fewer mistakes are made when diagnosing a sickness. But, doctors still need to be careful because AI is just a computer program and it can sometimes make errors. In conclusion, AI is a very helpful tool for the medical field. It helps doctors do their jobs faster and helps patients get their results quicker, which makes the hospital better for everyone.""","""AI is going to take over all the hospitals soon and maybe thats a bad thing. i saw a video where a robot was doing surgery and it makes you think what if it glitches out?? Doctors go to school for like 10 years but now a computer program can just do it in 5 seconds. Its crazy. my friend went to the hospital and they just used a machine to scan him and didn't even talk to him much. Also, AI is just going to cost to much money for normal people. If robots are doing everything we wont need nurses or radiolagists anymore they will all be fired by next year probably. so yeah AI is dangerous and we should probably stop using it before it gets too smart and takes all the jobs."""]
def scores(lst:list):
    s = []
    for essay in lst:
        initial_state ={"essay":essay}
        result = workflow.invoke(initial_state)
        s.append(result.get("average_score"))
    return s
x = scores(essay_samples)
print(x)

