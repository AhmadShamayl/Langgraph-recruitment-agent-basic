from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from IPython.display import Image, display

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash" , temperature = 0.2)

class State(TypedDict):
    categorization : str
    experience : str
    skill_match : str
    response : str


builder = StateGraph(State)

def categorize_experience(state: State) -> State:
    print("\nFor categorization of Expereience of Candidate")
    prompt = ChatPromptTemplate(
    "Based on the provided application, get the candidate experience."
    "Respond with 'Entry-Level', 'Mid-Level' or 'Senior-Level'"
    "Application : {application}")

    chain = prompt | llm
    experience_level = chain.invoke({'application' : state['applciation']}).content
    print(f'experience level: {experience_level}')
    return {'experience_level' : experience_level}


def assessing_skills(state: State) -> State:
    print("\nFor Assessing the Skills of Candidate")
    prompt = ChatPromptTemplate(
    "Based on the job application for a Python Developer, assess the candidate's skillset"
    "Respond with 'Match' or  'No Match'"
    "Application : {application}")

    chain = prompt | llm
    skill_match = chain.invoke({'application' : state['applciation']}).content
    print(f'Assessing Skills: {skill_match}')
    return {'skill_match' : skill_match}

def schedule_hr_interview(state: State) -> State:
    print("\Scheduling the interview")
    return {'response' : "Candidate has been shortlisted for an HE Interview"}

def escalate_to_recruiter(state: State) -> State:
  print("Escalating to recruiter")
  return {"response" : "Candidate has senior-level experience but doesn't match job skills."}

def reject_application(state: State) -> State:
  print("Sending rejecting email")
  return {"response" : "Candidate doesn't meet JD and has been rejected."}


builder.add_node('categorize_experience', categorize_experience)
builder.add_node('assessing_skills', assessing_skills)
builder.add_node('schedule_hr_interview', schedule_hr_interview)
builder.add_node('escalate_to_recruiter', escalate_to_recruiter)
builder.add_node('reject_application', reject_application)

def route_app(state : State) -> str:
   if state['skill_match'] == 'Match':
      return "schedule_hr_interview"
   elif state['experience_level'] == 'Senior-Level':
      return "escalate_to_recruiter"
   else: 
      return "reject_application"


   
builder.add_edge(START, 'categorize_experience')
builder.add_edge('categorize_experience' , 'assessing_skills' )

builder.add_conditional_edges('assessing_skills' , route_app , {
   "schedule_hr_interview" : "schedule_hr_interview",
   "escalate_to_recruiter" : "escalate_to_recruiter",
   "reject_application" : "reject_application", 
   "categorize_experience" : "categorize_experience"
})

builder.add_edge("schedule_hr_interview" , END)
builder.add_edge("escalate_to_recruiter" , END)
builder.add_edge("reject_application" , END)

app = builder.compile()
display(Image(app.get_graph().draw_mermaid_png()))

with open ('graph.png' , 'wb') as f:
   f.write(app.get_graph().draw_mermaid_png())

