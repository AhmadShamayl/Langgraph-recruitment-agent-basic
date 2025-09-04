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

<<<<<<< HEAD
=======

def run_cadidate_experience(application : str):
  results = app.invoke({'application' : application})
  return {
      'experience_level' : results['experience_level'] , # Corrected key from 'experience_level' to 'experience'
      'skill_match' : results['skill_match'],
      'response' : results['response']
  }

application_text = """ Dostar Ahmad
Lahore, Pakistan | dostarahmad8@gmail.com | 0309 6880449 | linkedin.com/in/dostar-ahmad | github.com/DostarAhmed4

SUMMARY
AI Engineer with 1+ years of hands-on experience in machine learning, deep learning, data science, and generative AI. Proven expertise in LLM fine-tuning, computer vision, and medical image analysis. Having developed data-driven AI solutions using Python and leading ML frameworks, delivering measurable impact across both commercial and healthcare domains.

EDUCATION
Information Technology University, BS in Electrical Engineering
Sept 2019 – Aug 2023
• Coursework: Machine Learning, Deep Learning, Computer Architecture, Internet of Things.

EXPERIENCE

Machine Learning Engineer, VulcanTech
Aug 2024 – Present
• Developed AI solutions across domains such as computer vision, natural language processing, medical imaging, and environmental forecasting.
• Fine-tuned and optimized deep learning models (YOLO, ResNet, Stable Diffusion, LLMs) for tasks like object detection, image segmentation, and generative art.
• Built robust data pipelines and performed extensive data preprocessing, augmentation, and feature engineering to enhance model performance and generalizability.
• Applied advanced machine learning techniques including time-series forecasting (ARIMA, Prophet), ensemble models (XGBoost, Random Forest).
• Integrated AI modules into real-time systems and interactive UIs using Python frameworks such as Streamlit, LangChain, Hugging Face, and SimpleITK.

Embedded Software Trainee Engineer, Rapid Silicon
Sep 2023 – Jul 2024
• Developed an automated testing infrastructure using Python and Pytest to streamline regression testing, increasing test coverage and efficiency.
• Performed functional validation of hardware prototypes by executing structured test cases across multiple system components.
• Strengthened embedded software development practices by applying clean coding principles and promoting maintainable codebases.
• Created and maintained clear documentation for software and hardware interfaces, improving long-term maintainability and team collaboration.

PROJECTS

AI-Powered Smart Surveillance System
• Built AI modules for real-time detection of fire, theft, aggression, and safety risks using YOLO-based object detection.
• Fine-tuned custom models for human classification, fall detection, and zone-based behavior monitoring.
• Created and augmented training datasets to optimize performance under varying camera and lighting conditions.
• Integrated bounding box inference and class-based confidence scoring for event tracking and alerting.
• Tools Used: Python, PyTorch, YOLOv8, OpenCV

AI Monogram Generation using Stable Diffusion
• Developed an AI system to generate artistic, personalized monograms using fine-tuned Stable Diffusion models.
• Explored full and LoRA fine-tuning approaches to enhance style fidelity and design uniqueness.
• Led model training and optimization, ensuring outputs were suitable for 3D printing and digital design.
• Built an automated generation pipeline enabling rapid production of custom name pendants.
• Tools Used: Python, Hugging Face, LoRA Fine-tuning, Stable Diffusers

Environmental Pollution Forecasting (EPD Project)
• Designed ETL pipelines for raw environmental data to enable structured analysis.
• Conducted EDA, spatial analysis, and time-series forecasting for pollutant trends using ARIMA, Prophet, XGBoost.
• Improved forecasting accuracy through feature engineering and model evaluation.
• Tools Used: Pandas, NumPy, scikit-learn, EDA, ARIMA, Prophet

Medical Image Registration
• Performed CT scan segmentation using TotalSegmentator and filtered skeletal structures.
• Implemented rigid registration via SimpleITK using Mattes Mutual Information and Euler3DTransform.
• Applied affine transformation for anatomical alignment of CT and label volumes.
• Tools Used: SimpleITK, TotalSegmentator, Euler3DTransform, 3D Slicer

Liver Tumor Segmentation with ResNet-34
• Preprocessed and augmented CT images to enhance training diversity and model robustness.
• Fine-tuned ResNet-34 for accurate segmentation of liver tumors.
• Assessed model effectiveness using Dice Coefficient, IoU, and accuracy metrics.
• Tools Used: PyTorch, ResNet-34, Transfer Learning, Dice Coefficient, IoU, Accuracy

TECHNOLOGIES
Languages: Python, Bash
Technologies & Frameworks: PyTorch, TensorFlow, scikit-learn, XGBoost, ARIMA, OpenCV, SimpleITK, Hugging Face Transformers
Tools & Platforms: VS Code, Jupyter Notebook, Google Colab, Git, GitHub, CUDA, Linux

CERTIFICATES AND AWARDS
NGIRI FYP Funding: Secured funding for NGIRI 2022–23 by Ignite for the Final Year Design Project titled "Autonomous Solar Panel Cleaning Robot".

Supervised Machine Learning (Stanford/Coursera): Completed the professional certification course "Supervised Machine Learning: Regression and Classification" by Stanford University on Coursera.

Open Source Software Development Specialization (Linux Foundation/Coursera): Completed a 4-course specialization on Open Source Development, Linux, and Git, authorized by The Linux Foundation via Coursera.

Linux Command Line Bootcamp (Udemy): Audited "The Linux Command Line Bootcamp: Beginner to Power User" course on Udemy."
"""
results = run_cadidate_experience(application_text)
print(f'\n\nComputed Results\n\nApplication: {application_text}\nExperience Level: {results['experience_level']}\n Skills Match: {results['skill_match']}\nResponse: {results['response']}')
>>>>>>> d2e51642 (setting things correctly)
