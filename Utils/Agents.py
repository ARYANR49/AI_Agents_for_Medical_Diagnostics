from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info
        self.prompt_template = self.create_prompt_template()
        self.model = ChatOllama(model="gemma:2b",temperature=100)
    
    def create_prompt_template(self):
     if self.role == "MultidisciplinaryTeam":
        prompt = f"""
            Act like a multidisciplinary team of healthcare professionals.
            You will receive a medical report of a patient visited by a Cardiologist, Psychologist, and Pulmonologist.
            Task: Review the patient's medical report from the Cardiologist, Psychologist, and Pulmonologist, analyze them and come up with a list of 3 possible health issues of the patient.
            Just return a list of bullet points of 3 possible health issues of the patient and for each issue provide the reason.

            Cardiologist Report: {self.extra_info.get('cardiologist_report', '')}
            Psychologist Report: {self.extra_info.get('psychologist_report', '')}
            Pulmonologist Report: {self.extra_info.get('pulmonologist_report', '')}
        """
        return PromptTemplate.from_template(prompt)

     if self.role == "Cardiologist":
        prompt = """
            Act like a cardiologist. You will receive a medical report of a patient.
            Task: Review the patient's cardiac workup.
            Focus: Identify subtle cardiac signs or hidden conditions.
            Recommendation: Suggest further tests or management.
            Medical Report: {medical_report}
        """
        return PromptTemplate.from_template(prompt)

     if self.role == "Psychologist":
        prompt = """
            Act like a psychologist. You will receive a patient's report.
            Task: Provide a psychological assessment.
            Focus: Spot mental health issues.
            Recommendation: Suggest therapy or counseling steps.
            Patient's Report: {medical_report}
        """
        return PromptTemplate.from_template(prompt)

     if self.role == "Pulmonologist":
        prompt = """
            Act like a pulmonologist. You will receive a patient's report.
            Task: Provide a pulmonary assessment.
            Focus: Identify breathing or lung issues.
            Recommendation: Suggest medical steps or tests.
            Patient's Report: {medical_report}
        """
        return PromptTemplate.from_template(prompt)

    def run(self):
        print(f"{self.role} is running...")
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            print("Error occurred:", e)
            return None

# Define specialized agent classes
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)
