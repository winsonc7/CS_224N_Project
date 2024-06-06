import os
from dotenv import load_dotenv
from openai import OpenAI
from abc import ABC, abstractmethod
from agent_prompts import prompts

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Agent(ABC):
    @abstractmethod
    def get_response(self, latest_message: str):
        pass

class EvalAgent(Agent):
    def create_gpt_response(self, instructions, input):
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": str(input)}
            ]
        )
        return completion.choices[0].message.content
    
    def get_response(self, latest_message: str):
        instructions = prompts.default_eval_agent_prompt()
        return self.create_gpt_response(self, instructions, latest_message)