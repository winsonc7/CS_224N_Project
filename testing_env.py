import argparse
import logging
import logging.config
from .config import *
from abc import ABC, abstractmethod
from agent_prompts import prompts
from agent import EvalAgent

logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.WARNING)

class Agent(ABC):
    @abstractmethod
    def get_response(self, latest_message: str):
        pass
    
class HumanInputAgent(Agent):
    def get_response(self):
        return input("Type your response: ")

class TestingFramework():
    def __init__(self, 
                 agent_prompt,
                 eval_prompt):
        self.agent_prompt = agent_prompt
        self.eval_prompt = eval_prompt
        self.user = HumanInputAgent()
        self.agent = EvalAgent()
        
    def run(self):
        user_message = ""
        agent_message = ""
        while True:
            # start with the buyer
            user_message = self.user.get_response()
            print("user_message: ", user_message)
            agent_message = self.agent.get_response(user_message)
            print("agent_message: ", agent_message)
            if user_message == "" or agent_message == "":
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent_prompt')
    parser.add_argument('--eval_prompt')

    args = parser.parse_args()

    framework = TestingFramework(args.agent_prompt, 
                                 args.eval_prompt)

    framework.run()