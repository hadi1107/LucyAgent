from perception import Perception
from brain import Brain
from action import Action
import json

class LucyAgent:
    """代表一个具有感知、大脑和行为能力的智能代理。"""
    def __init__(self, perception, brain, action):
        self.perception = perception
        self.brain = brain
        self.action = action

if __name__ == "__main__":

    with open("../resource/hutao.json", "r", encoding="utf-8") as json_file:
        loaded_data = json.load(json_file)

    hutao = LucyAgent(perception=None, brain=Brain.from_json(loaded_data), action=Action())

