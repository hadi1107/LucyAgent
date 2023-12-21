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

    hutao = LucyAgent(perception=Perception(), brain=Brain.from_json(loaded_data), action=Action())

    query = "你好呀"
    history = []
    response, history, thought = hutao.brain.cot_chat(query, history)

    print(hutao.brain.show_info())
    # 将更新后的大脑状态保存到JSON文件中。
    try:
        with open("../resource/hutao.json", "w", encoding="utf-8") as json_file:
            json.dump(hutao.brain.to_json(), json_file, indent=4, ensure_ascii=False)
    except IOError:
        print("无法写入文件。")

