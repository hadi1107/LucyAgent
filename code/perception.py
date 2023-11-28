class LucyAgent:
    """代表一个具有感知、大脑和行为能力的智能代理。"""
    def __init__(self, perception, brain, action, fsm):
        self.perception = perception
        self.brain = brain
        self.action = action
        self.fsm = fsm
class Perception:
    """代表智能代理的感知能力。当前未实现具体功能。"""
    def __init__(self):
        pass
