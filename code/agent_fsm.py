import apis

class LucyAgent:
    """代表一个具有感知、大脑和行为能力的智能代理。"""
    def __init__(self, perception, brain, action, fsm):
        self.perception = perception
        self.brain = brain
        self.action = action
        self.fsm = fsm

class AgentFSM:
    def __init__(self, initial_state, state_list):
        self.state = initial_state
        self.state_list = state_list

    def transition(self, trigger):
        new_state = self.determine_state_with_chatgpt(trigger, self.state)
        if new_state in self.state_list:
            self.state = new_state
            print(f"状态转移至：{self.state}，因为发生了：{trigger}")
        else:
            print(f"从ChatGPT收到不明确的状态：{new_state}。没有发生状态转移。")

    def determine_state_with_chatgpt(self, trigger, current_state):
        # 构造中文提示文本
        prompt = f"""
你现在要驱动角色状态的转换。
当前状态:'{current_state}'。
角色观察到的状态是'{trigger}'。
可能的状态有:{self.state_list}
根据当前状态和可能的状态返回下一个状态，内容为状态的字符串，比如<<<空闲>>>。不要进行任何额外输出。
        """
        response = apis.chatgpt(prompt)
        print(f"输出状态为:{response}")
        new_state = response.strip()
        return new_state

if __name__ == "__main__":
    # 使用示例
    state_list = ['空闲','对话','做咖啡','学习','读书']
    agent = AgentFSM("空闲",state_list)

    # 触发状态转移
    agent.transition("客户想要一杯咖啡")
    agent.transition("客户离开了")
    agent.transition("需要学习下咖啡知识")