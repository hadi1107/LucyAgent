import apis

class LucyAgent:
    """代表一个具有感知、大脑和行为能力的智能代理。"""
    def __init__(self, perception, brain, action):
        self.perception = perception
        self.brain = brain
        self.action = action

class AgentFSM:
    def __init__(self, initial_mood, mood_list, emoji_list):
        self.mood = initial_mood
        self.mood_list = mood_list
        self.emoji_list = emoji_list

    def mood_transition(self, trigger, thought):
        new_mood = self.determine_mood_transition(trigger, thought)
        if new_mood in self.mood_list:
            self.mood = new_mood
            print(f"状态转移至：{self.mood}，因为发生了：{trigger}")
        else:
            print(f"收到不明确的状态：{new_mood}。没有发生状态转移。")

    def determine_mood_transition(self, trigger, thought):
        # 构造中文提示文本
        mood_transition_prompt = f"""
角色当前心情：{self.mood}
观察到的事件：{trigger}
角色的想法：{thought}
可能的心情列表：{self.mood_list}
任务：根据事件和想法的情感色彩，推理下一个心情，心情可以是不变的。
请从可能的心情列表中选择一个心情，例如“{self.mood_list[0]}”，不要进行额外输出。
"""
        print(mood_transition_prompt)
        response = apis.chatgpt(mood_transition_prompt)
        print(f"输出状态为:{response}")
        new_mood = response.strip()
        return new_mood

    def get_current_emoji(self):
        """根据当前心情状态获取对应的emoji表情符号。"""
        try:
            # 获取当前心情状态在mood_list中的索引
            mood_index = self.mood_list.index(self.mood)
            # 返回对应的emoji表情符号
            return self.emoji_list[mood_index]
        except ValueError:
            # 如果当前心情状态不在mood_list中，则返回None或其他默认值
            print(f"当前心情状态'{self.mood}'不在心情列表中。")
            return None
