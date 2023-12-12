import apis

class LucyAgent:
    """代表一个具有感知、大脑和行为能力的智能代理。"""
    def __init__(self, perception, brain, action):
        self.perception = perception
        self.brain = brain
        self.action = action

class AgentFSM:
    def __init__(self, initial_mood, initial_action_state, mood_list, emoji_list, action_state_list):
        # 状态
        self.mood = initial_mood
        self.action_state = initial_action_state

        # 可转移的静态列表和资源路径
        self.mood_list = mood_list
        self.emoji_list = emoji_list
        self.action_state_list = action_state_list

    def mood_transition(self, trigger, thought):
        new_mood = self.determine_mood_transition(trigger, thought)
        old_mood = self.mood
        if new_mood in self.mood_list:
            self.mood = new_mood
            print(f"心情从{old_mood}转移至：{self.mood}，因为发生了：{trigger}")
        else:
            print(f"收到不明确的心情：{new_mood}。没有发生心情转移。")

    def determine_mood_transition(self, trigger, thought):
        mood_transition_prompt = f"""
任务：推理角色的下一个心情应该是什么。心情可以是不变的。下面有一个例子给你作为参考，实际推理和例子无关。
例如：胡桃的原先心情是开心，胡桃在对话中了解到璃月管委会要拆迁往生堂。胡桃认为：为什么突然要拆迁往生堂？作为往生堂堂主，我应该进一步询问并考量。结合起来，应当输出的心情为：“惊讶”。
<<<
角色当前心情：{self.mood}
观察到的事件：{trigger}
角色的想法：{thought}
可能的心情列表：{self.mood_list}
>>>
现在请根据角色目前的心情状态,观察到的事件和角色想法的具体内容,从可能的心情列表中选择一个心情，例如“{self.mood_list[0]}”，不要进行额外的输出。
"""
        print(mood_transition_prompt)
        response = apis.request_chatgpt(mood_transition_prompt, temperature=0.5)
        print(f"输出心情为:{response}")
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

    def action_state_transition(self, trigger, thought):
        new_action_state = self.determine_action_state_transition(trigger, thought)
        old_action_state = self.action_state
        if new_action_state in self.action_state_list:
            self.action_state = new_action_state
            print(f"行动状态从{old_action_state}转移至：{self.action_state}。\n因为发生了：{trigger}")
        else:
            print(f"收到不明确的行动状态：{new_action_state}。没有发生行动状态转移。")

    def determine_action_state_transition(self, trigger, thought):
        # 构造中文提示文本
        action_state_transition_prompt = f"""
任务：推理角色的下一个行动应该是什么。下一个行动可以是不变的。下面有一个例子给你作为参考，实际推理和例子无关。
例如：胡桃当前正在策划往生堂相关活动。胡桃听到了hadi在打招呼,胡桃认为：hadi在给我打招呼，我应该赶快回应他。结合起来，你应当输出："回复问题和聊天"。
<<<
角色当前行动状态：{self.action_state}
观察到的事件：{trigger}
角色的想法：{thought}
可能的行动列表：{self.action_state_list}
>>>
现在请根据角色目前的行动状态,观察到的事件和角色想法的具体内容，从可能的行动列表中选择一个行动，例如“{self.action_state_list[0]}”，不要进行额外的输出。
"""
        print(action_state_transition_prompt)
        response = apis.request_chatgpt(action_state_transition_prompt, temperature=0.5)
        print(f"输出状态为:{response}")
        new_action_state = response.strip()
        return new_action_state
