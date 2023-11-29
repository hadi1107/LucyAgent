import apis
import json

class LucyAgent:
    """代表一个具有感知、大脑和行为能力的智能代理。"""
    def __init__(self, perception, brain, action, fsm):
        self.perception = perception
        self.brain = brain
        self.action = action
        self.fsm = fsm

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
        prompt = f"""
你现在要驱动角色状态的转换。
角色当前状态:{self.mood}
角色观察到的事件是:{trigger}
角色的想法是：{thought}
可能的状态有:{self.mood_list}
根据当前状态和可能的状态返回下一个状态，内容为状态的字符串，比如<<<{self.mood_list[0]}>>>。不要进行任何额外输出。
        """
        print(prompt)
        response = apis.chatgpt(prompt)
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

if __name__ == "__main__":
    # 使用示例
    mood_list = ['惊讶','恼火','为难','兴奋','开心']
    emoji_list = ["../resource/pictures/hutao_jingya.webp","../resource/pictures/hutao_naohuo.webp",
                  "../resource/pictures/hutao_weinan.webp","../resource/pictures/hutao_xingfen.webp",
                  "../resource/pictures/hutao_yao.webp"]
    fsm = AgentFSM("为难",mood_list,emoji_list)

    from brain import Brain

    with open("../resource/hutao.json", "r", encoding="utf-8") as json_file:
        loaded_data = json.load(json_file)

    hutao = LucyAgent(perception=None, brain=Brain.from_json(loaded_data), action=None, fsm = fsm)

    emoji_file_path = hutao.fsm.get_current_emoji()
    print(emoji_file_path)
