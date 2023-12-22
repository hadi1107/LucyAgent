import apis

class AgentFSM:
    """
    AgentFSM类是一个代理有限状态机(Finite State Machine)的实现，用于管理和更新代理的心情和行动状态。

    类属性:
    - mood: 当前的心情状态 (字符串)
    - action_state: 当前的行动状态 (字符串)
    - mood_list: 可能的心情状态列表 (字符串列表)
    - emoji_list: 与心情状态对应的emoji表情符号列表 (字符串列表)
    - action_state_list: 可能的行动状态列表 (字符串列表)

    方法:
    - __init__(initial_mood, initial_action_state, mood_list, emoji_list, action_state_list): 初始化代理的心情和行动状态。
    - mood_transition(trigger, thought): 根据触发因素和思考内容更新当前的心情状态。
    - determine_mood_transition(trigger, thought): 根据触发事件和角色的思考来确定角色的新心情状态。
    - get_current_emoji(): 根据当前心情状态获取对应的emoji表情符号。
    - action_state_transition(trigger, thought): 根据触发事件和角色的思考来更新角色的行动状态。
    - determine_action_state_transition(trigger, thought): 根据触发事件和角色的思考来确定角色的新行动状态。

    使用这个类可以模拟一个角色的情绪和行动状态的变化，其中心情和行动状态的转变是通过与ChatGPT API的交互来确定的。每当有一个新的触发事件和角色的思考时，可以调用相应的方法来更新代理的状态。
    """
    def __init__(self, initial_mood, initial_action_state, mood_list, emoji_list, action_state_list):
        # 状态
        self.mood = initial_mood
        self.action_state = initial_action_state

        # 可转移的静态列表和资源路径
        self.mood_list = mood_list
        self.emoji_list = emoji_list
        self.action_state_list = action_state_list

    def mood_transition(self, trigger, thought):
        """
       根据触发因素和思考内容更新当前的心情状态。

       输入:
       - trigger: 触发心情转变的事件或情况 (字符串)
       - thought: 对触发事件的思考或解读 (字符串)

       输出:
       - 无返回值，但会打印心情转移的信息或错误信息
        """
        if not isinstance(trigger, str) or not isinstance(thought, str):
            print("错误：'trigger'和'thought'都应该是字符串类型。")
            return None

        if not hasattr(self, 'mood_list') or not isinstance(self.mood_list, list):
            print("错误：'mood_list'属性不存在或不是列表类型。")
            return None

        # 确定新的心情状态
        new_mood = self.determine_mood_transition(trigger, thought)
        # 获取当前心情状态
        old_mood = self.mood

        # 检查新心情是否在心情列表中
        if new_mood in self.mood_list:
            # 更新心情状态
            self.mood = new_mood
            # 打印心情转移信息
            print(f"心情从{old_mood}转移至：{self.mood}，因为发生了：{trigger}")
        else:
            # 打印错误信息
            print(f"收到不明确的心情：{new_mood}。没有发生心情转移。")

    def determine_mood_transition(self, trigger, thought):
        """
        根据触发事件和角色的思考来确定角色的新心情状态。

        输入:
        - trigger: 触发心情转变的事件或情况 (字符串)
        - thought: 角色对触发事件的思考或解读 (字符串)

        输出:
        - new_mood: 推理出的新心情状态 (字符串)
        """
        # 构建推理心情转移的提示信息
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
        # 打印提示信息
        print(mood_transition_prompt)

        # 发送请求并获取响应
        try:
            response = apis.request_chatgpt(mood_transition_prompt, temperature=0.5)
        except Exception as e:
            print(f"请求处理过程中发生错误：{e}")
            return None

        # 打印响应结果
        print(f"输出心情为:{response}")

        # 处理响应结果
        new_mood = response.strip()

        # 检查新心情是否在心情列表中
        if new_mood not in self.mood_list:
            print(f"错误：推理出的新心情'{new_mood}'不在可能的心情列表中。")
            return None

        # 返回新心情
        return new_mood

    def get_current_emoji(self):
        """
        根据当前心情状态获取对应的emoji表情符号。

        输入:
        - 无

        输出:
        - emoji: 对应心情的emoji符号 (字符串) 或 None
        """
        try:
            # 获取当前心情状态在mood_list中的索引
            mood_index = self.mood_list.index(self.mood)
            # 返回对应的emoji表情符号
            return self.emoji_list[mood_index]
        except ValueError:
            # 如果当前心情状态不在mood_list中，则返回None并打印错误信息
            print(f"当前心情状态'{self.mood}'不在心情列表中。")
            return None
        except AttributeError:
            # 如果mood_list或emoji_list属性不存在，则打印错误信息并返回None
            print("错误：'mood_list'或'emoji_list'属性不存在。")
            return None
        except IndexError:
            # 如果mood_index超出emoji_list的范围，则打印错误信息并返回None
            print(f"错误：心情索引'{mood_index}'超出emoji列表的范围。")
            return None

    def action_state_transition(self, trigger, thought):
        """
        根据触发事件和角色的思考来更新角色的行动状态。

        输入:
        - trigger: 触发行动状态转变的事件或情况 (字符串)
        - thought: 角色对触发事件的思考或解读 (字符串)

        输出:
        - 无 (直接更新类实例的状态)
        """
        new_action_state = self.determine_action_state_transition(trigger, thought)
        old_action_state = self.action_state
        if new_action_state in self.action_state_list:
            self.action_state = new_action_state
            print(f"行动状态从{old_action_state}转移至：{self.action_state}。\n因为发生了：{trigger}")
        else:
            print(f"收到不明确的行动状态：{new_action_state}。没有发生行动状态转移。")

    def determine_action_state_transition(self, trigger, thought):
        """
        根据触发事件和角色的思考来确定角色的新行动状态。

        输入:
        - trigger: 触发行动状态转变的事件或情况 (字符串)
        - thought: 角色对触发事件的思考或解读 (字符串)

        输出:
        - new_action_state: 推理得出的新行动状态 (字符串)
        """
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
