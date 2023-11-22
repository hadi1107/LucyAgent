import time
import json
import apis
import actions
class LucyAgent:
    def __init__(self, perception, brain, action):
        self.perception = perception
        self.brain = brain
        self.action = action

class Perception:
    def __init__(self):
        pass

class Brain:
    def __init__(self, name, seed_memory, language_style, basic_knowledge, current_state, memory_stream):
        self.name = name
        self.seed_memory = seed_memory
        self.language_style = language_style
        self.basic_knowledge = basic_knowledge
        self.current_state = current_state
        self.memory_stream = memory_stream

    # 将agent实例转换为json格式
    def to_json(self):
        return {
            "name": self.name,
            "seed_memory": self.seed_memory,
            "language_style": self.language_style,
            "basic_knowledge":self.basic_knowledge,
            "current_state": self.current_state,
            "memory_stream": self.memory_stream
        }

    # 类方法，从JSON数据创建Agent实例
    @classmethod
    def from_json(cls, json_data):
        return cls(
            json_data["name"],
            json_data["seed_memory"],
            json_data["language_style"],
            json_data["basic_knowledge"],
            json_data["current_state"],
            json_data["memory_stream"]
        )

    def create_memory(self, input, output):
        memory = {}
        summary_prompt = f"""
你是{self.name}。{self.seed_memory}
下方分隔符<<<和>>>包含的内容为你刚刚参与的事件。
现在你要从事件中总结你获得了什么信息，进行了什么行为，以你的视角返回陈述性的总结(100字以内)。
要求：不要修改事件的内容，仅输出陈述性的总结，严禁输出任何额外内容。
<<<
你接受的相关信息输入是：{input}
你做出的相关行为输出是：{output}
>>>
"""
        print(summary_prompt)
        summary = apis.chatgpt(summary_prompt)
        embedding = apis.embedding(summary)
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        memory["description"] = summary
        memory["create_time"] = time_string
        memory["embedding"] = embedding[0]
        print(memory)
        return memory

    def add_memory(self,memory):
        self.memory_stream.append(memory)

    def del_memory(self,index):
        if self.memory_stream:
            description = self.memory_stream[index]["description"]
            print(f"删除了\"{description}\"")
            del self.memory_stream[index]
        else:
            print("没有记忆")

    def show_memory(self):
        for memory in self.memory_stream:
            description = memory["description"]
            create_time = memory["create_time"]
            embedding = memory["embedding"]
            print(f"description:{description},create_time:{create_time},embedding_size:{len(embedding)}\n")


    def search_knowledge(self):
        pass
class Action:
    def __init__(self):
        pass

if __name__ == "__main__":
    with open("../resource/hutao.json", "r", encoding = "utf-8") as json_file:
        loaded_data = json.load(json_file)

    hutao_brain = Brain.from_json(loaded_data)
    hutao = LucyAgent(perception = None,brain = hutao_brain, action = None)

    # function功能测试
    query = "搜索八重樱的人物背景"
    response = actions.chatgpt_function(query)
    print(response)
    memory = hutao.brain.create_memory(query,response)
    hutao.brain.add_memory(memory)

    # hutao.brain.del_memory(0)
    with open("../resource/hutao.json", "w", encoding="utf-8") as json_file:
        json.dump(hutao.brain.to_json(), json_file, indent=4, ensure_ascii=False)

    hutao.brain.show_memory()