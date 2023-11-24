import time
import json
import apis  # 用于与外部API进行交互。
import numpy as np
import tools
import actions  # 用于执行特定的动作或功能。

class LucyAgent:
    """代表一个具有感知、大脑和行为能力的智能代理。"""
    def __init__(self, perception, brain, action):
        self.perception = perception
        self.brain = brain
        self.action = action

class Perception:
    """代表智能代理的感知能力。当前未实现具体功能。"""
    def __init__(self):
        pass

class Action:
    """代表智能代理的行为能力。当前未实现具体功能。"""
    def __init__(self):
        pass

class Brain:
    """代表智能代理的大脑，负责记忆和知识处理。"""
    def __init__(self, name, seed_memory, language_style, basic_knowledge, current_state, memory_stream):
        self.name = name
        self.seed_memory = seed_memory
        self.language_style = language_style
        self.basic_knowledge = basic_knowledge
        self.memory_stream = memory_stream
        self.current_state = current_state

    def to_json(self):
        """将大脑的状态转换为JSON格式的字典。"""
        return {
            "name": self.name,
            "seed_memory": self.seed_memory,
            "language_style": self.language_style,
            "basic_knowledge": self.basic_knowledge,
            "current_state": self.current_state,
            "memory_stream": self.memory_stream
        }

    @classmethod
    def from_json(cls, json_data):
        """从JSON格式的数据创建Brain实例。"""
        return cls(**json_data)

    def create_memory(self, input, output):
        """根据输入和输出创建一个记忆摘要，并返回记忆字典。"""
        summary_prompt = f"""
你是{self.name}。{self.seed_memory}
下方的分隔符<<<和>>>包含的内容为一段你参与的事件的文本描述。
要求：现在你要从事件中总结你获得了什么信息，进行了什么行为，以你的视角返回陈述性的总结(100字以内)。
限制：不要修改事件的内容，仅输出陈述性的总结，严禁输出任何额外内容。
<<<
你接受的相关信息输入是：{input}
你做出的相关行为输出是：{output}
>>>
"""
        summary = apis.chatgpt(summary_prompt)  # 假设这个函数调用返回一个字符串摘要。
        embedding_list = apis.embedding(summary)  # 假设这个函数调用返回一个嵌入向量。
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        memory = {
            "description": summary,
            "create_time": time_string,
            "embedding": embedding_list[0]
        }
        return memory

    def add_memory(self, memory):
        """将一个记忆添加到记忆流中。"""
        self.memory_stream.append(memory)

    def del_memory(self, mode = "single", index = 0 ):
        """从记忆流中删除指定索引的记忆。"""
        if mode == "single":
            try:
                description = self.memory_stream[index]["description"]
                del self.memory_stream[index]
                print(f"删除了\"{description}\"")
            except IndexError:
                print("提供的索引超出了记忆流的范围。")
        elif mode == "all":
            self.memory_stream.clear()  # 清空整个列表
            print("已清空所有记忆。")

    def show_memory(self):
        """展示所有记忆的摘要、创建时间和嵌入向量的大小。"""
        if not self.memory_stream:
            print("没有记忆")
            return
        for memory in self.memory_stream:
            description = memory["description"]
            create_time = memory["create_time"]
            embedding = memory["embedding"]
            print("---------------------------------------")
            print(f"记忆描述: {description}\n创建时间: {create_time}\n嵌入向量大小: {len(embedding)}")
            print("---------------------------------------")

    def search_memory(self):
        pass

    def extract_knowledge(self,source):
        summary_prompt = f"""
你是{self.name}。{self.seed_memory}
下方的分隔符<<<和>>>包含的内容为一段包含知识点的文本。
要求：现在你要阅读相关文本，忽略不必要的缩进和编码问题，然后陈述性地，准确地总结其中的知识点(100字以内)。
限制：不要修改知识点的内容，仅输出知识点的总结，严禁输出任何额外内容。
<<<
包含知识点的文本：{source}
>>>
"""
        summary = apis.chatgpt(summary_prompt)
        return summary

    def add_knowledge(self, text):
        """添加一个知识添加到知识库中。"""
        embedding_list = apis.embedding(text)
        knowledge = {
            "text": text,
            "embedding": embedding_list[0]
        }
        self.basic_knowledge.append(knowledge)

    def del_knowledge(self, mode="single", index=0):
        """从记忆流中删除指定索引的记忆。"""
        if mode == "single":
            try:
                text= self.basic_knowledge[index]["text"]
                del self.basic_knowledge[index]
                print(f"删除了\"{text}\"")
            except IndexError:
                print("提供的索引超出了知识库的范围。")
        elif mode == "all":
            self.basic_knowledge.clear()  # 清空整个列表
            print("已清空所有知识。")

    def show_knowledge(self):
        """展示所有知识。"""
        if not self.basic_knowledge:
            print("没有知识")
            return
        for knowledge in self.basic_knowledge:
            text = knowledge["text"]
            embedding = knowledge["embedding"]
            print("---------------------------------------")
            print(f"知识描述: {text}\n嵌入向量大小: {len(embedding)}")
            print("---------------------------------------")
    def search_knowledge(self, query_text):
        """搜索知识的方法，只返回最相似的一个知识项"""
        # 计算查询文本的嵌入
        query_embedding = apis.embedding(query_text)[0]

        # 初始化最高相似度和相应的知识项
        max_similarity = -1
        most_similar_knowledge = None

        # 遍历所有知识项
        for knowledge in self.basic_knowledge:
            # 计算点积
            dot_product = np.dot(query_embedding, knowledge["embedding"])
            # 计算范数
            norm_query = np.linalg.norm(query_embedding)
            norm_knowledge = np.linalg.norm(knowledge["embedding"])
            # 计算余弦相似度
            similarity = dot_product / (norm_query * norm_knowledge)
            # 更新最高相似度和相应的知识项
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_knowledge = knowledge

        return most_similar_knowledge

if __name__ == "__main__":
    # 加载存储在JSON文件中的大脑状态。
    try:
        with open("../resource/hutao.json", "r", encoding="utf-8") as json_file:
            loaded_data = json.load(json_file)
    except FileNotFoundError:
        print("未找到指定的文件。")
        exit()

    hutao = LucyAgent(perception=None, brain=Brain.from_json(loaded_data), action=None)

    # 测试记忆相关功能
    # query = ""
    # search_results = apis.bing_search(query)
    # memory = hutao.brain.create_memory(query, search_results)
    # hutao.brain.add_memory(memory)

    # 测试知识相关功能
    # text = "璃月，是游戏《原神》及其衍生作品中的国家，提瓦特大陆七国中的一国，位于大陆东方的富饶国度。其商港璃月港是全提瓦特大陆最繁华且吞吐量最大的港口"
    # hutao.brain.add_knowledge(text)

    # query = "绝区零"
    # wiki_object = tools.get_wikipedia_text(query)  # 假设这个函数调用返回查询的响应。
    # print(wiki_object)
    # summary = hutao.brain.extract_knowledge(wiki_object)
    # hutao.brain.add_knowledge(summary)

    # 打印状态
    hutao.brain.show_memory()
    hutao.brain.show_knowledge()
    knowledge = hutao.brain.search_knowledge("绝区零二测情况怎么说？")
    print(knowledge["text"])

    # 将更新后的大脑状态保存到JSON文件中。
    try:
        with open("../resource/hutao.json", "w", encoding="utf-8") as json_file:
            json.dump(hutao.brain.to_json(), json_file, indent=4, ensure_ascii=False)
    except IOError:
        print("无法写入文件。")

