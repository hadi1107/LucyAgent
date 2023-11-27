import time
import json
import numpy as np
import apis  # 用于与外部API进行交互。

MEMORY_LIMIT = 10

def cosine_similarity(embedding1, embedding2):
    """计算两个嵌入向量之间的余弦相似度。"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    print(dot_product / (norm1 * norm2))
    return dot_product / (norm1 * norm2)

class LucyAgent:
    """代表一个具有感知、大脑和行为能力的智能代理。"""
    def __init__(self, perception, brain, action):
        self.perception = perception
        self.brain = brain
        self.action = action

class Brain:
    """代表智能代理的大脑，负责记忆和知识处理。"""
    def __init__(self, name, seed_memory, language_style, basic_knowledge, current_state, memory_stream):
        self.name = name
        self.seed_memory = seed_memory
        self.language_style = language_style
        self.current_state = current_state
        self.basic_knowledge = basic_knowledge
        self.memory_stream = memory_stream
        self.memory_limit = MEMORY_LIMIT

    def to_json(self):
        """将大脑的状态转换为JSON格式的字典。"""
        return {
            "name": self.name,
            "seed_memory": self.seed_memory,
            "language_style": self.language_style,
            "current_state": self.current_state,
            "basic_knowledge": self.basic_knowledge,
            "memory_stream": self.memory_stream
        }

    @classmethod
    def from_json(cls, json_data):
        """从JSON格式的数据创建Brain实例。"""
        return cls(**json_data)

    def create_memory(self, input, output):
        """根据输入和输出创建一个记忆摘要，并返回记忆字典。"""
        summary_prompt = f"""
你的名称：{self.name}
你的初始记忆：{self.seed_memory}
事件描述：你刚刚参与了一些事件，需要添加到你的记忆流里。下方的<<<和>>>之间是一段描述你参与的事件的文本。
任务要求：基于事件内容，总结你从中获得的信息和你采取的行为。以第一人称视角提供一个简洁、陈述性的总结，字数不超过100字。
约束条件：不要修改事件的实际内容，只提供总结，不包含任何额外信息。
事件开始<<<
输入信息：{input}
行为输出：{output}
事件结束>>>
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
        # 检查记忆流是否达到上限
        if len(self.memory_stream) == self.memory_limit:
            self.summarize_memory()

    def summarize_memory(self):
        embeddings = np.array([memory['embedding'] for memory in self.memory_stream[:-1]])
        latest_embedding = self.memory_stream[-1]['embedding']

        # 计算最新记忆与其他记忆的相似度
        similarities = np.array([cosine_similarity(latest_embedding, emb) for emb in embeddings])
        # 获取最相似的五个记忆的索引
        top_indices = np.argsort(similarities)[-5:]
        descriptions_to_summarize = ""

        # 从最高索引开始删除，以避免改变较低索引的元素
        top_indices = sorted(top_indices, reverse=True)

        for i in top_indices:
            descriptions_to_summarize += self.memory_stream[i]['description'] + " "
            # 删除操作移动到循环外进行

        # 删除选中的记忆
        for i in top_indices:
            del self.memory_stream[i]

        # 创建总结记忆并将其添加到记忆流
        summary_prompt = f"""
你的名称：{self.name}
你的初始记忆：{self.seed_memory}
记忆总结任务：你将会看到一系列相似的记忆描述，它们在下方的分隔符<<<和>>>之间呈现。
任务要求：反思这些记忆，并从中提取关键信息。以第一人称视角编写一个语义层面更高级的总结，长度控制在100字以内。
约束条件：不要改变原始记忆的内容，只提供陈述性的总结。不要包含任何额外的信息或评论。
记忆描述开始<<<
{descriptions_to_summarize}
记忆描述结束>>>
        """
        print(summary_prompt)
        summary = apis.chatgpt(summary_prompt)  # 假设这个函数调用返回一个字符串摘要。
        print(summary)
        embedding_list = apis.embedding(summary)  # 假设这个函数调用返回一个嵌入向量。
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        memory = {
            "description": summary,
            "create_time": time_string,
            "embedding": embedding_list[0]
        }
        self.add_memory(memory)

    def del_memory(self, mode="single", index=0, query=""):
        """从记忆流中删除记忆。

        参数:
        - mode: 删除模式，可以是 "single"、"all" 或 "search"。
        - index: 当 mode 为 "single" 时，指定要删除的记忆的索引。
        - query: 当 mode 为 "search" 时，指定要搜索和删除的记忆的查询文本。
        """
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
        elif mode == "search":
            if query:
                # 搜索匹配的记忆
                memory = self.search_memory(query)
                if memory:
                    # 如果找到匹配的记忆，从记忆流中删除
                    try:
                        self.memory_stream.remove(memory)
                        print(f"删除了匹配查询\"{query}\"的记忆：\"{memory['description']}\"")
                    except ValueError:
                        print("未能删除记忆，可能已被删除。")
                else:
                    print("没有找到匹配的记忆。")
            else:
                print("查询文本为空，请提供有效的查询文本。")
        else:
            print("未知的删除模式。请使用 'single', 'all', 或 'search'。")

    def show_memory(self):
        """展示所有记忆的摘要、创建时间和嵌入向量的大小。"""
        if not self.memory_stream:
            print("没有记忆")
            return
        print(f"记忆条数：{len(self.memory_stream)}")
        for memory in self.memory_stream:
            description = memory["description"]
            create_time = memory["create_time"]
            embedding = memory["embedding"]
            print("---------------------------------------")
            print(f"记忆描述: {description}\n创建时间: {create_time}\n嵌入向量大小: {len(embedding)}")
            print("---------------------------------------")

    def search_memory(self, query_text):
        """搜索记忆的方法，只返回最相似的一个记忆项"""
        if self.memory_stream:
            # 计算查询文本的嵌入
            query_embedding = apis.embedding(query_text)[0]

            # 初始化最高相似度和相应的记忆项
            max_similarity = -1
            most_similar_memory = None

            # 遍历所有记忆项
            for memory in self.memory_stream:
                # 计算点积
                similarity = cosine_similarity(query_embedding, memory["embedding"])
                # 更新最高相似度和相应的记忆项
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_memory = memory

            return most_similar_memory

        return None

    def extract_knowledge(self,source):
        """从包含知识的文本中提取知识"""
        summary_prompt = f"""
你的名称：{self.name}
你的初始记忆：{self.seed_memory}
知识总结任务：你正在提取知识，下方的分隔符<<<和>>>之间的文本包含了需要被总结的知识点。
任务要求：阅读和理解文本内容，忽略任何不必要的缩进和编码问题。提供一个准确的、陈述性的知识点总结，长度限制在100字以内。
约束条件：不要改变知识点的原始内容，只提供总结。不要添加任何额外的信息。
知识点文本开始<<<
{source}
知识点文本结束>>>
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
        print(f"知识条数：{len(self.basic_knowledge)}")
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
            similarity = cosine_similarity(query_embedding,knowledge["embedding"])
            # 更新最高相似度和相应的知识项
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_knowledge = knowledge

        return most_similar_knowledge

    def chat(self,input,history):
        if history is None:
            history = []
        history.append(f"hadi:{input}")
        context = ""
        for chat in history:
            context = context + chat + "\n"
        memory = self.search_memory(input)
        knowledge = self.search_knowledge(input)
        prompt = f'''
你的名称：{self.name}
你的初始记忆：{self.seed_memory}
你的当前状态：{self.current_state}
对话任务：你正在进行对话，下方的分隔符<<<和>>>之间的文本包含了对话的上下文。
任务要求：你正在作为{self.name}进行回复。回复长度限制在100字以内。
约束条件：不要扮演其他角色，只作为{self.name}回复。不要添加任何额外的信息和格式。
辅助信息：你从记忆流中检索到了相关记忆”“”{memory["description"]}“”“你从你的知识库中检索到了相关知识：”“”{knowledge["text"]}“”“
上下文开始<<<
{self.language_style}
{context}
上下文结束>>>
'''
        print(prompt)
        response = apis.chatgpt(prompt)
        print(response)
        history.append(f"{response}")
        return response,history

if __name__ == "__main__":
    # 加载存储在JSON文件中的大脑状态。
    try:
        with open("../resource/hutao.json", "r", encoding="utf-8") as json_file:
            loaded_data = json.load(json_file)
    except FileNotFoundError:
        print("未找到指定的文件。")
        exit()

    hutao = LucyAgent(perception=None, brain=Brain.from_json(loaded_data), action=None)

    # 测试知识相关功能
    # text = "璃月，提瓦特大陆七国中的一国，位于大陆东方的富饶国度。其商港璃月港是全提瓦特大陆最繁华且吞吐量最大的港口"
    # hutao.brain.add_knowledge(text)
    # knowledge = hutao.brain.search_knowledge("绝区零二测情况怎么说？")
    # print(knowledge["text"])

    # query = "绝区零"
    # wiki_object = tools.get_wikipedia_text(query)  # 假设这个函数调用返回查询的响应。
    # print(wiki_object)
    # summary = hutao.brain.extract_knowledge(wiki_object)
    # hutao.brain.add_knowledge(summary)

    # memory = hutao.brain.create_memory("你好","你好，我是胡桃，要喝点什么吗？")
    # hutao.brain.add_memory(memory)
    # memory = hutao.brain.create_memory("你知道璃月吗？", "璃月，提瓦特大陆七国中的一国，位于大陆东方的富饶国度。其商港璃月港是全提瓦特大陆最繁华且吞吐量最大的港口")
    # hutao.brain.add_memory(memory)

    # hutao.brain.del_memory(mode="search",query="无关紧要的事")

    # 打印状态
    hutao.brain.show_memory()
    hutao.brain.show_knowledge()

    # history = []
    # while True:
    #     text = input("hadi:")
    #     if text == '0':
    #         break
    #     hutao.brain.chat(text,history)

    # 将更新后的大脑状态保存到JSON文件中。
    try:
        with open("../resource/hutao.json", "w", encoding="utf-8") as json_file:
            json.dump(hutao.brain.to_json(), json_file, indent=4, ensure_ascii=False)
    except IOError:
        print("无法写入文件。")