import time
import json
import numpy as np
import logging

import apis
from agent_fsm import AgentFSM

MEMORY_LIMIT = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='../agent.log',  # 指定日志文件的名称
    filemode='w',  # 'a' 表示追加模式，如果每次运行时都创建新文件，可以使用 'w'
)
logger = logging.getLogger(__name__)

def cosine_similarity(embedding1, embedding2):
    """计算两个嵌入向量之间的余弦相似度。"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    print(f"cosine_similarity:{dot_product / (norm1 * norm2)}")
    return dot_product / (norm1 * norm2)

class LucyAgent:
    """代表一个具有感知、大脑和行为能力的智能代理。"""
    def __init__(self, perception, brain, action):
        self.perception = perception
        self.brain = brain
        self.action = action

class Brain:
    """代表智能代理的大脑，负责记忆和知识处理。"""
    def __init__(self, name, seed_memory, language_style, basic_knowledge, memory_stream, mood_list, emoji_list):
        self.name = name
        self.seed_memory = seed_memory
        self.language_style = language_style
        self.basic_knowledge = basic_knowledge

        self.memory_stream = memory_stream
        self.memory_limit = MEMORY_LIMIT

        self.mood_list = mood_list
        self.emoji_list = emoji_list
        self.fsm = AgentFSM("开心",mood_list,emoji_list)

    def to_json(self):
        """将大脑的状态转换为JSON格式的字典。"""
        return {
            "name": self.name,
            "seed_memory": self.seed_memory,
            "language_style": self.language_style,
            "basic_knowledge": self.basic_knowledge,
            "memory_stream": self.memory_stream,
            "mood_list":self.mood_list,
            "emoji_list": self.emoji_list
        }

    @classmethod
    def from_json(cls, json_data):
        """从JSON格式的数据创建Brain实例。"""
        logger.info(f"按照{json_data}对Brain模块进行了初始化")
        return cls(**json_data)

    def show_info(self):
        # 创建一个描述大脑状态的字符串
        info = f"Brain Info:\n"
        info += f"Name: {self.name}\n\n"
        info += f"Seed Memory: {self.seed_memory}\n\n"
        info += f"Language Style: \n{self.language_style}\n\n"
        info += f"Memory Limit: {self.memory_limit}\n\n"
        info += f"mood_list: {self.mood_list}\n\n"
        info += f"emoji_list: {self.emoji_list}\n\n"
        info += f"mood: {self.fsm.mood}\n\n"
        info += self.show_knowledge()
        info += self.show_memory()
        logger.info(f"打印了Brain模块的相关信息：{info}")
        return info

    def create_memory(self, input, output):
        """根据输入和输出创建一个记忆摘要，并返回记忆字典。"""
        summary_prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
任务：从事件描述中总结信息和行为。
字数限制：不超过100字。
事件描述：
<<<
输入信息：{input}
行为输出：{output}
>>>
请提供一个简洁、陈述性的总结，不要修改事件的实际内容或添加额外信息。
"""
        summary = apis.chatgpt(summary_prompt)  # 假设这个函数调用返回一个字符串摘要。
        embedding_list = apis.embedding(summary)  # 假设这个函数调用返回一个嵌入向量。
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        memory = {
            "description": summary,
            "create_time": time_string,
            "embedding": embedding_list[0]
        }
        logger.info(f"从{input}和{output}中创建了新记忆：{summary}")
        return memory

    def add_memory(self, memory):
        """将一个记忆添加到记忆流中。"""
        self.memory_stream.append(memory)
        description = memory["description"]
        logger.info(f"添加了新记忆：{description}")
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

        # 删除选中的记忆
        for i in top_indices:
            description = self.memory_stream[i]["description"]
            del self.memory_stream[i]
            logger.info(f"因为需要总结而删除了记忆：{description}")

        # 创建总结记忆并将其添加到记忆流
        summary_prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
任务：提取并总结一系列相似记忆描述中的关键信息。
字数限制：不超过100字。
记忆描述：
<<<
{descriptions_to_summarize}
>>>
请以第一人称视角编写一个高语义层次的总结，不要改变原始记忆的内容或添加额外信息。
"""
        print(summary_prompt)
        summary = apis.chatgpt(summary_prompt)  # 假设这个函数调用返回一个字符串摘要。
        print(summary)
        logger.info(f"从记忆子集：{descriptions_to_summarize}\n总结了相关记忆：{summary}")
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
                logger.info(f"删除了记忆:{description}")
                print(f"删除了记忆:{description}")
            except IndexError:
                print("提供的索引超出了记忆流的范围。")
        elif mode == "all":
            self.memory_stream.clear()  # 清空整个列表
            logger.info("已清空所有记忆。")
            print("已清空所有记忆。")
        elif mode == "search":
            if query:
                # 搜索匹配的记忆
                query_embedding = apis.embedding(query)[0]
                memory = self.search_memory(query_embedding)
                if memory:
                    # 如果找到匹配的记忆，从记忆流中删除
                    try:
                        self.memory_stream.remove(memory)
                        logger.info(f"删除了匹配查询\"{query}\"的记忆：\"{memory['description']}\"")
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
        memory_str = ""
        if not self.memory_stream:
            memory_str = "没有记忆单元"
            print(memory_str)
            return memory_str

        memory_count = len(self.memory_stream)
        memory_str += f"记忆条数：{memory_count}\n"
        memory_str += f"{'-' * 40}\n"  # 添加分隔线

        for idx, memory in enumerate(self.memory_stream, 1):
            description = memory["description"]
            create_time = memory["create_time"]
            embedding_size = len(memory["embedding"])
            memory_str += (
                f"记忆 #{idx}\n"
                f"描述: {description}\n"
                f"创建时间: {create_time}\n"
                f"嵌入向量大小: {embedding_size}\n"
                f"{'-' * 40}\n"
            )

        print(memory_str)
        return memory_str

    def search_memory(self, query_embedding):
        """搜索记忆的方法，只返回最相似的一个记忆项"""
        # 如果没有记忆项，直接返回提示信息
        if not self.memory_stream:
            return {
                "description": "没有记忆单元",
                "create_time": "没有记忆单元",
                "embedding": []
            }

        # 初始化最高相似度和相应的记忆项
        max_similarity = -1
        most_similar_memory = None

        # 遍历所有记忆项
        for memory in self.memory_stream:
            # 计算相似度
            similarity = cosine_similarity(query_embedding, memory["embedding"])
            # 更新最高相似度和相应的记忆项
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_memory = memory

        description = most_similar_memory["description"]
        logger.info(f"找到了相关记忆：{description}")
        return most_similar_memory

    def extract_knowledge(self,source):
        """从包含知识的文本中提取知识"""
        summary_prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
任务：总结文本中包含的知识点。
字数限制：不超过100字。
知识点文本：
<<<
{source}
>>>
请提供一个准确的、陈述性的知识点总结，不要改变原始内容或添加额外信息。
"""
        summary = apis.chatgpt(summary_prompt)
        logger.info(f"从{source}提取了知识点：{summary}")
        return summary

    def add_knowledge(self, text):
        """添加单个知识添加到知识库中。"""
        embedding_list = apis.embedding(text)
        knowledge = {
            "text": text,
            "embedding": embedding_list[0]
        }
        self.basic_knowledge.append(knowledge)
        logger.info(f"添加了知识：{text}")

    def del_knowledge(self, mode="single", index=0):
        """从记忆流中删除指定索引的记忆。"""
        if mode == "single":
            try:
                text= self.basic_knowledge[index]["text"]
                del self.basic_knowledge[index]
                logger.info(f"删除了记忆:{text}")
                print(f"删除了\"{text}\"")
            except IndexError:
                print("提供的索引超出了知识库的范围。")
        elif mode == "all":
            self.basic_knowledge.clear()  # 清空整个列表
            print("已清空所有知识。")

    def show_knowledge(self):
        """展示所有知识。"""
        knowledge_str = ""
        if not self.basic_knowledge:
            knowledge_str = "没有知识单元"
            print(knowledge_str)
            return knowledge_str

        knowledge_count = len(self.basic_knowledge)
        knowledge_str += f"知识条数：{knowledge_count}\n"
        knowledge_str += f"{'-' * 40}\n"  # 添加分隔线

        for idx, knowledge in enumerate(self.basic_knowledge, 1):
            text = knowledge["text"]
            embedding_size = len(knowledge["embedding"])
            knowledge_str += (
                f"知识 #{idx}\n"
                f"描述: {text}\n"
                f"嵌入向量大小: {embedding_size}\n"
                f"{'-' * 40}\n"
            )

        print(knowledge_str)
        return knowledge_str

    def search_knowledge(self, query_embedding):
        """搜索知识的方法，只返回最相似的一个知识项"""
        # 如果没有知识项，直接返回提示信息
        if not self.basic_knowledge:
            return {
                "text": "没有知识单元",
                "embedding": []
            }

        # 初始化最高相似度和相应的知识项
        max_similarity = -1
        most_similar_knowledge = None

        # 遍历所有知识项
        for knowledge in self.basic_knowledge:
            # 计算点积
            similarity = cosine_similarity(query_embedding, knowledge["embedding"])
            # 更新最高相似度和相应的知识项
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_knowledge = knowledge

        logger.info(f"找到了最相似的知识：{most_similar_knowledge['text']}")
        return most_similar_knowledge


    def chat(self, input, history):
        query_embedding = apis.embedding(input)[0]

        if history is None:
            history = []
        history.append(f"hadi:{input}")
        context = ""
        for chat in history:
            context = context + chat + "\n"

        memory = self.search_memory(query_embedding)
        knowledge = self.search_knowledge(query_embedding)

        prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
当前心情：{self.fsm.mood}
任务：作为{self.name}进行回复。
字数限制：不超过100字。
辅助信息：相关记忆：“{memory['description']}” 相关知识：“{knowledge['text']}”
对话上下文：
<<<
{self.language_style}
{context}
>>>
请以{self.name}的身份回复，不要扮演其他角色或添加额外信息。
"""
        print(prompt)
        logger.info(f"生成了对话提示：{prompt}")
        response = apis.chatgpt(prompt)
        print(response)
        logger.info(f"生成了回复：{response}")
        history.append(f"{response}")
        return response, history

    def create_thought(self, memory, knowledge, context):
        prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
当前心情：{self.fsm.mood}
任务：进行内心思考并返回内容。
字数限制：不超过100字。
辅助信息：相关记忆：“{memory['description']}” 相关知识：“{knowledge['text']}”
对话上下文：
<<<
{self.language_style}
{context}
>>>
请仅返回思考内容，不要添加额外信息或格式。
"""
        print(prompt)
        logger.info(f"生成了思考提示：{prompt}")
        thought = apis.chatgpt(prompt)
        print(thought)
        logger.info(f"生成了思考内容：{thought}")
        return thought

    def cot_chat(self, input, history):
        query_embedding = apis.embedding(input)[0]

        if history is None:
            history = []
        history.append(f"hadi:{input}")
        context = ""
        for chat in history:
            context = context + chat + "\n"

        memory = self.search_memory(query_embedding)
        knowledge = self.search_knowledge(query_embedding)
        thought = self.create_thought(memory, knowledge, context)

        prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
当前心情：{self.fsm.mood}
任务：基于思考内容进行回复。
字数限制：不超过100字。
辅助信息：思考内容：“{thought}”
对话上下文：
<<<
{self.language_style}
{context}
>>>
请在思考内容的基础上进行回复，不要扮演其他角色或添加额外信息。
"""
        print(prompt)
        logger.info(f"生成了对话提示：{prompt}")
        response = apis.chatgpt(prompt)
        print(response)
        logger.info(f"生成了回复：{response}")
        history.append(f"{response}")
        return response, history, thought

if __name__ == "__main__":
    # 加载存储在JSON文件中的大脑状态。
    try:
        with open("../resource/hutao.json", "r", encoding="utf-8") as json_file:
            loaded_data = json.load(json_file)
    except FileNotFoundError:
        print("未找到指定的文件。")
        exit()

    hutao = LucyAgent(perception=None, brain=Brain.from_json(loaded_data), action=None)

    # 打印状态
    hutao.brain.del_memory()
    hutao.brain.show_info()

    # history = None
    # hutao.brain.chat("胡桃可以给我来一杯咖啡吗？",history)
    # 结果：
    # 胡桃: 当然可以！请问您喜欢什么口味的咖啡呢？我们这里有各种不同的选择。


    # history = None
    # hutao.brain.cot_chat("胡桃可以给我来一杯咖啡吗？",history)
    # 结果：
    # 我在思考对方的请求是否合适。作为往生堂的堂主，我应该保持礼貌并尽力帮助他人。但是，作为一名读书人，我也需要专注于我的学习。
    # 或许我可以告诉他我正在读书，并询问他是否还需要其他帮助。这样既能满足他的需求，也不会打断我的学习。
    # 胡桃: 哈迪，你好！很抱歉，我现在正在读书，可能无法立即为你泡咖啡。不过我可以告诉你，我在往生堂经营一家咖啡店，提供一些饮品和小食。如果你还需要其他帮助，我会尽力满足你的需求。

    # 将更新后的大脑状态保存到JSON文件中。
    try:
        with open("../resource/hutao.json", "w", encoding="utf-8") as json_file:
            json.dump(hutao.brain.to_json(), json_file, indent=4, ensure_ascii=False)
    except IOError:
        print("无法写入文件。")