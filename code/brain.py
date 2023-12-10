import os
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
    filename='../agent_brain.log',  # 指定日志文件的名称
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
    def __init__(self, name, seed_memory, language_style,
                 mood_list, emoji_list, action_state_list,
                 basic_knowledge, memory_stream):

        # 结构性超参数（不放在实例化agent的内容json中，代码定义）
        self.memory_limit = MEMORY_LIMIT

        # 静态属性
        self.name = name
        self.seed_memory = seed_memory
        self.language_style = language_style
        self.mood_list = mood_list
        self.emoji_list = emoji_list
        self.action_state_list = action_state_list

        # 动态属性，注意到mood的初始化是固定的
        self.basic_knowledge = basic_knowledge
        self.memory_stream = memory_stream
        self.fsm = AgentFSM(initial_mood=mood_list[0],initial_action_state=action_state_list[0],
                            mood_list=mood_list, emoji_list=emoji_list, action_state_list=action_state_list)

    def to_json(self):
        """将大脑的状态转换为JSON格式的字典。"""
        return {
            "name": self.name,
            "seed_memory": self.seed_memory,
            "language_style": self.language_style,
            "mood_list": self.mood_list,
            "emoji_list": self.emoji_list,
            "action_state_list": self.action_state_list,
            "basic_knowledge": self.basic_knowledge,
            "memory_stream": self.memory_stream,
        }

    @classmethod
    def from_json(cls, json_data):
        """从JSON格式的数据创建Brain实例。"""
        return cls(**json_data)

    def show_info(self):
        """创建一个描述大脑状态的字符串"""
        info = f"静态的Brain Info:\n"
        info += f"Name: {self.name}\n\n"
        info += f"Seed Memory: {self.seed_memory}\n\n"
        info += f"Language Style(使用问候的对话作为对话基调): \n{self.language_style}\n\n"
        info += f"Mood List: {self.mood_list}\n\n"
        info += f"Emoji List: {self.emoji_list}\n\n"
        info += f"Action State List: {self.action_state_list}\n\n"

        info += f"动态的Brain Info:\n"
        info += f"Mood: {self.fsm.mood}\n\n"
        info += f"Action State: {self.fsm.action_state}\n\n"

        info += self.show_knowledge()
        info += f"Memory Limit: {self.memory_limit}\n\n"
        info += self.show_memory()

        logger.info(f"打印了Brain模块的相关信息：{info}")
        return info

    def create_memory(self, perception, output):
        """根据输入和输出创建一个记忆摘要，并返回记忆字典。"""
        summary_prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
任务：从事件描述中总结信息和行为。
字数限制：不超过100字。
事件描述：
<<<
{self.name}感知到的信息：{perception}
{self.name}的行为：{output}
>>>
请提供一个简洁，陈述性的总结，不要添加额外格式，不要修改事件的实际内容或添加额外信息。
"""
        print(summary_prompt)
        summary = apis.chatgpt(summary_prompt,0.5)  # 假设这个函数调用返回一个字符串摘要。
        embedding_list = apis.embedding(summary)  # 假设这个函数调用返回一个嵌入向量。
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        memory = {
            "description": summary,
            "create_time": time_string,
            "embedding": embedding_list[0]
        }
        logger.info(f"从{perception}和{output}中创建了新记忆：{summary}")
        return memory

    def add_memory(self, memory):
        """将一个记忆添加到记忆流中。"""
        self.memory_stream.append(memory)
        description = memory["description"]
        logger.info(f"添加了新记忆：{description}")
        # 检查记忆流是否达到上限，到达后就做总结
        if len(self.memory_stream) == self.memory_limit:
            self.summarize_memory()


    def summarize_memory(self):
        """以最新的记忆为标准，找到与其最语气相似的记忆对象，进行一次总结"""
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
        summary = apis.chatgpt(summary_prompt, 0.5)  # 假设这个函数调用返回一个字符串摘要。
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

    def del_memory(self, index=0, mode="single",  query=""):
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
                return f"删除了记忆:{description}"
            except IndexError:
                print("提供的索引超出了记忆流的范围。")
                return f"提供的索引超出了记忆流的范围。"
        elif mode == "all":
            self.memory_stream.clear()  # 清空整个列表
            logger.info("已清空所有记忆。")
            print("已清空所有记忆。")
            return "已清空所有记忆。"
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
                        return f"删除了匹配查询\"{query}\"的记忆：\"{memory['description']}\""
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
            memory_str = "没有记忆单元\n"
            print(memory_str)
            return memory_str

        memory_count = len(self.memory_stream)
        memory_str += f"记忆条数：{memory_count}\n"
        memory_str += f"{'-' * 40}\n"  # 添加分隔线

        for idx, memory in enumerate(self.memory_stream, 0):
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
        # 如果没有记忆项或查询向量，直接返回提示信息
        if not self.memory_stream:
            return {
                "description": "没有记忆单元",
                "create_time": "没有记忆单元",
                "embedding": []
            }

        if not query_embedding:
            return {
                "description": "查询向量为空",
                "create_time": "查询向量为空",
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

    @classmethod
    def extract_knowledge(cls, source):
        """从包含知识的文本中提取知识，也可以生成子知识的总结文本"""
        summary_prompt = f"""
任务：提取并总结文本中包含的信息或知识点。
字数限制：不超过500字。
知识点文本：
<<<
{source}
>>>
请提供准确的、陈述性的知识点总结,不要改变原始内容或添加额外信息,不要丢失关键信息。
"""
        summary = apis.chatgpt(summary_prompt, 0.3)
        logger.info(f"从{source}提取了知识点：{summary}")
        return summary

    def add_knowledge_list(self, knowledge_list):
        """
        将一个知识对象列表添加到知识库中，并记录日志。

        参数:
        knowledge_list (list): 包含知识对象的列表，每个知识对象都是一个包含"text"和其他可能的信息的字典。

        返回:
        无
        """
        for knowledge in knowledge_list:
            self.basic_knowledge.append(knowledge)
            logger.info(f"添加了知识：{knowledge['text']}")

    def add_knowledge_from_text(self, text, sub_knowledge=None):
        """
        根据提供的文本添加一个知识对象到知识库中，并记录日志。可选地，可以指定一个子知识的引用。

        参数:
        text (str): 要添加到知识库的知识文本。
        sub_knowledge (str, optional): 子知识的引用路径或标识符，默认为None。

        返回:
        无
        """
        embedding_list = apis.embedding(text)
        knowledge = {
            "text": text,
            "embedding": embedding_list[0],
            "sub_knowledge": sub_knowledge
        }
        self.basic_knowledge.append(knowledge)
        logger.info(f"添加了知识：{text}")

    def add_knowledge_with_sub_knowledge(self, summary_text, sub_knowledge_list):
        """
        为具有子知识列表的知识总结创建一个JSON文件来存储子知识，并将知识总结添加到知识库中。

        参数:
        summary_text (str): 知识总结的文本。
        sub_knowledge_list (list): 子知识的列表，每个子知识通常包含文本描述等信息。

        返回:
        无
        """
        sub_knowledge_file = f"../knowledge/{hash(summary_text)}_sub_knowledge.json"
        with open(sub_knowledge_file, 'w', encoding="utf-8") as file:
            json.dump(sub_knowledge_list, file, ensure_ascii=False, indent=4)

        logger.info(f"为知识总结：{summary_text} 添加了子知识文件：{sub_knowledge_file}")
        self.add_knowledge_from_text(summary_text, sub_knowledge=sub_knowledge_file)

    def del_knowledge(self, index=0, mode="single"):
        """
        从知识库中删除指定索引的知识单元。如果该知识单元有子知识，也会一并删除。

        参数:
        index (int): 要删除的知识单元的索引，默认为0。
        mode (str): 删除模式，"single"表示删除单个知识单元，"all"表示删除所有知识单元。

        返回:
        str: 删除操作的结果消息。
        """
        if mode == "single":
            try:
                delete_str = ""
                knowledge = self.basic_knowledge[index]
                text = knowledge["text"]
                delete_str += f"删除了知识:{text}"

                # 如果存在子知识文件，删除该文件
                sub_knowledge_file = knowledge.get("sub_knowledge")
                if sub_knowledge_file:
                    try:
                        os.remove(sub_knowledge_file)
                        delete_str += f"删除了子知识文件:{sub_knowledge_file}"
                        logger.info(f"删除了子知识文件:{sub_knowledge_file}")
                    except FileNotFoundError:
                        delete_str += f"删除了子知识文件:{sub_knowledge_file}"
                        logger.info(f"子知识文件:{sub_knowledge_file}未找到或已被删除")
                # 删除知识单元
                del self.basic_knowledge[index]
                logger.info(f"删除了知识:{text}")
                print(delete_str)
                return delete_str
            except IndexError:
                print("提供的索引超出了知识库的范围。")
                return "提供的索引超出了知识库的范围。"

        elif mode == "all":
            for knowledge in self.basic_knowledge:
                sub_knowledge_file = knowledge.get("sub_knowledge")
                if sub_knowledge_file:
                    try:
                        os.remove(sub_knowledge_file)
                    except FileNotFoundError:
                        pass
            self.basic_knowledge.clear()  # 清空整个列表
            print("已清空所有知识。")
            return "已清空所有知识。"

    def show_knowledge(self):
        """
        展示知识库中的所有知识单元及其子知识（如果有）。

        参数:
        无

        返回:
        str: 知识库内容的字符串表示。
        """
        knowledge_str = ""
        if not self.basic_knowledge:
            knowledge_str = "没有知识单元\n"
            print(knowledge_str)
            return knowledge_str

        knowledge_count = len(self.basic_knowledge)
        knowledge_str += f"知识条数：{knowledge_count}\n"
        knowledge_str += f"{'-' * 40}\n"  # 添加分隔线

        for idx, knowledge in enumerate(self.basic_knowledge, 0):
            text = knowledge["text"]
            embedding_size = len(knowledge["embedding"])
            knowledge_str += (
                f"知识 #{idx}\n"
                f"描述: {text}\n"
                f"嵌入向量大小: {embedding_size}\n"
            )
            # 展示子知识
            sub_knowledge_file = knowledge.get("sub_knowledge")
            if sub_knowledge_file:
                try:
                    with open(sub_knowledge_file, 'r', encoding="utf-8") as file:
                        sub_knowledges = json.load(file)
                    knowledge_str += f"子知识:\n"
                    for sub_knowledge in sub_knowledges:
                        knowledge_str += f"- {sub_knowledge['text']}\n"
                except FileNotFoundError:
                    knowledge_str += "有子知识,子知识文件未找到或已被删除\n"
            knowledge_str += f"{'-' * 40}\n"

        print(knowledge_str)
        return knowledge_str

    def search_knowledge(self, query_embedding):
        """
        该函数用于在预定义的知识库中搜索与给定查询向量最相似的知识项，并返回与之最相似的知识文本。

        参数:
        query_embedding: 查询的向量表示，用于与知识库中的知识项进行相似度比较。

        返回:
        knowledge_text: 与查询向量最相似的知识项的文本。如果没有知识库或查询向量为空，则返回相应的提示信息。
        """
        # 如果没有知识项或查询向量，直接返回提示信息
        if not self.basic_knowledge:
            return "没有知识单元"

        if not query_embedding:
            return "查询向量为空"

        # 初始化最高相似度和相应的知识项
        max_similarity = -1
        most_similar_knowledge = None
        knowledge_text = ""

        # 遍历所有知识项
        for knowledge in self.basic_knowledge:
            # 计算点积
            similarity = cosine_similarity(query_embedding, knowledge["embedding"])
            # 更新最高相似度和相应的知识项
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_knowledge = knowledge

        knowledge_text += most_similar_knowledge["text"]
        logger.info(f"找到了最相似的知识：{most_similar_knowledge['text']}")

        sub_knowledge_file = most_similar_knowledge.get("sub_knowledge")
        if sub_knowledge_file:
            try:
                with open(sub_knowledge_file, 'r', encoding="utf-8") as file:
                    sub_knowledges = json.load(file)

                max_similarity = -1
                most_similar_knowledge = None

                for sub_knowledge in sub_knowledges:
                    # 计算点积
                    similarity = cosine_similarity(query_embedding, sub_knowledge["embedding"])
                    # 更新最高相似度和相应的知识项
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_knowledge = sub_knowledge

                knowledge_text += most_similar_knowledge["text"]
                logger.info(f"找到了最相似的子知识：{most_similar_knowledge['text']}")

            except FileNotFoundError:
                knowledge_text += "有子知识,但子知识文件未找到或已被删除\n"

        return knowledge_text

    def chat(self, user_input, history):
        query_embedding = apis.embedding(user_input)[0]

        if history is None:
            history = []
        history.append(f"hadi:{user_input}")

        context = ""
        for chat in history:
            context = context + chat + "\n"

        memory = self.search_memory(query_embedding)
        knowledge_text = self.search_knowledge(query_embedding)

        prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
当前心情：{self.fsm.mood}
任务：作为{self.name}进行回复。
字数限制：不超过100字。
辅助信息：相关记忆：“{memory['description']}” 相关知识：“{knowledge_text}”
对话上下文：
<<<
{self.language_style}
{context}
>>>
请以{self.name}的身份回复，不要扮演其他角色或添加额外信息。
"""
        print(prompt)
        logger.info(f"生成了对话提示：{prompt}")
        response = apis.chatgpt(prompt,1.0)
        print(response)
        logger.info(f"生成了回复：{response}")
        history.append(f"{response}")
        return response, history

    def create_thought_from_perception(self, trigger):
        trigger_embedding = apis.embedding(trigger)[0]
        memory = self.search_memory(trigger_embedding)
        knowledge_text = self.search_knowledge(trigger_embedding)
        prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
任务：分析角色感知到的信息，基于角色第一视角进行思考，包含对角色相关事件的判断和角色的心理反应。
字数限制：不超过100字。
辅助信息：相关记忆：“{memory['description']}” 相关知识：“{knowledge_text}”
感知到的信息：
<<<
{trigger}
>>>
请仅返回第一人称视角下的思考内容，不要添加额外信息或格式。
"""
        print(prompt)
        logger.info(f"生成了思考提示：{prompt}")
        thought = apis.chatgpt(prompt,1.0)
        print(thought)
        logger.info(f"生成了思考内容：{thought}")
        return thought

    def create_thought_from_query(self, memory, knowledge_text, context):
        prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
当前心情：{self.fsm.mood}
任务：根据对话上下文和辅助信息，基于角色视角第一视角进行思考，包含对角色相关事件的判断和角色的心理反应。
字数限制：不超过100字。
辅助信息：相关记忆：“{memory['description']}” 相关知识：“{knowledge_text}”
对话上下文：
<<<
{self.language_style}
{context}
>>>
请仅返回第一人称视角下的思考内容，不要添加额外信息或格式。
"""
        print(prompt)
        logger.info(f"生成了思考提示：{prompt}")
        thought = apis.chatgpt(prompt,1.0)
        print(thought)
        logger.info(f"生成了思考内容：{thought}")
        return thought

    def cot_chat(self, user_input, history):
        query_embedding = apis.embedding(user_input)[0]

        if history is None:
            history = []
        history.append(f"hadi:{user_input}")
        context = ""
        for chat in history:
            context = context + chat + "\n"

        memory = self.search_memory(query_embedding)
        knowledge_text = self.search_knowledge(query_embedding)
        thought = self.create_thought_from_query(memory, knowledge_text, context)

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
        response = apis.chatgpt(prompt,1.0)
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
    print(hutao.brain.show_info())


    # 将更新后的大脑状态保存到JSON文件中。
    try:
        with open("../resource/hutao.json", "w", encoding="utf-8") as json_file:
            json.dump(hutao.brain.to_json(), json_file, indent=4, ensure_ascii=False)
    except IOError:
        print("无法写入文件。")