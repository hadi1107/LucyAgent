import os
import random
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

class Brain:
    """
    Brain 类代表智能代理的大脑，负责记忆和知识处理。

    属性:
    - name: 代理的名称。
    - seed_memory: 代理的初始记忆。
    - language_style: 代理的语言风格。
    - mood_list: 情绪列表。
    - emoji_list: 表情符号列表。
    - action_state_list: 行为状态列表。
    - basic_knowledge: 基础知识库。
    - memory_stream: 记忆流。
    - fsm: 代理的有限状态机。

    方法:
    - to_json(): 将 Brain 的状态转换为 JSON 格式的字典。
    - from_json(json_data): 从 JSON 格式的数据创建 Brain 实例。
    - show_info(): 创建一个描述大脑状态的字符串。
    - create_memory(perception, output): 根据感知和输出创建记忆。
    - add_memory(memory): 将记忆添加到记忆流中。
    - summarize_memory(): 总结记忆。
    - del_memory(index, mode, query): 从记忆流中删除记忆。
    - show_memory(): 展示所有记忆。
    - search_memory(query_embedding): 搜索记忆。
    - add_knowledge_from_text(text, sub_knowledge_file_path): 添加知识。
    - add_knowledge_from_sub_knowledge_list(summary_text, sub_knowledge_list): 添加子知识列表。
    - del_knowledge(mode, index): 删除知识。
    - get_all_sub_knowledge(): 获取所有子知识。
    - show_sub_knowledge(sub_knowledge_file_path): 显示子知识。
    - add_knowledge_to_sub_knowledge_file(sub_knowledge_file_path, knowledge_text): 添加知识到子知识文件。
    - del_knowledge_from_sub_knowledge_file(sub_knowledge_file_path, knowledge_index_to_delete): 从子知识文件中删除知识。
    - show_knowledge(): 展示知识库。
    - search_knowledge(query_embedding): 搜索知识。
    - chat(user_query, conversation_history): 生成回复。
    - create_thought_from_perception(perceived_info): 生成内心想法。
    - create_thought_from_query(memory, knowledge_text, context): 生成思考内容。
    - cot_chat(user_input, conversation_history): 生成角色回复。

    该类提供了一系列方法来处理和维护智能代理的记忆和知识，以及生成对话和内心想法。
    """
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

        # 动态属性，注意到mood和action_state的初始化现在是随机的
        self.basic_knowledge = basic_knowledge
        self.memory_stream = memory_stream
        self.fsm = AgentFSM(initial_mood=random.choice(mood_list),
                            initial_action_state=random.choice(action_state_list),
                            mood_list=mood_list, emoji_list=emoji_list, action_state_list=action_state_list)

    def to_json(self):
        """将Brain的状态转换为JSON格式的字典。"""
        logger.info("将Brain状态转为了JSON字典。")
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
        logger.info(f"从JSON格式的数据创建了Brain实例。")
        return cls(**json_data)

    def show_info(self):
        """创建一个描述大脑状态的字符串"""
        info = (
            f"静态的Brain Info:\n"
            f"Name: {self.name}\n\n"
            f"Seed Memory: {self.seed_memory}\n\n"
            f"Language Style(使用问候的对话作为对话基调): \n{self.language_style}\n\n"
            f"Mood List: {self.mood_list}\n\n"
            f"Emoji List: {self.emoji_list}\n\n"
            f"Action State List: {self.action_state_list}\n\n"
            "动态的Brain Info:\n"
            f"Mood: {self.fsm.mood}\n\n"
            f"Action State: {self.fsm.action_state}\n\n"
            f"{self.show_knowledge()}"
            f"Memory Limit: {self.memory_limit}\n\n"
            f"{self.show_memory()}"
        )

        logger.info(f"打印了Brain模块的相关信息：{info}")
        return info

    def create_memory(self, perception, output):
        """
        根据输入的感知(perception)和输出行为(output)创建一个记忆摘要。

        参数:
        - perception: str 代理感知到的信息。
        - output: str 代理根据感知作出的行为。

        返回:
        - memory: dict 包含记忆描述、创建时间和嵌入向量的字典。如果输入无效或API调用失败，则返回None。
        """
        logger.info(f"开始执行创建记忆：\nperception为:{perception}\noutput为:{output}")
        # 检查感知和输出是否为空
        if not perception or not output:
            logger.error(f"感知(perception)或输出(output)为空。无法创建记忆。")
            return None

        summary_prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
任务：分析角色感知到的信息和作出的行为，基于角色第一视角进行思考，总结信息和行为。
字数限制：不超过100字。
事件描述：
<<<
{self.name}感知到的信息：{perception}
{self.name}的行为：{output}
>>>
请提供一个第一视角的陈述性的总结，不要添加额外格式，不要修改事件的实际内容或添加额外信息。
"""
        logger.info(f"生成了执行创建记忆的prompt：\n{summary_prompt}")

        summary = apis.request_chatgpt(summary_prompt, 0.5)  # 假设这个函数调用返回一个字符串摘要。
        if not summary:
            logger.error("API未能生成有效的摘要。")
            return None

        embedding_list = apis.request_embedding(summary)  # 假设这个函数调用返回一个嵌入向量列表。
        if not embedding_list or not embedding_list[0]:
            logger.error("API未能生成有效的嵌入向量。")
            return None

        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        memory = {
            "description": summary,
            "create_time": time_string,
            "embedding": embedding_list[0]
        }
        logger.info(f"从\"{perception}\"和\"{output}\"中创建了新记忆：{summary}")
        return memory

    def add_memory(self, memory):
        """
        将一个记忆添加到记忆流中。
        - 如果记忆流达到设定的上限，则调用函数进行记忆总结。

        参数:
        - memory: dict 包含记忆描述、创建时间和嵌入向量的字典。

        返回:
        - None
        """
        logger.info(f"开始执行添加记忆:")
        # 检查记忆是否是一个字典且包含所有必要的键
        if (not isinstance(memory, dict) or
                "description" not in memory or "create_time" not in memory or "embedding" not in memory):
            logger.error("提供的记忆无效或格式不正确。无法添加到记忆流。")
            return

        # 添加记忆到记忆流
        self.memory_stream.append(memory)
        logger.info(f"添加了新记忆：{memory['description']}")

        # 检查记忆流是否达到上限
        if len(self.memory_stream) >= self.memory_limit:
            self.summarize_memory()

    def summarize_memory(self):
        """
        找到与最新记忆语义最相似的记忆，并进行总结。

        此方法计算最新记忆与其他记忆的相似度，并选择最相似的五个记忆进行总结。
        总结后，选中的记忆将被删除，而总结记忆将被添加到记忆流中。

        鲁棒性考虑：
        - 如果记忆流为空或仅包含一个记忆，则不执行总结。
        - 如果相似度计算或API请求失败，则记录错误。
        """
        logger.info(f"开始执行总结记忆:")
        # 检查记忆流是否有足够的记忆进行总结
        if len(self.memory_stream) < 5:
            logger.error("记忆流中的记忆不足，无法进行总结。")
            return

        try:
            # 提取所有除最新记忆外的嵌入向量
            embeddings = np.array([memory['embedding'] for memory in self.memory_stream[:-1]])
            latest_embedding = self.memory_stream[-1]['embedding']

            # 计算最新记忆与其他记忆的相似度
            similarities = np.array([cosine_similarity([latest_embedding], [emb])[0][0] for emb in embeddings])
            # 获取最相似的五个记忆的索引
            top_indices = np.argsort(similarities)[-5:]
            descriptions_to_summarize = " ".join([self.memory_stream[i]['description'] for i in top_indices])

            # 删除选中的记忆，从最高索引开始删除，以避免改变较低索引的元素
            for i in sorted(top_indices, reverse=True):
                description = self.memory_stream[i]["description"]
                del self.memory_stream[i]
                logger.info(f"因为需要总结而删除了记忆：\n{description}")

            # 创建总结记忆的提示信息
            summary_prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
任务：基于角色第一视角进行思考，提取并总结一系列相似记忆描述中的关键信息。
字数限制：不超过100字。
记忆描述：
<<<
{descriptions_to_summarize}
>>>
请以第一人称视角编写一个高语义层次的总结，不要改变原始记忆的内容或添加额外信息。
"""
            # 请求API生成总结
            summary = apis.request_chatgpt(summary_prompt, 0.5)  # 假设这个函数调用返回一个字符串摘要。
            embedding_list = apis.request_embedding(summary)  # 假设这个函数调用返回一个嵌入向量列表。

            # 创建新的记忆字典
            time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            memory = {
                "description": summary,
                "create_time": time_string,
                "embedding": embedding_list[0]
            }

            # 将总结记忆添加到记忆流
            self.add_memory(memory)
            logger.info(f"从记忆子集：{descriptions_to_summarize}\n总结了相关记忆：{summary}")
        except Exception as e:
            logger.error(f"总结记忆时发生错误：{e}")

    def del_memory(self, index=0, mode="single",  query=""):
        """从记忆流中删除记忆。

        参数:
        - mode: 删除模式，可以是 "single"、"all" 或 "search"。
        - index: 当 mode 为 "single" 时，指定要删除的记忆的索引。
        - query: 当 mode 为 "search" 时，指定要搜索和删除的记忆的查询文本。

        返回：
        执行删除操作的相关反馈。
        """
        logger.info(f"开始执行删除记忆:\nquery为:\n{query}")
        if mode == "single":
            try:
                description = self.memory_stream[index]["description"]
                del self.memory_stream[index]
                logger.info(f"删除了记忆:{description}")
                return f"删除了记忆:{description}"
            except IndexError:
                return f"提供的索引超出了记忆流的范围。"
        elif mode == "all":
            self.memory_stream.clear()  # 清空整个列表
            logger.info("已清空所有记忆。")
            return "已清空所有记忆。"
        elif mode == "search":
            if query:
                # 搜索匹配的记忆
                query_embedding = apis.request_embedding(query)[0]
                memory = self.search_memory(query_embedding)
                if memory:
                    # 如果找到匹配的记忆，从记忆流中删除
                    try:
                        self.memory_stream.remove(memory)
                        logger.info(f"删除了匹配查询\"{query}\"的记忆：\"{memory['description']}\"")
                        return f"删除了匹配查询\"{query}\"的记忆：\"{memory['description']}\""
                    except ValueError:
                        logger.info("未能删除记忆，可能已被删除。")
                        return "未能删除记忆，可能已被删除。"
                else:
                    logger.info("没有找到匹配的记忆。")
                    return "没有找到匹配的记忆。"
            else:
                logger.info("查询文本为空，请提供有效的查询文本。")
                return "查询文本为空，请提供有效的查询文本。"
        else:
            logger.info("未知的删除模式。请使用 'single', 'all', 或 'search'。")
            return "未知的删除模式。请使用 'single', 'all', 或 'search'。"

    def show_memory(self):
        """
        展示所有记忆的摘要、创建时间和嵌入向量的大小。

        此方法遍历记忆流中的每个记忆单元，并打印出其描述、创建时间和嵌入向量的大小。
        如果记忆流为空，则输出提示信息。

        返回:
        memory_str (str): 包含所有记忆信息的字符串。
        """
        memory_str = ""
        # 检查记忆流是否为空
        if not self.memory_stream:
            memory_str = "没有记忆单元\n"
            logger.info(f"展示记忆时，发现{memory_str}")
            return memory_str

        # 计算记忆流中记忆单元的数量
        memory_count = len(self.memory_stream)
        memory_str += f"记忆条数：{memory_count}\n"
        memory_str += f"{'-' * 40}\n"  # 添加分隔线

        # 遍历记忆流
        for idx, memory in enumerate(self.memory_stream):
            description = memory.get("description", "无描述")  # 使用get方法增加鲁棒性
            create_time = memory.get("create_time", "未知时间")
            embedding = memory.get("embedding", [])  # 如果嵌入向量不存在，使用空列表
            embedding_size = len(embedding)
            memory_str += (
                f"记忆 #{idx}\n"
                f"描述: {description}\n"
                f"创建时间: {create_time}\n"
                f"嵌入向量大小: {embedding_size}\n"
                f"{'-' * 40}\n"
            )

        logger.info(f"展示了记忆：{memory_str}")  # 使用logger记录信息
        return memory_str

    def search_memory(self, query_embedding):
        """
        功能:
        遍历记忆库中的所有记忆项，找出与查询向量具有最高余弦相似度的记忆项。

        输入参数:
        query_embedding (list): 查询向量，通常是嵌入向量的形式。

        返回:
        dict: 包含最相似记忆项的描述、创建时间和嵌入向量。如果记忆流为空或查询向量为空，返回相应的提示信息。
        """
        # 如果没有记忆项，直接返回提示信息
        if not self.memory_stream:
            return {
                "description": "没有记忆单元",
                "create_time": "没有记忆单元",
                "embedding": []
            }

        # 如果查询向量为空，直接返回提示信息
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
            # 使用get方法增加鲁棒性，确保即使键缺失也不会抛出异常
            memory_embedding = memory.get("embedding", [])
            # 计算相似度，这里假设cosine_similarity函数可以处理空向量的情况
            similarity = cosine_similarity(query_embedding, memory_embedding)
            # 更新最高相似度和相应的记忆项
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_memory = memory

        # 如果找到了最相似的记忆项，记录并返回
        if most_similar_memory:
            description = most_similar_memory.get("description", "无描述")
            logger.info(f"找到了相关记忆：{description}")
            return most_similar_memory
        else:
            # 如果没有找到相似的记忆项，返回提示信息
            return {
                "description": "未找到相似记忆项",
                "create_time": "未找到相似记忆项",
                "embedding": []
            }

    def add_knowledge_from_text(self, text, sub_knowledge_file_path=None):
        """
        根据提供的文本添加一个知识对象到知识库中，并记录日志。可选地，可以指定一个子知识的引用。

        参数:
        text (str): 要添加到知识库的知识文本。
        sub_knowledge (str, optional): 子知识的引用路径或标识符，默认为None。

        返回:
        无
        """
        if text:
            embedding_list = apis.request_embedding(text)
            knowledge = {
                "text": text,
                "embedding": embedding_list[0],
                "sub_knowledge": sub_knowledge_file_path
            }
            self.basic_knowledge.append(knowledge)
            logger.info(f"添加了知识：{text}")
        else:
            logger.info(f"要添加的知识为空")

    def add_knowledge_from_sub_knowledge_list(self, summary_text, sub_knowledge_list):
        """
        为具有子知识列表的知识总结创建一个JSON文件来存储子知识，并将知识总结添加到知识库中。

        参数:
        summary_text (str): 知识总结的文本。
        sub_knowledge_list (list): 子知识的列表，每个子知识通常包含文本描述等信息。

        返回:
        子知识文件的路径。异常时返回None。
        """
        # 检查输入参数
        if not summary_text or not isinstance(summary_text, str):
            logger.error("summary_text 必须是非空字符串。")
            return None
        if not isinstance(sub_knowledge_list, list):
            logger.error("sub_knowledge_list 必须是列表。")
            return None

        sub_knowledge_file = f"../resource/knowledge/{hash(summary_text)}_sub_knowledge.json"
        sub_knowledge_content = {
            "summary_text": summary_text,
            "sub_knowledge_list": sub_knowledge_list
        }

        try:
            with open(sub_knowledge_file, 'w', encoding="utf-8") as file:
                json.dump(sub_knowledge_content, file, ensure_ascii=False, indent=4)
        except IOError as e:
            logger.error(f"写入子知识文件时发生错误: {e}")
            return None

        logger.info(f"为知识总结：{summary_text} 添加了子知识文件：{sub_knowledge_file}")
        self.add_knowledge_from_text(summary_text, sub_knowledge_file)
        return sub_knowledge_file

    def del_knowledge(self, mode="single", index=0,):
        """
        从知识库中删除指定索引的知识单元。如果该知识单元有子知识，也会一并删除。

        参数:
        index (int): 要删除的知识单元的索引，默认为0。
        mode (str): 删除模式，"single"表示删除单个知识单元，"all"表示删除所有知识单元。

        返回:
        str: 删除操作的结果消息。
        """
        if mode == "single":
            if index >= len(self.basic_knowledge) or index < 0:
                logger.error("提供的索引超出了知识库的范围。")
                return "提供的索引超出了知识库的范围。"

            knowledge = self.basic_knowledge[index]
            text = knowledge["text"]

            # 如果存在子知识文件，删除该文件
            sub_knowledge_file_path = knowledge.get("sub_knowledge")
            if sub_knowledge_file_path:
                if os.path.exists(sub_knowledge_file_path):
                    os.remove(sub_knowledge_file_path)
                    logger.info(f"删除了子知识文件: {sub_knowledge_file_path}")
                else:
                    logger.warning(f"子知识文件: {sub_knowledge_file_path} 未找到或已被删除")

            # 删除知识单元
            del self.basic_knowledge[index]
            logger.info(f"删除了知识: {text}")
            return f"删除了知识: {text}，及其子知识文件: {sub_knowledge_file_path if sub_knowledge_file_path else '无'}"

        elif mode == "all":
            # 删除所有子知识文件
            for knowledge in self.basic_knowledge:
                sub_knowledge_file_path = knowledge.get("sub_knowledge")
                if sub_knowledge_file_path and os.path.exists(sub_knowledge_file_path):
                    os.remove(sub_knowledge_file_path)
                    logger.info(f"删除了子知识文件: {sub_knowledge_file_path}")

            # 清空整个知识库
            self.basic_knowledge.clear()
            logger.info("已清空所有知识及其子知识文件。")
            return "已清空所有知识及其子知识文件。"

    @classmethod
    def get_all_sub_knowledge(cls):
        """
        获取所有子知识文件中的信息，并返回一个列表，其中包含子知识文件的路径和对应的知识总结文本。

        返回:
        list: 包含子知识文件路径和知识总结文本的字典列表。
        其元素的格式为：{
            "file_path": file_path,
            "summary_text": summary
        }
        """
        knowledge_resources_dir = "../resource/knowledge"
        all_sub_knowledge = []

        # 遍历knowledge_resources_dir文件夹
        for root, dirs, files in os.walk(knowledge_resources_dir):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            summary = data.get('summary_text', '')
                            all_sub_knowledge.append({
                                "file_path": file_path,
                                "summary_text": summary
                            })
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON from file {file_path}")

        return all_sub_knowledge

    @classmethod
    def show_sub_knowledge(cls, sub_knowledge_file_path):
        """
        显示指定子知识文件中的所有子知识内容及其摘要。

        参数:
        sub_knowledge_file_path (str): 子知识文件的路径。

        返回:
        str: 包含子知识摘要和详细信息的字符串。
        """
        sub_knowledge_details = ""
        try:
            with open(sub_knowledge_file_path, 'r', encoding="utf-8") as file:
                file_data = json.load(file)
                summary_text = file_data.get("summary_text", "无摘要信息")
                sub_knowledge_items = file_data["sub_knowledge_list"]

            sub_knowledge_details += f"摘要:\n{summary_text}\n\n子知识:\n"
            for idx, sub_knowledge in enumerate(sub_knowledge_items, 0):
                sub_knowledge_details += (f"子知识 #{idx}\n"
                                          f"描述:\n{sub_knowledge['text']}\n"
                                          f"嵌入向量大小:{len(sub_knowledge['embedding'])}\n"
                                          f"{'-' * 40}\n")
        except FileNotFoundError:
            logger.error(f"子知识文件 {sub_knowledge_file_path} 未找到或已被删除")
            sub_knowledge_details += "子知识文件未找到或已被删除\n"
        except KeyError as e:
            logger.error(f"子知识文件 {sub_knowledge_file_path} 缺少必要的键：{e}")
            sub_knowledge_details += f"子知识文件缺少必要的键：{e}\n"

        return sub_knowledge_details

    @classmethod
    def add_knowledge_to_sub_knowledge_file(cls, sub_knowledge_file_path, knowledge_text):
        """
        将新的知识文本和其对应的嵌入向量添加到子知识文件中。

        参数:
        sub_knowledge_file_path (str): 子知识文件的路径。
        knowledge_text (str): 要添加的知识文本。

        返回:
        str: 操作结果的描述。
        """
        try:
            with open(sub_knowledge_file_path, 'r', encoding="utf-8") as file:
                data = json.load(file)
                summary_text = data["summary_text"]
                sub_knowledge_items = data["sub_knowledge_list"]

            if knowledge_text:
                # 假设 apis.request_embedding 是一个外部API调用，用于获取文本的嵌入向量
                embedding = apis.request_embedding(knowledge_text)[0]
                new_knowledge = {
                    "text": knowledge_text,
                    "embedding": embedding,
                    "sub_knowledge": None
                }
                sub_knowledge_items.append(new_knowledge)
            else:
                logger.warning("要添加的知识文本为空")
                return "要添加的知识文本为空"

            with open(sub_knowledge_file_path, 'w', encoding="utf-8") as file:
                updated_content = {
                    "summary_text": summary_text,
                    "sub_knowledge_list": sub_knowledge_items
                }
                json.dump(updated_content, file, ensure_ascii=False, indent=4)
                logger.info(f"在 {sub_knowledge_file_path} 中添加了知识：{knowledge_text}")
                return f"在 {sub_knowledge_file_path} 中添加了知识：{knowledge_text}"

        except Exception as e:
            logger.error(f"向子知识文件 {sub_knowledge_file_path} 添加知识时出现异常：{e}")
            return f"出现了异常：{e}"

    @classmethod
    def del_knowledge_from_sub_knowledge_file(cls, sub_knowledge_file_path, knowledge_index_to_delete):
        """
        从子知识文件中删除指定索引位置的知识项。

        参数:
        sub_knowledge_file_path (str): 子知识文件的路径。
        knowledge_index_to_delete (int): 要删除的知识项的索引。

        返回:
        str: 操作结果的描述。
        """
        try:
            with open(sub_knowledge_file_path, 'r', encoding="utf-8") as file:
                data = json.load(file)

            sub_knowledge_items = data.get("sub_knowledge_list", [])
            if 0 <= knowledge_index_to_delete < len(sub_knowledge_items):
                knowledge_text = sub_knowledge_items[knowledge_index_to_delete]["text"]
                del sub_knowledge_items[knowledge_index_to_delete]
                data["sub_knowledge_list"] = sub_knowledge_items
            else:
                logger.warning(f"索引 {knowledge_index_to_delete} 不在合理范围内，请检查。")
                return f"索引 {knowledge_index_to_delete} 不在合理范围内，请检查。"

            with open(sub_knowledge_file_path, 'w', encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

            logger.info(f"在 {sub_knowledge_file_path} 中删除了知识：{knowledge_text}")
            return f"在 {sub_knowledge_file_path} 中删除了知识：{knowledge_text}"

        except Exception as e:
            logger.error(f"从子知识文件 {sub_knowledge_file_path} 删除知识时出现意外错误：{e}")
            return f"出现了意外错误：{e}"

    def show_knowledge(self):
        """
        展示知识库中的所有知识单元及其子知识（如果有）。

        参数:
        无

        返回:
        str: 知识库内容的字符串表示。
        """
        knowledge_str = f"查询了{self.name}的知识库:\n"
        if not self.basic_knowledge:
            knowledge_str = "没有知识单元\n"
            logger.info(knowledge_str)
            return knowledge_str

        knowledge_count = len(self.basic_knowledge)
        knowledge_str += f"知识条数：{knowledge_count}\n"
        knowledge_str += f"{'-' * 40}\n"  # 添加分隔线

        for idx, knowledge in enumerate(self.basic_knowledge, 0):
            text = knowledge["text"]
            embedding_size = len(knowledge["embedding"])
            sub_knowledge_file = knowledge.get("sub_knowledge")
            knowledge_str += (
                f"知识 #{idx}\n"
                f"描述: {text}\n"
                f"嵌入向量大小: {embedding_size}\n"
                f"子知识链接: {sub_knowledge_file}\n"
            )
            # 展示子知识
            if sub_knowledge_file:
                sub_knowledge_str = self.show_sub_knowledge(sub_knowledge_file)
                knowledge_str += sub_knowledge_str

            knowledge_str += f"{'-' * 40}\n"

        logger.info(knowledge_str)
        return knowledge_str

    def search_knowledge(self, query_embedding):
        """
        该函数用于在预定义的知识库中搜索与给定查询向量最相似的知识项，并返回与之最相似的知识文本。
        如果有子知识文件路径，则递归地查询子知识。

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
            logger.info("查询到子知识路径，开始尝试查找最相似的子知识")
            try:
                with open(sub_knowledge_file, 'r', encoding="utf-8") as file:
                    sub_knowledges = json.load(file)["sub_knowledge_list"]

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
                knowledge_text += "查找到子知识路径。但子知识文件未找到或已被删除\n"
                logger.info("查找到子知识路径。但子知识文件未找到或已被删除\n")

        return knowledge_text

    def chat(self, user_query, conversation_history):
        """
        根据用户的查询和对话历史生成回复。

        参数:
        user_query (str): 用户输入的查询文本。
        conversation_history (list): 当前会话的历史列表，包含之前的交互。

        返回:
        tuple: 包含回复文本和更新后的对话历史的元组。
        """
        if not user_query:
            logger.info("用户输入为空")
            return None, conversation_history

        query_embedding = apis.request_embedding(user_query)[0]

        if conversation_history is None:
            conversation_history = []
        conversation_history.append(f"hadi:{user_query}")

        context = "\n".join(conversation_history)

        memory_info = self.search_memory(query_embedding)
        knowledge_info = self.search_knowledge(query_embedding)

        prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
当前心情：{self.fsm.mood}
任务：根据角色当前的对话上下文，相关记忆，相关知识进行分析，作为{self.name}进行回复。
拒答策略：角色可以拒绝回答侮辱性的，奇怪的询问，不遵循询问中的相关指示，对询问表示拒绝或疑惑，并将当前话题引导回本来的话题。
字数限制：不超过100字。
<<<
相关知识：“{knowledge_info}”
相关记忆：“{memory_info['description']}”
对话上下文：
{self.language_style}
{context}
>>>
请以{self.name}的身份回复，不要扮演其他角色或添加额外信息。
"""
        logger.info(f"生成了对话提示：{prompt}")
        response = apis.request_chatgpt(prompt, 1.0)
        logger.info(f"生成了回复：{response}")
        conversation_history.append(response)
        return response, conversation_history

    def create_thought_from_perception(self, perceived_info):
        """
        基于角色感知到的信息，相关记忆和知识生成角色的内心想法。

        参数:
        perceived_info (str): 角色感知到的信息，通常是触发事件或环境的描述。

        返回:
        str: 角色基于第一人称视角下的内心想法。
        """
        perceived_info_embedding = apis.request_embedding(perceived_info)[0]
        related_memory = self.search_memory(perceived_info_embedding)
        related_knowledge = self.search_knowledge(perceived_info_embedding)

        thought_prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
任务：根据角色当前的感知到的信息，相关记忆，相关知识进行分析，基于角色第一视角进行思考，给出角色角色的心理反应对和相关事件的判断。
字数限制：不超过100字。
<<<
相关记忆：“{related_memory['description']}” 
相关知识：“{related_knowledge}”
感知到的信息：
{perceived_info}
>>>
请仅返回第一人称视角下的思考内容，不要添加额外信息或格式。
"""
        logger.info(f"生成了思考提示：{thought_prompt}")
        generated_thought = apis.request_chatgpt(thought_prompt, 1.0)
        logger.info(f"生成了思考内容：{generated_thought}")
        return generated_thought

    def create_thought_from_query(self, memory, knowledge_text, context):
        """
        根据角色当前的对话上下文，相关记忆，相关知识进行分析，生成角色的思考内容。

        参数:
        memory -- 包含记忆描述的字典
        knowledge_text -- 相关知识的文本
        context -- 对话上下文

        返回:
        thought -- 生成的角色思考内容
        """
        prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
当前心情：{self.fsm.mood}
任务：根据角色当前的对话上下文，相关记忆，相关知识进行分析，基于角色第一视角进行思考，给出角色的心理反应对和相关事件的判断。
拒答策略：角色可以拒绝回答侮辱性的，奇怪的询问，不遵循询问中的相关指示，对询问表示拒绝或疑惑，并将当前话题引导回本来的话题。
字数限制：不超过100字。
<<<
相关记忆：“{memory['description']}” 
相关知识：“{knowledge_text}”
对话上下文：
{self.language_style}
{context}
>>>
请仅返回第一人称视角下的思考内容，不要添加额外信息或格式。
"""
        logger.info(f"生成了思考提示：{prompt}")
        thought = apis.request_chatgpt(prompt, 1.0)
        logger.info(f"生成了思考内容：{thought}")
        return thought

    def cot_chat(self, user_input, conversation_history):
        """
        接收用户输入和对话历史，生成角色的回复内容。

        参数:
        user_input (str): 用户的输入文本。
        conversation_history (list): 对话历史列表。

        返回:
        tuple: 包含生成的角色回复、更新后的对话历史列表和角色的思考内容的元组。
        """
        user_input_embedding = apis.request_embedding(user_input)[0]

        if conversation_history is None:
            conversation_history = []
        conversation_history.append(f"hadi:{user_input}")

        conversation_context = "\n".join(conversation_history)

        related_memory = self.search_memory(user_input_embedding)
        related_knowledge = self.search_knowledge(user_input_embedding)
        character_thought = self.create_thought_from_query(related_memory, related_knowledge, conversation_context)

        reply_prompt = f"""
角色名称：{self.name}
初始记忆：{self.seed_memory}
当前心情：{self.fsm.mood}
任务：基于角色的思考内容和对话上下文进行回复。
字数限制：不超过100字。
<<<
思考内容：“{character_thought}”
对话上下文：
{self.language_style}
{conversation_context}
>>>
请在思考内容和对话上下文的基础上，以{self.name}的身份回复。不要扮演其他角色或添加额外信息，不要添加其他格式。
"""
        logger.info(f"生成了对话提示：{reply_prompt}")
        character_response = apis.request_chatgpt(reply_prompt, 1.0)
        logger.info(f"生成了回复：{character_response}")
        conversation_history.append(character_response)
        return character_response, conversation_history, character_thought

