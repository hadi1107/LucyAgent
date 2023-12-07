import gradio as gr
import json
import apis
from perception import Perception
from action import Action
from brain import Brain
from lucy_agent import LucyAgent

# 理论上应该从perception的一些传感器接收到一些传感数据，然后再转为自然语言描述用于驱动状态转移
# 假设perception可以获得时间，视觉（周边感知），专有接口（比如玩家摁键要打招呼，发了短信等等）
# 根据状态不同，perception能收到的信息也不同,并且action理论上是“准备去做“而不是已经在做了
# 作为模拟，直接给出一些场景可能的自然语言描述，并直接切换状态，因为重点在于测试LLM基于感知描述进行决策的能力

EVENTS_LIST = [
    "胡桃看到了hadi在往生堂门口,听到了hadi在打招呼。",
    "胡桃收到璃月管委会的消息,内容为：往生堂第七十七代堂主，您好。近来璃月要举行一场特别的送别之仪，请您着手策划。",
    "胡桃从对话中了解到，hadi点了一杯拿铁。",
    "胡桃听到自己的闹钟响了，查看后发现备注为：记得看看璃月的历史书！"
]

def save_to_file(file_path:str, conversations)-> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)

def load_prompts(json_file: str) -> dict:
    with open(json_file, 'r',encoding='utf-8') as f:
        data = json.load(f)
    return {item['key']: item['prompt'] for item in data}

def save_agent_json(agent_brain):
    # 存到“中文name.json”当中，不影响用来初始化的文件
    name = agent_brain.name
    file_path = f"../resource/{name}.json"
    # 用来修改初始化文件
    # file_path = f"../resource/hutao.json"
    try:
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(agent_brain.to_json(), json_file, indent=4, ensure_ascii=False)
    except IOError:
        print("无法写入文件。")

if __name__ == "__main__":
    # 用初始json进行初始化
    with open("../resource/hutao.json", "r", encoding="utf-8") as json_file:
        loaded_data = json.load(json_file)

    perception = Perception()
    action = Action()
    hutao = LucyAgent(perception=perception, brain=Brain.from_json(loaded_data), action=action)

    # 创建一个 Gradio 界面
    with gr.Blocks() as demo:

        # 创建一个状态对象，用于存储历史记录
        state = gr.State([])

        with gr.Tab("和胡桃互动！\U0001F917"):
            def show_action_state():
                scene_path = ""
                if hutao.brain.fsm.action_state == "休息":
                    scene_path = "../resource/pictures/hutao_xiuxi.webp"
                if hutao.brain.fsm.action_state == "看书":
                    scene_path = "../resource/pictures/hutao_kanshu.webp"
                if hutao.brain.fsm.action_state == "策划往生堂相关活动":
                    scene_path = "../resource/pictures/hutao_cehua.jpg"
                if hutao.brain.fsm.action_state == "做咖啡并递交给客户":
                    scene_path = "../resource/pictures/hutao_coffee.jpg"
                if hutao.brain.fsm.action_state == "回复问题和聊天":
                    scene_path = "../resource/pictures/hutao_yao.webp"
                return f"胡桃正在{hutao.brain.fsm.action_state}", scene_path

            def perceive_and_change_action(trigger):
                if not trigger:
                    return "下拉菜单为空或没有接收到下拉菜单的值", "../resource/pictures/hutao_naohuo.webp"
                old_action_state = hutao.brain.fsm.action_state
                thought = hutao.brain.create_thought_from_perception(trigger)
                hutao.brain.fsm.action_state_transition(trigger, thought)
                action_state_str, scene_path = show_action_state()
                action_state_str = (f"胡桃原先正在{old_action_state},因为{trigger}胡桃认为:{thought}"
                                    f"\n\n因而决定{hutao.brain.fsm.action_state}")

                memory = hutao.brain.create_memory(trigger, f"胡桃进行了思考：{thought}")
                hutao.brain.add_memory(memory)
                save_agent_json(hutao.brain)

                print(action_state_str)
                return action_state_str, scene_path

            state_box = gr.Textbox(label="胡桃的行动情况🚴‍♂️")
            image_box = gr.Image(label="胡桃正在干什么？\U0001F60A", height=300)
            query_button = gr.Button("查询胡桃的行动情况🤶")
            query_button.click(show_action_state, inputs=[], outputs=[state_box, image_box])

            events_dropdown = gr.Dropdown(choices=EVENTS_LIST, label="可以模拟的事件列表🧐")
            event_button = gr.Button("模拟一些事件的发生⚡")
            event_button.click(perceive_and_change_action, inputs=events_dropdown, outputs=[state_box, image_box])

        with gr.Tab("和胡桃对话！🥳"):
            def talk_with_hutao(query, history=None):
                if hutao.brain.fsm.action_state != "回复问题和聊天":
                    action_state_str, scene_path = show_action_state()
                    return f"{action_state_str},没法回复你嘞，请去第一个标签页改变胡桃状态", "../resource/audios/这是一段测试音频哟.wav", scene_path
                if not query:
                    return f"请不要不说话嘞", "../resource/audios/这是一段测试音频哟.wav", "../resource/pictures/hutao_naohuo.webp"
                if history is None:
                    history = []
                response, history, thought = hutao.brain.cot_chat(query, history)

                input = f"胡桃收到了来自hadi的询问：{query}"
                output = f"进行了思考：{thought},做出了回复：{response}"
                memory = hutao.brain.create_memory(input,output)
                hutao.brain.add_memory(memory)

                history_text = ""  # 初始化历史记录文本
                for chat in history:  # 遍历历史记录
                    history_text = history_text + chat + "\n"
                save_to_file("../resource/conversations.json", history)

                # 获取心情驱动的表情包
                hutao.brain.fsm.mood_transition(input, thought)
                image_path = hutao.brain.fsm.get_current_emoji()

                # 生成响应的音频
                audio_file_path = apis.genshin_tts(text=response.lstrip("胡桃:"), speaker="胡桃")

                if audio_file_path == "Error":
                    default_audio_path = "../resource/audios/这是一段测试音频哟.wav"
                    audio_file_path = default_audio_path

                save_agent_json(hutao.brain)

                return history_text, audio_file_path, image_path  # 返回历史记录文本和音频文件路径

            with gr.Row():
                with gr.Column():
                    # 创建一个用于显示历史记录的文本框
                    history_box = gr.Textbox(lines=10, label="对话历史记录\U0001F4DD")
                    # 创建一个文本框，用于输入文本
                    txt = gr.Textbox(show_label=False, placeholder="输入文本，例如\"胡桃可以给我来一杯咖啡吗？\"\U0001F4AC")

                with gr.Column():
                    # 创建一个音频播放器
                    audio_box = gr.Audio(label="胡桃返回的音频\U0001F3B5")
                    # 创建 Image 组件并设置默认图片
                    image_box = gr.Image(label="胡桃心情对应的表情\U0001F60A",height = 200)

            # 按钮被点击时，调用 talk_with_hutao 函数，并将文本框的内容和状态对象作为参数，将历史记录文本框和音频播放器作为输出
            button = gr.Button("发送 \U0001F600")
            button.click(talk_with_hutao, [txt, state], [history_box, audio_box, image_box])

        with gr.Tab("观察和管理胡桃的Brain模块状态🔧"):
            with gr.Row():
                with gr.Column():
                    agent_state = gr.Textbox(lines=25, max_lines=25, label="胡桃的Brain模块状态\U0001F4C4")
                    button = gr.Button("查询 \U0001F600")
                    button.click(hutao.brain.show_info ,inputs=[], outputs=agent_state)

                with gr.Column():
                    def del_memory(memory_index):
                        memory_str = ""
                        if isinstance(memory_index, int):
                            memory_str = hutao.brain.del_memory(mode="single", index=memory_index)
                            save_agent_json(hutao.brain)
                        if not memory_index:
                            memory_str = "下拉菜单为空或没有接收到下拉菜单的值"

                        memory_keys = list(range(len(hutao.brain.memory_stream)))
                        memory_dropdown = gr.Dropdown(memory_keys, label="要删除的记忆序号\U0001F600")

                        return memory_str, memory_dropdown

                    memory_keys = list(range(len(hutao.brain.memory_stream)))
                    memory_dropdown = gr.Dropdown(memory_keys, label="要删除的记忆序号\U0001F600")
                    memory_deleted = gr.Textbox(label="已删除的胡桃记忆🧠")
                    button = gr.Button("删除记忆🧠")
                    button.click(fn=del_memory, inputs=memory_dropdown, outputs=[memory_deleted,memory_dropdown])

                    def del_knowledge(knowledge_index):
                        knowledge_str = ""
                        if isinstance(knowledge_index, int):
                            knowledge_str = hutao.brain.del_knowledge(mode="single",index=knowledge_index)
                            save_agent_json(hutao.brain)
                        if not knowledge_str:
                            knowledge_str = "下拉菜单为空或没有接收到下拉菜单的值"

                        knowledge_keys = list(range(len(hutao.brain.basic_knowledge)))
                        knowledge_dropdown = gr.Dropdown(knowledge_keys, label="要删除的知识序号\U0001F600")

                        return knowledge_str, knowledge_dropdown

                    knowledge_keys = list(range(len(hutao.brain.basic_knowledge)))
                    knowledge_dropdown = gr.Dropdown(knowledge_keys, label="要删除的知识序号\U0001F600")
                    knowledge_deleted = gr.Textbox(label="已删除的胡桃知识📚")
                    button = gr.Button("删除知识📚")
                    button.click(fn=del_knowledge, inputs=knowledge_dropdown, outputs=[knowledge_deleted,knowledge_dropdown])

        with gr.Tab("注入一些知识\U0001F4D6"):
            def split_text_and_add_to_knowledge(content):
                # 若源内容过长就先切分
                max_unit_length = 500
                split = False

                # 如果 tokens 数量超过了限制，进行切分处理
                if len(content) > max_unit_length:
                    split = True
                    segments = Perception.split_text(content, min_length=max_unit_length, buffer_min_length=int(max_unit_length*0.3))
                    pairs = Perception.get_text_embedding_pairs(segments)
                    hutao.brain.add_knowledge_list(pairs)
                    save_agent_json(hutao.brain)

                    pairs_str = ""
                    for idx, pair in enumerate(pairs, 0):
                        pairs_str += (f"知识单元{idx}\n"
                                      f"知识描述:\n{pair['text']}\n"
                                      f"嵌入向量大小:{len(pair['embedding'])}\n"
                                      f"{'-' * 40}\n")

                    return pairs_str, split

                else:
                    knowledge = hutao.brain.add_knowledge_from_text(content)
                    save_agent_json(hutao.brain)
                    knowledge_str = (f"知识描述:\n{knowledge['text']}\n"
                                     f"嵌入向量大小:{len(knowledge['embedding'])}\n"
                                     f"{'-' * 40}\n")

                    return knowledge_str, split

            def add_knowledge_from_webpage(webpage_content):
                # 从页面输入获得知识
                pairs_str, split = split_text_and_add_to_knowledge(webpage_content)

                if split:
                    webpage_str = (f"从页面输入的内容中获得了以下知识:\n\n"
                                   f"由于知识源文本过长而进行了切分：\n\n{pairs_str}")
                else:
                    webpage_str = (f"从页面输入的内容中获得了以下知识:\n\n"
                                   f"\n\n{pairs_str}")
                return webpage_str

            gr.Interface(fn=add_knowledge_from_webpage,
                         inputs=gr.Textbox(label="输入知识文本📝"),
                         outputs=gr.Textbox(label="从页面输入中提取到的知识📝"),
                         allow_flagging="never")

            def add_knowledge_from_pdf(pdf_path):
                # 对pdf进行切分，直接加载到知识库
                pdf_content = hutao.perception.read_pdf(pdf_path)
                pairs_str, split = split_text_and_add_to_knowledge(pdf_content)

                if split:
                    pdf_str = (f"基于PDF:{pdf_path}加载了如下内容:\n\n{pdf_content}"
                               f"\n\n由于知识源文本过长而进行了切分：\n\n{pairs_str}")
                else:
                    pdf_str = (f"基于PDF:{pdf_path}加载了如下内容:\n\n{pdf_content}"
                               f"\n\n{pairs_str}")
                return pdf_str

            gr.Interface(fn=add_knowledge_from_pdf,
                         inputs=gr.File(label="上传PDF📝"),
                         outputs=gr.Textbox(label="从PDF提取到的知识📝"),
                         allow_flagging="never")

            def add_knowledge_from_wiki(search_query):
                # 获取wiki内容
                wiki_object = hutao.action.use_wiki(search_query)
                wiki_url = wiki_object['url']
                wiki_content = wiki_object['content']
                pairs_str, split = split_text_and_add_to_knowledge(wiki_content)

                if split:
                    wiki_str = (f"基于Wiki的查询:{search_query}\n找到了URL:{wiki_url},内容如下："
                                f"\n\n{wiki_content}\n\n由于知识源文本过长而进行了切分：\n\n{pairs_str}")
                else:
                    wiki_str = (f"基于Wiki的查询:{search_query}\n找到了URL:{wiki_url},内容如下："
                                f"\n\n{wiki_content}\n\n{pairs_str}")
                return wiki_str

            gr.Interface(fn=add_knowledge_from_wiki,
                         inputs=gr.Textbox(label="搜索Wiki百科📝"),
                         outputs=gr.Textbox(label="从Wiki提取到的知识📝"),
                         allow_flagging="never")

        with gr.Tab("常用的Prompts\U0001F4AD"):
            # 加载预制prompt
            def display_prompt(key: str) -> str:
                prompts = load_prompts("../resource/key_prompt.json")
                prompt = prompts.get(key, '没有找到对应的prompt')
                return prompt

            prompt_keys = load_prompts('../resource/key_prompt.json').keys()  # 获取所有的key
            iface = gr.Interface(display_prompt,
                                 gr.Dropdown(prompt_keys, label="需要的Prompt功能🌱"),
                                 gr.Textbox(label="通用Prompt内容💬", show_copy_button = True),
                                 allow_flagging="never")

    demo.queue().launch(share=False)
