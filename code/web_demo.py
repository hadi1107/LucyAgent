import gradio as gr
import json
import apis
from perception import Perception
from action import Action
from brain import Brain
from lucy_agent import LucyAgent

# 简单的事件模拟，本来应该在沙盒环境里面去定义。沙盒环境相关工程量太大了，暂时没做。
PERCEPTION_LIST = [
    "胡桃听到了hadi在打招呼。",
    "胡桃收到了璃月管委会的消息,内容为：往生堂第七十七代堂主,您好。近来璃月要举行一场特别的送别之仪,请您着手策划。",
    "胡桃收到了咖啡店前台系统的消息,内容为：hadi点了一杯拿铁。",
    "胡桃听到了自己的闹钟响了,查看后发现备注为：记得看看璃月的历史书！"
]

with open("../resource/hutao.json", "r", encoding="utf-8") as json_file:
    loaded_data = json.load(json_file)

perception = Perception()
action = Action()
brain = Brain.from_json(loaded_data)
hutao = LucyAgent(perception, brain, action)

def save_to_file(file_path:str, conversations)-> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)

def load_prompts(json_file: str) -> dict:
    with open(json_file, 'r',encoding='utf-8') as f:
        data = json.load(f)
    return {item['key']: item['prompt'] for item in data}

def save_agent_json(agent_brain):
    # 默认存到 “中文name.json” 当中，不影响用来初始化的文件
    name = agent_brain.name
    file_path = f"../resource/{name}.json"
    try:
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(agent_brain.to_json(), json_file, indent=4, ensure_ascii=False)
    except IOError:
        print("无法写入文件。")

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

def show_action_state():
    scene_path = "../resource/pictures/hutao_xiuxi.webp"
    if hutao.brain.fsm.action_state == "休息":
        scene_path = "../resource/pictures/hutao_xiuxi.webp"
    if hutao.brain.fsm.action_state == "看璃月的历史书":
        scene_path = "../resource/pictures/hutao_kanshu.webp"
    if hutao.brain.fsm.action_state == "策划往生堂的特别活动":
        scene_path = "../resource/pictures/hutao_cehua.jfif"
    if hutao.brain.fsm.action_state == "做咖啡并递交给客户":
        scene_path = "../resource/pictures/hutao_coffee.webp"
    if hutao.brain.fsm.action_state == "回复问题和聊天":
        scene_path = "../resource/pictures/hutao_yao.webp"
    return f"胡桃正在{hutao.brain.fsm.action_state}", scene_path


def user(query, gr_states, history):
    gr_states.append([query, None])
    history.append([query, None])
    print(f"query, gr_states, history:{query ,gr_states, history}")
    return "", gr_states, history
    

def bot(gr_states, history):
    query = history[-1][0]
    if hutao.brain.fsm.action_state != "回复问题和聊天":
        action_state_str, scene_path = show_action_state()
        gr_states[-1][1] = action_state_str
        history[-1][1] = action_state_str
        print(f"gr_states, history:{gr_states, history}")
        return gr_states, history, scene_path
    
    if not query:
        gr_states[-1][1] = "请不要不说话嘞"
        history[-1][1] = "请不要不说话嘞"
        print(f"gr_states, history:{gr_states, history}")
        return gr_states, history, "../resource/pictures/hutao_naohuo.webp"

    response, _, thought = hutao.brain.cot_chat(query, history)
    gr_states[-1][1] = response
    history[-1][-1] = response

    input = f"胡桃收到了来自hadi的询问：{query}"
    output = f"进行了思考：{thought},做出了回复：{response}"
    memory = hutao.brain.create_memory(input,output)
    hutao.brain.add_memory(memory)

    # 获取心情驱动的表情包
    hutao.brain.fsm.mood_transition(input, thought)
    image_path = hutao.brain.fsm.get_current_emoji()

    save_agent_json(hutao.brain)
    print(f"gr_states, history:{gr_states, history}")
    return gr_states, history, image_path


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


def del_knowledge(knowledge_index):
    knowledge_str = ""
    if isinstance(knowledge_index, int):
        knowledge_str = hutao.brain.del_knowledge(mode="single", index=knowledge_index)
        save_agent_json(hutao.brain)
    if not knowledge_str:
        knowledge_str = "下拉菜单为空或没有接收到下拉菜单的值"

    knowledge_keys = list(range(len(hutao.brain.basic_knowledge)))
    knowledge_dropdown = gr.Dropdown(knowledge_keys, label="要删除的知识序号\U0001F600,注意子知识也会被删除")

    return knowledge_str, knowledge_dropdown

def split_text_and_add_to_knowledge(content, summary_text):
    # 若源内容过长就先切分
    max_unit_length = 500
    split = False

    # 如果 tokens 数量超过了限制，进行切分处理
    if len(content) > max_unit_length:
        split = True
        knowledge_str = ""
        segments = Perception.split_text(content,
                                            min_length=max_unit_length,
                                            buffer_min_length=int(max_unit_length*0.3))
        knowledge_list = Perception.generate_knowledge_units(segments)
        sub_knowledge_file = hutao.brain.add_knowledge_from_sub_knowledge_list(summary_text, knowledge_list)
        knowledge_str += f"加入到知识库中的根知识为:{summary_text}\n\n其子知识文件路径为:{sub_knowledge_file}\n\n"
        for idx, knowledge in enumerate(knowledge_list, 0):
            knowledge_str += (f"知识单元{idx}\n"
                                f"知识描述:\n{knowledge['text']}\n"
                                f"嵌入向量大小:{len(knowledge['embedding'])}\n"
                                f"子知识文件路径：{knowledge['sub_knowledge']}\n"
                                f"{'-' * 40}\n")
        save_agent_json(hutao.brain)

        return knowledge_str, split

    else:
        knowledge = hutao.brain.add_knowledge_from_text(content)
        save_agent_json(hutao.brain)
        knowledge_str = (f"知识描述:\n{knowledge['text']}\n"
                            f"嵌入向量大小:{len(knowledge['embedding'])}\n"
                            f"由于输入的知识文本较短，没有发生切分或产生子知识文件\n"
                            f"{'-' * 40}\n")

        return knowledge_str, split


def add_knowledge_from_webpage(webpage_content, summary_text):
    if not webpage_content:
        return "请输入要添加的知识内容哟"
    if not summary_text:
        return "请输入总结文本！实在麻烦就复制一点~"
    # 从页面输入获得知识
    knowledge_str, split = split_text_and_add_to_knowledge(webpage_content, summary_text)

    if split:
        webpage_str = (f"从页面输入的内容中获得了以下知识:\n\n"
                        f"由于知识源文本过长而进行了切分,并以总结文本为根生成了子知识文件：\n\n{knowledge_str}")
    else:
        webpage_str = (f"从页面输入的内容中获得了以下知识:\n\n"
                        f"\n\n{knowledge_str}")
    return webpage_str


if __name__ == "__main__":
    # 创建一个 Gradio 界面
    with gr.Blocks() as demo:
        # 创建一个状态对象，用于存储历史记录
        gr_states = gr.State([])
        with gr.Tab("和胡桃互动！\U0001F917"):
            state_box = gr.Textbox(label="胡桃的行动情况🚴‍♂️")
            image_box = gr.Image(label="胡桃正在干什么？\U0001F60A", height=300)
            query_button = gr.Button("查询胡桃的行动情况🤶")
            query_button.click(show_action_state, inputs=[], outputs=[state_box, image_box])

            events_dropdown = gr.Dropdown(choices=PERCEPTION_LIST,
                                          label="可以模拟的事件列表🧐,也可以试试自己写！",
                                          allow_custom_value=True)
            event_button = gr.Button("模拟一些事件的发生⚡")
            event_button.click(perceive_and_change_action, inputs=events_dropdown, outputs=[state_box, image_box])


        with gr.Tab("chatbot\U0001F917"):
            with gr.Row():
                with gr.Column(scale = 7):
                    with gr.Row():
                        chatbot = gr.Chatbot(label='聊天界面', value=[], render_markdown=False, height=800, visible=True)
                    with gr.Row():
                        msg = gr.Textbox(label='对话输入框（按Enter发送消息）', interactive=True, visible=True)
                with gr.Column(scale=3):
                    image_box = gr.Image(label="胡桃心情对应的表情\U0001F60A",height = 300, value="../resource/pictures/hutao_xiuxi.webp",
                                         show_download_button=False)
            
        msg.submit(user,[msg, gr_states, chatbot],[msg, gr_states, chatbot]).then(bot, [gr_states, chatbot], [gr_states, chatbot, image_box])

        with gr.Tab("观察和管理胡桃的Brain模块状态🔧"):
            with gr.Row():
                with gr.Column():
                    agent_state = gr.Textbox(lines=25, max_lines=25, label="胡桃的Brain模块状态\U0001F4C4")
                    button = gr.Button("查询 \U0001F600")
                    button.click(hutao.brain.show_info, inputs=[], outputs=agent_state)

                with gr.Column():
                    memory_keys = list(range(len(hutao.brain.memory_stream)))
                    memory_dropdown = gr.Dropdown(memory_keys, label="要删除的记忆序号\U0001F600")
                    memory_deleted = gr.Textbox(label="已删除的胡桃记忆🧠")
                    button = gr.Button("删除记忆🧊")
                    button.click(fn=del_memory,
                                 inputs=memory_dropdown,
                                 outputs=[memory_deleted, memory_dropdown])

                    knowledge_keys = list(range(len(hutao.brain.basic_knowledge)))
                    knowledge_dropdown = gr.Dropdown(knowledge_keys, label="要删除的知识序号\U0001F600")
                    knowledge_deleted = gr.Textbox(label="已删除的胡桃知识📝")
                    button = gr.Button("删除知识🧊")
                    button.click(fn=del_knowledge,
                                 inputs=knowledge_dropdown,
                                 outputs=[knowledge_deleted, knowledge_dropdown])

        with gr.Tab("注入一些知识\U0001F4D6"):
            gr.Interface(fn=add_knowledge_from_webpage,
                         inputs=[gr.Textbox(label="直接输入知识文本📝"),
                                 gr.Textbox(label="输入知识文本的总结📝,这对分块索引很重要")],
                         outputs=gr.Textbox(label="从页面输入中提取到的知识📝"),
                         allow_flagging="never")

            def add_knowledge_from_pdf(pdf_path, summary_text):
                if not pdf_path:
                    return "请上传PDF哟"
                if not summary_text:
                    return "请输入总结文本！实在麻烦就复制一点~"
                # 对pdf进行切分，直接加载到知识库
                pdf_content = hutao.perception.read_pdf(pdf_path)
                pairs_str, split = split_text_and_add_to_knowledge(pdf_content, summary_text)

                if split:
                    pdf_str = (f"基于PDF:{pdf_path}加载了如下内容:\n\n{pdf_content}"
                               f"\n\n由于知识源文本过长而进行了切分,并以总结文本为根生成了子知识文件：\n\n{pairs_str}")
                else:
                    pdf_str = (f"基于PDF:{pdf_path}加载了如下内容:\n\n{pdf_content}"
                               f"\n\n{pairs_str}")
                return pdf_str

            gr.Interface(fn=add_knowledge_from_pdf,
                         inputs=[gr.File(label="上传PDF📝"),
                                 gr.Textbox(label="输入知识文本的总结📝,这对分块索引很重要")],
                         outputs=gr.Textbox(label="从PDF提取到的知识📝"),
                         allow_flagging="never")

            def add_knowledge_from_wiki(search_query, summary_text):
                if not search_query:
                    return "请输入要搜索的wiki对象名哟"
                if not summary_text:
                    return "请输入总结文本！实在麻烦就复制一点~"
                # 获取wiki内容
                wiki_object = hutao.action.use_wiki(search_query)
                wiki_url = wiki_object['url']
                wiki_content = wiki_object['content']
                pairs_str, split = split_text_and_add_to_knowledge(wiki_content, summary_text)

                if split:
                    wiki_str = (f"基于Wiki的查询:{search_query}\n找到了URL:{wiki_url},内容如下："
                                f"\n\n{wiki_content}\n\n由于知识源文本过长而进行了切分,并以总结文本为根生成了子知识文件：\n\n{pairs_str}")
                else:
                    wiki_str = (f"基于Wiki的查询:{search_query}\n找到了URL:{wiki_url},内容如下："
                                f"\n\n{wiki_content}\n\n{pairs_str}")
                return wiki_str

            gr.Interface(fn=add_knowledge_from_wiki,
                         inputs=[gr.Textbox(label="搜索Wiki百科📝"),
                                 gr.Textbox(label="输入知识文本的总结📝,这对分块索引很重要")],
                         outputs=gr.Textbox(label="从Wiki提取到的知识📝"),
                         allow_flagging="never")

        with gr.Tab("管理子知识文件\U0001F4D6"):
            summary_to_path_map = {}
            sub_knowledge_list = Brain.get_all_sub_knowledge()
            if sub_knowledge_list:
                summary_to_path_map = {item["summary_text"]: item["file_path"]
                                       for item in sub_knowledge_list}
            sub_knowledge_summary_text_list = list(summary_to_path_map.keys())
            sub_knowledge_dropdown = gr.Dropdown(label="子知识文件列表📝", choices=sub_knowledge_summary_text_list)
            def show_sub_knowledge(summary_text):
                # 使用总结文本来找文件
                if not summary_text:
                    return "下拉菜单为空或没有接收到下拉菜单的值"
                # 使用映射来查找对应的文件路径
                file_path = summary_to_path_map[summary_text]
                # 读取文件并返回内容
                return Brain.show_sub_knowledge(file_path)

            def add_knowledge_to_sub_knowledge_file(summary_text, knowledge_text):
                # 使用总结文本来找文件
                if not summary_text:
                    return "下拉菜单为空或没有接收到下拉菜单的值"
                # 使用映射来查找对应的文件路径
                file_path = summary_to_path_map[summary_text]
                # 添加知识
                return Brain.add_knowledge_to_sub_knowledge_file(file_path, knowledge_text)

            def del_knowledge_from_sub_knowledge_file(summary_text, del_knowledge_index):
                # 使用总结文本来找文件
                if not summary_text:
                    return "下拉菜单为空或没有接收到下拉菜单的值"
                # 使用映射来查找对应的文件路径
                file_path = summary_to_path_map[summary_text]
                # 删除知识
                try:
                    int(del_knowledge_index)
                except Exception as e:
                    return f"输入内容无法转型为下标，请输入数值"
                return Brain.del_knowledge_from_sub_knowledge_file(file_path, int(del_knowledge_index))

            with gr.Row():
                with gr.Column():
                    sub_knowledge_text = gr.Textbox(label="子知识文件的内容📝")
                    show_sub_knowledge_button = gr.Button(value="浏览子知识内容📝")
                    show_sub_knowledge_button.click(fn=show_sub_knowledge,
                                                    inputs=sub_knowledge_dropdown,
                                                    outputs=sub_knowledge_text)
                with gr.Column():
                    add_knowledge_text = gr.Textbox(label="给子知识文件添加知识📚")
                    add_knowledge_button = gr.Button("添加知识📚")
                    del_knowledge_index = gr.Textbox(label="删除子知识文件的部分知识🧊")
                    del_knowledge_button = gr.Button("删除知识🧊")

                    feedback_text = gr.Textbox(label="相关操作的反馈📝")
                    add_knowledge_button.click(fn=add_knowledge_to_sub_knowledge_file,
                                               inputs=[sub_knowledge_dropdown, add_knowledge_text],
                                               outputs=feedback_text)
                    del_knowledge_button.click(fn=del_knowledge_from_sub_knowledge_file,
                                               inputs=[sub_knowledge_dropdown, del_knowledge_index],
                                               outputs=feedback_text)

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

    demo.queue().launch(share=True)
