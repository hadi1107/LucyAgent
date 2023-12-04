import gradio as gr
import json
from perception import Perception
from action import  Action
from brain import Brain
from lucy_agent import LucyAgent

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
    with open("../resource/胡桃.json", "r", encoding="utf-8") as json_file:
        loaded_data = json.load(json_file)

    perception = Perception()
    action = Action()
    hutao = LucyAgent(perception=perception, brain=Brain.from_json(loaded_data), action=action)

    # 创建一个 Gradio 界面
    with gr.Blocks() as demo:
        # 创建一个状态对象，用于存储历史记录
        state = gr.State([])


        with gr.Tab("和胡桃对话！\U0001F917"):
            def talk_with_hutao(input, history=None):
                if history is None:
                    history = []
                response, history, thought = hutao.brain.cot_chat(input, history)

                input = f"收到了来自hadi的询问：{input}"
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
                default_audio_path = "../resource/audios/这是一段测试音频哟.wav"
                # audio_file_path = apis.genshin_tts(text=response.lstrip("胡桃:"), speaker="胡桃")
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
                        memory_str = hutao.brain.del_memory(mode="single", index=memory_index)
                        save_agent_json(hutao.brain)
                        return memory_str

                    memory_keys = list(range(len(hutao.brain.memory_stream)))
                    gr.Interface(fn=del_memory,
                                 # title="删除记忆🧠",
                                 inputs=gr.Dropdown(memory_keys, label="选择要删除的记忆序号\U0001F600"),
                                 outputs=gr.Textbox(label="已删除的胡桃记忆🧠"),
                                 allow_flagging="never")

                    def del_knowledge(knowledge_index):
                        knowledge_index = int(knowledge_index)
                        knowledge_str = hutao.brain.del_knowledge(mode="single",index=knowledge_index)
                        save_agent_json(hutao.brain)
                        return knowledge_str

                    knowledge_keys = list(range(len(hutao.brain.basic_knowledge)))
                    gr.Interface(fn=del_knowledge,
                                 # title="删除知识📚",
                                 inputs=gr.Dropdown(knowledge_keys, label="选择要删除的知识序号\U0001F600"),
                                 outputs=gr.Textbox(label="已删除的胡桃知识📚"),
                                 allow_flagging="never")


        with gr.Tab("注入一些知识\U0001F4D6"):
            def split_text_and_add_to_knowledge(content):
                # 若源内容过长就先切分
                tokens = len(content)
                max_tokens = 500
                splited = False

                # 如果 tokens 数量超过了限制，进行切分处理
                if tokens > max_tokens:
                    splited = True
                    segments = Perception.split_text(content, min_length=max_tokens, buffer_min_length=int(max_tokens*0.3))
                    pairs = Perception.get_text_embedding_pairs(segments)
                    hutao.brain.add_knowledge_list(pairs)
                    save_agent_json(hutao.brain)

                    pairs_str = ""
                    for idx, pair in enumerate(pairs, 0):
                        pairs_str += (f"知识单元{idx}\n"
                                      f"知识描述:\n{pair['text']}\n"
                                      f"嵌入向量大小:{len(pair['embedding'])}\n"
                                      f"{'-' * 40}\n")

                    return pairs_str, splited

                else:
                    knowledge = hutao.brain.add_knowledge_from_text(content)
                    save_agent_json(hutao.brain)
                    knowledge_str = (f"知识描述:\n{knowledge['text']}\n"
                                     f"嵌入向量大小:{len(knowledge['embedding'])}\n"
                                     f"{'-' * 40}\n")

                    return knowledge_str, splited

            def add_knowledge_from_pdf(pdf_path):
                # 对pdf进行切分，直接加载到知识库
                pdf_content = hutao.perception.read_pdf(pdf_path)
                pairs_str, splited = split_text_and_add_to_knowledge(pdf_content)

                if splited:
                    pdf_str = (f"基于PDF:{pdf_path}加载了如下内容:\n\n{pdf_content}"
                               f"由于知识源文本过长而进行了切分：\n\n{pairs_str}")
                else:
                    pdf_str = (f"基于PDF:{pdf_path}加载了如下内容:\n\n{pdf_content}"
                               f"\n\n{pairs_str}")
                return pdf_str

            gr.Interface(fn=add_knowledge_from_pdf,
                         title="将PDF文件的知识注入到知识库\U0001F600",
                         inputs=gr.File(label="上传PDF📝"),
                         outputs=gr.Textbox(label="从PDF提取到的知识📝"),
                         allow_flagging="never")

            def add_knowledge_from_wiki(search_query):
                # 获取wiki内容
                wiki_object = hutao.action.use_wiki(search_query)
                wiki_url = wiki_object['url']
                wiki_content = wiki_object['content']

                pairs_str, splited = split_text_and_add_to_knowledge(wiki_content)
                if splited:
                    wiki_str = (f"基于Wiki的查询:{search_query}\n找到了url:{wiki_url},内容如下："
                                f"\n\n{wiki_content}\n\n由于知识源文本过长而进行了切分：\n\n{pairs_str}")
                else:
                    wiki_str = (f"基于Wiki的查询:{search_query}\n找到了url:{wiki_url},内容如下："
                                f"\n\n{wiki_content}\n\n{pairs_str}")
                return wiki_str

            gr.Interface(fn=add_knowledge_from_wiki,
                         title="从Wiki检索知识并注入到知识库\U0001F600",
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
