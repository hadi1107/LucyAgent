import gradio as gr
import json
from perception import Perception
from brain import LucyAgent
from brain import Brain

def save_to_file(file_path:str, conversations)-> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)

def load_prompts(json_file: str) -> dict:
    with open(json_file, 'r',encoding='utf-8') as f:
        data = json.load(f)
    return {item['key']: item['prompt'] for item in data}

if __name__ == "__main__":
    with open("../resource/hutao.json", "r", encoding="utf-8") as json_file:
        loaded_data = json.load(json_file)

    perception = Perception()
    hutao = LucyAgent(perception=perception, brain=Brain.from_json(loaded_data), action=None)

    # 创建一个 Gradio 界面
    with gr.Blocks() as demo:
        # 创建一个状态对象，用于存储历史记录
        state = gr.State([])

        with gr.Tab("Talk with Hutao"):
            def mypredict(input, history=None):
                if history is None:
                    history = []
                response, history, thought = hutao.brain.cot_chat(input, history)

                # input = f"收到了来自hadi的询问：{input}"
                # output = f"进行了思考：{thought},做出了回复：{response}"
                # memory = hutao.brain.create_memory(input,output)
                # print(memory)
                # hutao.brain.add_memory(memory)

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

                try:
                    with open("../resource/hutao.json", "w", encoding="utf-8") as json_file:
                        json.dump(hutao.brain.to_json(), json_file, indent=4, ensure_ascii=False)
                except IOError:
                    print("无法写入文件。")

                return history_text, audio_file_path, image_path  # 返回历史记录文本和音频文件路径

            with gr.Row():
                with gr.Column():
                    # 创建一个用于显示历史记录的文本框
                    history_box = gr.Textbox(lines=10, label="Talk history")
                    # 创建一个文本框，用于输入文本
                    txt = gr.Textbox(show_label=False, placeholder="Input text,such as \"胡桃可以给我来一杯咖啡吗？\"")

                with gr.Column():
                    # 创建一个音频播放器
                    audio_box = gr.Audio(label = "Hutao's audio")
                    # 创建 Image 组件并设置默认图片
                    image_box = gr.Image(label="Hutao's mood",height = 200)

            # 创建一个按钮，当按钮被点击时，调用 mypredict 函数，并将文本框的内容和状态对象作为参数，将历史记录文本框和音频播放器作为输出
            button = gr.Button("发送 \U0001F600")
            button.click(mypredict, [txt, state], [history_box, audio_box, image_box])

        with gr.Tab("Observe Hutao"):
            agent_state = gr.Textbox(label="Hutao's state")
            button = gr.Button("查询 \U0001F600")
            button.click(hutao.brain.show_info ,inputs=[], outputs=agent_state)

        with gr.Tab("Read PDF"):
            def turn_pdf_into_segments(pdf_path):
                text = hutao.perception.read_pdf(pdf_path)
                segments = hutao.perception.split_text(text)
                segments_str = ""
                for idx,segment in enumerate(segments,1):
                    segments_str += f"文本段{idx}\n描述:\n{segment}\n{'-' * 40}\n"

                return segments_str

            gr.Interface(fn=turn_pdf_into_segments, inputs="file", outputs="text",allow_flagging="never")

        with gr.Tab("常用prompt"):
            # 加载预制prompt
            def display_prompt(key: str) -> str:
                prompts = load_prompts("../resource/key_prompt.json")
                prompt = prompts.get(key, '没有找到对应的prompt')
                return prompt

            keys = load_prompts('../resource/key_prompt.json').keys()  # 获取所有的key
            iface = gr.Interface(display_prompt, gr.Dropdown(keys, label="prompt功能"), gr.Textbox( label="通用prompt",show_copy_button = True),allow_flagging="never")

    demo.queue().launch(share=False)
