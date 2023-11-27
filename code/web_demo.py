import gradio as gr
import apis
import json
from lucy_agent import LucyAgent
from brain import Brain

def save_to_file(file_path:str, conversations)-> None:
    """
    用于将对话列表保存到文件中

    参数:
    file_path: 文件地址
    conversations:对话列表
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)

def load_prompts(json_file: str) -> dict:
    """
    从json文件中加载提示信息

    参数:
    json_file: json文件的路径
    """
    with open(json_file, 'r',encoding='utf-8') as f:
        data = json.load(f)
    return {item['key']: item['prompt'] for item in data}

def display_prompt(key: str) -> str:
    """
    显示选定的提示信息并复制到剪贴板

    参数:
    key: 选定的提示的key

    返回:
    prompt: 选定的提示的key对应的prompt
    """
    prompts = load_prompts("../resource/key_prompt.json")
    prompt = prompts.get(key, '没有找到对应的prompt')
    return prompt

def mypredict(input, history=None):
    """
    调用hutao.brain.chat函数，返回历史信息和音频地址，供gradio界面使用

    参数:
    input: 用户的输入
    history: 聊天的历史记录

    返回:
    history_text：解析出的历史文本
    audio_file_path：音频地址
    """
    if history is None:  # 如果历史记录为空，则初始化为空列表
        history = []
    # 调用hutao.brain.chat函数获取响应和更新后的历史记录
    response, history = hutao.brain.chat(input, history)
    # 生成响应的音频
    audio_file_path = apis.genshin_tts(text=response.lstrip("胡桃:"), speaker="胡桃")
    history_text = ""  # 初始化历史记录文本
    for chat in history:  # 遍历历史记录
        history_text = history_text + chat + "\n"

    save_to_file("../resource/conversations.json",history)
    return history_text, audio_file_path  # 返回历史记录文本和音频文件路径

if __name__ == "__main__":
    try:
        with open("../resource/hutao.json", "r", encoding="utf-8") as json_file:
            loaded_data = json.load(json_file)
    except FileNotFoundError:
        print("未找到指定的文件。")
        exit()

    hutao = LucyAgent(perception=None, brain=Brain.from_json(loaded_data), action=None)

    # 打印状态
    hutao.brain.show_memory()
    hutao.brain.show_knowledge()

    # 将更新后的大脑状态保存到JSON文件中。
    try:
        with open("../resource/hutao.json", "w", encoding="utf-8") as json_file:
            json.dump(hutao.brain.to_json(), json_file, indent=4, ensure_ascii=False)
    except IOError:
        print("无法写入文件。")

    # 创建一个 Gradio 界面
    with gr.Blocks() as demo:
        # 创建一个状态对象，用于存储历史记录
        state = gr.State([])

        # 创建一个用于显示历史记录的文本框
        history_box = gr.Textbox(lines=10, label="对话记录")

        # 创建一个音频播放器
        audio_box = gr.Audio()

        # 创建一个文本框，用于输入文本
        txt = gr.Textbox(show_label=False, placeholder="输入文本")

        # 创建一个按钮，当按钮被点击时，调用 mypredict 函数，并将文本框的内容和状态对象作为参数，将历史记录文本框和音频播放器作为输出
        button = gr.Button("发送 \U0001F600")
        button.click(mypredict, [txt, state], [history_box, audio_box])

        # 加载预制prompt
        keys = load_prompts('../resource/key_prompt.json').keys()  # 获取所有的key
        iface = gr.Interface(display_prompt, gr.Dropdown(keys, label="prompt功能"), gr.Textbox( label="通用prompt",show_copy_button = True),allow_flagging="never")

    demo.queue().launch()
