import gradio as gr
import apis
import json

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

def chat(input, history):
    """
    这个函数接收用户的输入和历史记录，然后将用户的输入作为响应返回，并将其添加到历史记录中。
    同时调用 genshin_tts 函数生成音频。

    参数:
    input: 用户的输入
    history: 聊天的历史记录

    返回:
    response: 聊天机器人的响应
    history: 更新后的历史记录
    audio_file_path: 聊天机器人的响应音频文件路径
    """
    response = f"响应:{input}"  # 将用户的输入作为响应
    audio_file_path = apis.genshin_tts(text=response, speaker="银狼")  # 生成响应的音频
    history.append((input, response, audio_file_path))  # 将用户的输入、响应和音频路径添加到历史记录中
    conversations = []
    for chat in history:
        conversation = {
            "prompt":chat[0],
            "response":chat[1]
        }
        conversations.append(conversation)
    save_to_file("../resource/conversations.json",conversations)
    return response, history, audio_file_path

def mypredict(input, history=None):
    """
    调用chat函数，返回历史信息和音频地址，供gradio界面使用

    参数:
    input: 用户的输入
    history: 聊天的历史记录

    返回:
    history_text：解析出的历史文本
    audio_file_path：音频地址
    """
    if history is None:  # 如果历史记录为空，则初始化为空列表
        history = []
    response, history, audio_file_path = chat(input, history)  # 调用 chat 函数获取响应和更新后的历史记录
    history_text = ""  # 初始化历史记录文本
    for query, response, audio in history:  # 遍历历史记录
        history_text += "user：" + query + "\n"  # 将用户的查询添加到历史记录文本中
        history_text += "agent：" + response + "\n"  # 将聊天机器人的响应添加到历史记录文本中
    return history_text, audio_file_path  # 返回历史记录文本和音频文件路径

if __name__ == "__main__":
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

        # 加载预制prompt
        keys = load_prompts('../resource/key_prompt.json').keys()  # 获取所有的key
        iface = gr.Interface(display_prompt, gr.Dropdown(keys, label="prompt功能"), gr.Textbox( label="通用prompt",show_copy_button = True),allow_flagging="never")

        # 创建一个按钮，当按钮被点击时，调用 mypredict 函数，并将文本框的内容和状态对象作为参数，将历史记录文本框和音频播放器作为输出
        button = gr.Button("发送 \U0001F600")
        button.click(mypredict, [txt, state], [history_box, audio_box])

    demo.queue().launch()
