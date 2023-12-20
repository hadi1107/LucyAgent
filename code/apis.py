"""
apis.py:定义需要apikey的api调用功能
"""
import os
import re
import json
import openai
import requests
import tiktoken
from gradio_client import Client

# openai接入点
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
openai_api_base = "https://api.openai.com/v1/embeddings"
openai.api_key = os.getenv('OPENAI_API_KEY')

# 外部api服务接入点
tts_api_key = os.getenv('TTS_API_KEY')
bing_api_key = os.getenv('BING_API_KEY')

# easygpt接入点
easygpt_api_base = "https://chat.eqing.tech/v1/chat/completions"
easygpt_api_key = os.getenv('EASYGPT_API_KEY')

def request_chatgpt(prompt, temperature=0.8, api_key=easygpt_api_key, url=easygpt_api_base):
    """
    以requests库的方式调用自定义接入点的GPT模型进行聊天。

    参数:
    prompt: 用户的输入消息。
    temperature: 控制回答的随机性。
    api_key: OpenAI提供的API密钥。
    url: API的URL。

    返回:
    GPT模型的回复消息。
    """

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    try:
        response = requests.post(url=url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

def request_embedding(things:list or str, api_key=openai.api_key, url=openai_api_base):
    """
    以requests库的方式调用自定义接入点的text-embedding-ada-002模型获取输入词语的embedding。

    参数:
    things: 需要获取embedding的词语，可以是一个词语的字符串或者是多个词语的列表。

    返回:
    输入词语的embedding列表。
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    data = {
        'input': things,  # 需要获取embedding的词语
        'model': "text-embedding-ada-002",  # 使用text-embedding-ada-002模型
    }

    try:
        response = requests.post(url=url, headers=headers, json=data)
        print(response.json())
        # 获取返回的embedding数据
        data = response.json()["data"]
        embeddings = [item['embedding'] for item in data]
        return embeddings

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def chatgpt(prompt:str, temperature = 0.8) -> str:
    """
    使用OpenAI的GPT-3.5-turbo模型进行聊天。

    参数:
    prompt: 用户的输入消息。

    返回:
    GPT模型的回复消息。
    """
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            timeout=10,  # 设置请求超时时间为10秒
            temperature=temperature,  # 控制输出的随机性，值越高输出越随机，值越低输出越确定
        )
        response = completion.choices[0].message["content"]
        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

def instructgpt(prompt:str) -> str:
    """
    使用OpenAI的GPT-3.5-turbo-instruct模型。

    参数:
    prompt: 用户的输入消息。

    返回:
    GPT模型的回复消息。
    """
    try:
        # 创建一个指令完成任务
        completion = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",  # 使用GPT-3.5-turbo-instruct模型
            prompt=prompt,
            temperature=0  # 控制输出的随机性，值为0表示输出确定性最高
        )
        response = completion.choices[0]['text']
        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

def embedding(things:list or str) -> list:
    """
    使用OpenAI的text-embedding-ada-002模型获取输入词语的embedding。

    参数:
    things: 需要获取embedding的词语，可以是一个词语的字符串或者是多个词语的列表。

    返回:
    输入词语的embedding列表。
    """
    try:
        # 创建一个embedding任务
        response = openai.Embedding.create(
            input=things,  # 需要获取embedding的词语
            model="text-embedding-ada-002"  # 使用text-embedding-ada-002模型
        )

        # 获取返回的embedding数据
        data = response['data']
        embeddings = [item['embedding'] for item in data]
        return embeddings

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def tokens(text):
    num_tokens = len(encoding.encode(text))
    print("Token count:", num_tokens)
    return num_tokens

def genshin_tts_v2(prompt, speaker):
    client = Client("https://v2.genshinvoice.top/", output_dir="../resource/audios")
    try:
        result = client.predict(
            prompt,
            speaker,
            0.2,  # float (numeric value between 0 and 1) in 'SDP Ratio' Slider component
            0.6,  # float (numeric value between 0.1 and 2) in 'Noise' Slider component
            0.8,  # float (numeric value between 0.1 and 2) in 'Noise_W' Slider component
            1,  # float (numeric value between 0.1 and 2) in 'Length' Slider component
            "ZH,ZH",  # str (Option from: ['ZH', 'JP', 'EN', 'mix', 'auto']) in 'Language' Dropdown component
            True,  # bool  in '按句切分    在按段落切分的基础上再按句子切分文本' Checkbox component
            2,  # int | float (numeric value between 0 and 10) in '段间停顿(秒)，需要大于句间停顿才有效' Slider component
            0.5,  # int | float (numeric value between 0 and 5) in '句间停顿(秒)，勾选按句切分才生效' Slider component
            "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",
            "Howdy!",  # str  in 'Text prompt' Textbox component
            "Howdy!",
            0.1,  # int | float (numeric value between 0 and 1) in 'Weight' Slider component
            fn_index=2
        )
        print(result)
        return result[1]
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error"

def genshin_tts(text:str, speaker:str) -> str:
    """
    使用原神tts-api获取音频wav文件

    参数:
    text:转语音的文本
    speaker:转语音的音色

    返回:
    音频文件的地址
    """
    url = "https://tts.ai-lab.top"
    data = {
        "token": tts_api_key,
        "speaker": speaker,
        "text": text,
        "sdp_ratio": 0.2,
        "noise": 0.5,
        "noisew": 0.9,
        "length": 1.0
    }
    try:
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 200:
            response_data = response.json()
            print("Audio URL: ", response_data["audio"])
            print("Message: ", response_data["message"])

            # 创建音频文件夹
            audio_folder = "../resource/audios"
            if not os.path.exists(audio_folder):
                os.makedirs(audio_folder)

            # 清理文本，移除或替换特殊字符,使用清理后的文本作为文件名
            clean_text = re.sub(r"[<>:\"/\\|?*]+", "_", text)
            truncated_text = clean_text[:10]
            audio_file_name = f"{truncated_text}.wav"
            audio_url = response_data["audio"]
            audio_file_path = os.path.join(audio_folder, audio_file_name)

            # get音频文件
            response = requests.get(audio_url)
            with open(audio_file_path, 'wb') as f:
                f.write(response.content)

            # 返回音频文件路径
            return audio_file_path
        else:
            print("Error: ", response.status_code)
            return "Error"
    except requests.exceptions.Timeout:
        print("Timeout occurred")
    except Exception as e:
        print(f"An error occurred: {e}")

def bing_search(query: str, mkt: str = "zh-CN") -> list:
    """
    使用Bing搜索API搜索指定的查询字符串，并返回搜索结果的网页信息。

    参数:
    query: str - 搜索查询字符串
    mkt: str - 市场代码，默认为"zh-CN"

    返回:
    search_results: 搜索结果的网页信息list，元素为url和概述的字典
    """
    try:
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": bing_api_key
        }
        params = {
            "q": query,
            "count": 3,
            "mkt": mkt
        }

        response = requests.get(url, headers=headers, params=params)
        response = response.json()
        # 将搜索结果保存到一个临时JSON文件中
        with open("../resource/bing_temp.json", 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=4)

        # 解析搜索结果
        search_results = []
        page_values = response["webPages"]["value"]
        for value in page_values:
            search_results.append(
                {
                    "url": value["url"],
                    "summary": value["snippet"]
                }
            )

        if response.get("news"):
            news_values = response["news"]["value"]
            for value in news_values:
                search_results.append(
                    {
                        "url": value["url"],
                        "summary": value["description"]
                    }
                )

        return search_results
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

if __name__ == "__main__":
    # print(request_embedding("你好"))
    # print(request_chatgpt("你好"))
    print(genshin_tts("我不在！有事请留言哟","胡桃"))
