"""
apis.py:定义需要apikey的api调用功能
"""
import os
import re
import json
import openai
import requests

# ！！！！！！应当重构到环境变量当中
openai.api_key = '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'
tts_api_key = '3c1c2c0822b56c92ceb5cd46d30e497c'
bing_api_key = "d6fca707accf4a83a7afe9b3db0442bd"

def chatgpt(prompt:str) -> str:
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
            timeout = 10,  # 设置请求超时时间为10秒
            temperature=0.8,  # 控制输出的随机性，值越高输出越随机，值越低输出越确定
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
            prompt=prompt,  # 用户的输入消息
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
        return f"An error occurred: {e}"

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

            # 返回音频文件
            return audio_file_path
        else:
            print("Error: ", response.status_code)
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
    search_result = bing_search("知乎热榜")
    print(search_result)
