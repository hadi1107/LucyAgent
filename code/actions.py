import openai
import json
import apis
import tools

def chatgpt_function(prompt: str):
    """
    使用chatgpt的function call功能

    参数：
    prompt: 用户输入

    返回：
    chatgpt输出或function调用结果
    """
    functions = [
        {
            "name": "play_wav",
            "description": f"通过电脑音频播放\".wav\"音频",
            "parameters": {
                "type": "object",
                "properties": {
                    "wave_file": {
                        "type": "string",
                        "description": "the audio file path",
                    }
                },
                "required": ["wave_file"],
            }
        },
        {
            "name": "web_search",
            "description": f"通过联网搜索获得时效性信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the topic to search",
                    }
                },
                "required": ["query"],
            }
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        functions=functions,
        function_call="auto",
    )
    message = response["choices"][0]["message"]

    if (message.get("function_call")):
        # 解析函数名和函数参数
        function_name = message["function_call"]["name"]
        arguments = json.loads(message["function_call"]["arguments"])
        # 播放本地音频
        if (function_name == "play_wav"):
            try:
                wave_file = arguments["wave_file"]
                tools.play_wav(wave_file)
                print(f"play_wav success,wave_file : {wave_file}")
                return f"play_wav success,wave_file : {wave_file}"
            except Exception as e:
                print(f"An error occurred: {e}")
                return f"An error occurred: {e}"
        # 调用网络搜索
        if (function_name == "web_search"):
            try:
                query = arguments["query"]
                web_summary = web_search(query)
                print(f"web_search success,query : {query}")
                return web_summary
            except Exception as e:
                print(f"An error occurred: {e}")
                return f"An error occurred: {e}"
    else:
        return message["content"]

def web_search(query:str) -> str:
    """
    根据提供的搜索查询，使用bing搜索引擎进行网络检索，再使用chatgpt总结搜索结果。

    参数:
    query: 用户提供的搜索查询字符串。

    返回:
    web_summary:根据bing搜索结果,chatgpt生成的摘要。
    """
    web_response = apis.bing_search(query)
    web_summary_prompt = f"""
现在有以下的问题:{query}
针对该问题，进行了基于bing搜索的网络检索。下面会给出一段网络检索的结果，包括url和内容摘要。
1.阅读相关检索结果，整理缩进，编码等使其变得条理清晰。
2.将其中的url和摘要分点列出，尊重原内容，不要做修改。
3.基于检索结果，给出一些建设性评价性的意见。
文本如下：
{web_response}
"""
    # print(web_summary_prompt)
    # web_summary = apis.chatgpt(web_summary_prompt)
    # return web_summary
    return f"基于{query}在bing引擎上进行了搜索，结果为{web_response}"

if __name__ == "__main__":
    query = "搜索知乎，针对最新消息，有必要为了进国家电网考六级吗？"
    response = chatgpt_function(query)
    print(response)


