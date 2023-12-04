import openai
import json
import tools
import apis

class LucyAgent:
    """代表一个具有感知、大脑和行为能力的智能代理。"""
    def __init__(self, perception, brain, action):
        self.perception = perception
        self.brain = brain
        self.action = action

class Action:
    """代表智能代理的行为能力。当前未实现具体功能。"""
    def __init__(self):
        pass

    def chatgpt_function(self,prompt: str):
        """
        使用chatgpt的function call功能

        参数：
        prompt: 用户输入

        返回：
        chatgpt输出或function调用结果
        """
        functions = [
            {
                "name": "use_wiki",
                "description": f"检索wiki百科以补充不了解的知识",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "需要检索的事物",
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

            if (function_name == "use_wiki"):
                try:
                    query = arguments['query']
                    wiki_object = self.use_wiki(query)
                    print(f"use_wiki success,query : {query}")
                    return wiki_object
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return f"An error occurred: {e}"
        else:
            return message["content"]

    def use_wiki(self,search_query):
        """
        return {
            "url":page_url,
            "content":content
        }
        """
        wiki_object = tools.get_wikipedia_text(search_query)
        return wiki_object

    # def use_scraper(self,url):
    #     page_object = tools.scrape_webpage(url)
    #     return page_object
    #
    # def use_bing(self,query):
    #     bing_obejct = apis.bing_search(query)
    #     return bing_obejct

