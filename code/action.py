import openai
import json
import tools
import apis

class Action:
    """
    Action 类封装了与 OpenAI GPT-3 模型进行交互的方法，以及其他外部工具和API的调用。

    方法:
    - chatgpt_function(prompt): 使用chatgpt的function call功能来处理用户输入。
    - use_wiki(search_query): 使用维基百科API检索信息。
    - use_scraper(url): 使用网络爬虫工具抓取网页内容。
    - use_bing(query): 使用Bing搜索API进行网络搜索。

    该类的目的是提供一个接口来调用不同的功能和API，以便于集成到更大的系统中。
    """

    def __init__(self):
        pass

    def chatgpt_function(self, prompt: str):
        """
        使用chatgpt的function call功能。

        参数:
        - prompt: 用户输入的提示信息。

        返回:
        - chatgpt输出或function调用结果。
        """
        functions = [
            {
                "name": "use_wiki",
                "description": "检索wiki百科以补充不了解的知识",
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

        if message.get("function_call"):
            # 解析函数名和函数参数
            function_name = message["function_call"]["name"]
            arguments = json.loads(message["function_call"]["arguments"])

            if function_name == "use_wiki":
                try:
                    query = arguments['query']
                    wiki_object = self.use_wiki(query)
                    print(f"use_wiki success, query: {query}")
                    return wiki_object
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return f"An error occurred: {e}"
        else:
            return message["content"]

    def use_wiki(self, search_query):
        """
        使用维基百科API检索信息。

        参数:
        - search_query: 需要检索的查询字符串。

        返回:
        - 包含URL和内容的字典。
        """
        wiki_object = tools.get_wikipedia_text(search_query)
        return wiki_object

    def use_scraper(self, url):
        """
        使用网络爬虫工具抓取网页内容。

        参数:
        - url: 需要抓取的网页URL。

        返回:
        - 抓取的网页内容。
        """
        page_object = tools.scrape_webpage(url)
        return page_object

    def use_bing(self, query):
        """
        使用Bing搜索API进行网络搜索。

        参数:
        - query: 搜索查询字符串。

        返回:
        - 搜索结果对象。
        """
        bing_object = apis.bing_search(query)
        return bing_object