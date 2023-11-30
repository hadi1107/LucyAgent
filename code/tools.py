"""
tools.py:定义一些工具性的agent可用函数
"""
import json
import requests
import simpleaudio as sa
from bs4 import BeautifulSoup

def play_wav(wave_file:str) -> None:
    """
    使用simpleaudio播放wav音频

    参数：
    wave_file:音频地址
    """
    wave_obj = sa.WaveObject.from_wave_file(wave_file)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def scrape_webpage(url: str) -> dict:
    """
    爬取指定URL的网页，提取并返回网页的标题和所有文本内容。

    参数:
    url: 网页URL

    返回:
    data: 包含网页标题和所有文本内容的字典
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # 设置正确的编码
        response.encoding = response.apparent_encoding

        # 解析HTML内容
        soup = BeautifulSoup(response.text, 'html.parser')

        # 提取标题
        title = soup.title.string if soup.title else "No title found"

        # 提取整个页面的文本内容，忽略脚本和样式
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()  # 从文档树中移除这些元素及其内容

        text = soup.get_text(separator='\n', strip=True)

        return {
            'title': title,
            'text': text,
        }

    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")

def get_wikipedia_text(search_query:str, language='zh')->dict:
    """
    搜索Wikipedia并获取与查询最相关的页面的完整内容。

    参数:
    search_query:搜索查询词。
    language:Wikipedia的语言版本，默认为中文。

    返回:
    wiki_object:包含url和content的字典
    - url:wiki百科地址
    - content:最相关页面完整内容的文本。
    """
    # MediaWiki API的URL
    SEARCH_API_URL = f"https://{language}.wikipedia.org/w/api.php"

    # 设置搜索请求参数
    search_params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': search_query,
        'srlimit': 1,  # 只获取最相关的一个结果
    }

    try:
        # 发送搜索请求
        search_response = requests.get(SEARCH_API_URL, params=search_params)
        search_response.raise_for_status()

        # 解析搜索响应数据
        search_data = search_response.json()
        search_results = search_data.get('query', {}).get('search', [])

        # 检查是否有搜索结果
        if not search_results:
            return 'No search results found.'

        # 获取最相关页面的标题
        relevant_page_title = search_results[0]['title']
        BASE_PAGE_URL = 'https://zh.wikipedia.org/wiki/'
        page_url = BASE_PAGE_URL + relevant_page_title.replace(' ', '_')

        # 设置获取页面内容的请求参数
        content_params = {
            "action": "query",
            "format": "json",
            "titles": relevant_page_title,
            "prop": "extracts",
            "explaintext": True,
        }

        # 发送获取页面内容的请求
        content_response = requests.get(SEARCH_API_URL, params=content_params)
        content_response.raise_for_status()

        # 解析页面内容响应数据
        content_data = content_response.json()

        with open("../resource/wiki_temp.json","w",encoding="utf-8") as f:
            json.dump(content_data, f, ensure_ascii=False, indent=4)

        pages = content_data['query']['pages']
        page_id = next(iter(pages))
        content = pages[page_id].get('extract', 'No content available.')

        return {
            "url":page_url,
            "content":content
        }

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as err:
        return f"An unexpected error occurred: {err}"

# 使用示例
if __name__ == "__main__":
    url = "https://zhuanlan.zhihu.com/p/664880302"
    web_page_object = scrape_webpage(url)
    print( web_page_object["text"])

    page_title = "腾讯光子"
    wiki_obejct = get_wikipedia_text(page_title)
    print(wiki_obejct["url"])
    print(wiki_obejct["content"])