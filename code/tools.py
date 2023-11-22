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

def scrape_webpage(url:str) -> (str,list):
    """
    爬取指定URL的网页，提取并返回网页的标题和段落文本。

    参数:
    url: 网页URL

    返回:
    title: 网页标题
    paragraphs: 网页段落文本列表
    """
    try:
        response = requests.get(url)
        response.encoding = "utf-8"

        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string
        all_text = soup.get_text()  # 获取网页中的所有文本

        return title, all_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

if __name__ == "__main__":
    title , text = scrape_webpage("https://zhuanlan.zhihu.com/p/657952237")
    print(text)