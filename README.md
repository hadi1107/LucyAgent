# LucyAgent - LLM-based agent类

LucyAgent 是一个模拟具有感知、思考和行动能力的智能代理的类库。该库提供了一个智能代理的基本框架，包括感知(Perception)、大脑(Brain)和行为(Action)三个主要组成部分。其中的一些函数接口基于SimpleAPI，simpleAPI是对一些LLM & TTS模型，网络检索工具的简单封装。

## 功能

### 感知(Perception)

代表智能代理的感知能力。当前未实现具体功能，可根据需要进行扩展。

### 大脑(Brain)

负责记忆和知识处理，包括创建、添加、删除和搜索记忆和知识。

### 行为(Action)

代表智能代理的行为能力。当前未实现具体功能，可根据需要进行扩展。

# SimpleAPI - 轻度封装的agent开发依赖库

SimpleAPI是一个Python库，旨在简化与各种OpenAI模型和网络服务的集成，如大型语言模型（LLMs）、文本转语音（TTS）、网络搜索工具等。该库提供了一系列函数，简化了与这些服务交互的过程，使得将AI能力嵌入到您的应用程序中变得更加容易。

## 函数接口

- **chatgpt**: 利用OpenAI的GPT-3.5-turbo模型进行对话式响应。
- **instructgpt**: 使用GPT-3.5-turbo-instruct模型进行基于指令的查询。
- **embedding**: 使用OpenAI的text-embedding-ada-002模型为单词或短语生成嵌入向量。
- **genshin_tts**: 访问原神TTS API以生成不同声音的语音。
- **bing_search**: 使用必应搜索API执行网络搜索。
- **play_wav**: 本地播放`.wav`音频文件。
- **scrape_webpage**: 提取网页的标题和文本。
- **get_wikipedia_text**: 检索并总结维基百科的内容。
- **chatgpt_function**: 利用ChatGPT的函数调用功能执行上述功能的集成。
