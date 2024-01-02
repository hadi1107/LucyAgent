# 🤖 LucyAgent：基于LLM的智能代理设计实践

## 🎉 基本介绍

## 🌟 主要特性
- 🧠 基于Perception， Brain， Action的类框架
- 📚 简单的记忆流与知识库，原生JSON实现
- ✍️ 基于想法的CoT风格化对话，多要素的prompt设计
- 🔧 工具库（例如Wiki检索，处理PDF📝）
- 🎮 使用LLM驱动FSM（例如Agent心情，行为欲望）
- 🎨 可交互的Gradio Web Demo
- 👩‍🔧 细节优化（例如平滑的文本分割，多级RAG子知识结构）

## 🌊不需要的框架
- 🚫 不依赖LangChain
- 🚫 不依赖向量数据库
- 🚫 模型调用基于API调用实现，无本地部署

## 💬 Gradio Demo一览

![示例图片](./markdown/talk.png)

## 🚕上手

检查apis.py文件，选择对应服务，设置相关api-key环境变量。设置完成后：
```
python web_demo.py
```
由于项目结构简单，需要更改角色配置时，查看resource文件夹，对hutao.json和其他文件进行修改。
亦可使用web_demo.py在web界面上改变角色配置。

## 🛴提交代码


## 🚄Todo

- 重构apis.py：应该把多个服务集中在一个接口
