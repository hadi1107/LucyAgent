# 🤖 LucyAgent：基于LLM的智能代理设计实践

## 🌟 主要特性
- 🧠 Perception, Brain, Action框架
- 📚 记忆流
- 🏫 知识库
- 🔧 工具库（例如wiki检索）
- 🎮 LLM决策的有限状态机
- ✍️ 充分的prompt设计
- 📔 完整的logging
- 🎨 Gradio可视化
- 🚫 不依赖langchain

## 🧠 brain模块

### 📝 记忆模块设计

#### 🧩 记忆单元
- 记忆单元由其记忆描述、创建时间、描述的嵌入构成。
- 记忆流是记忆单元的简单列表。

```python
memory = {
    "description": "记忆描述",
    "create_time": "创建时间",
    "embedding": "嵌入向量",
}
```

#### 🌐 记忆系统设计
- 🆕 创建记忆 (`create_memory`)：根据输入和输出创建一个记忆摘要。
- ➕ 添加记忆 (`add_memory`)：将记忆添加到记忆流中。
- 📝 总结记忆 (`summarize_memory`)：总结相似记忆并形成新的记忆。
- ❌ 删除记忆 (`del_memory`)：按索引删除记忆或清空所有记忆。
- 👀 展示记忆 (`show_memory`)：列出所有记忆摘要和相关信息。
- 🔍 搜索记忆 (`search_memory`)：找到最匹配的记忆。

#### 📚 知识单元
- 知识单元由其知识文本和文本的嵌入构成。
- 知识库是知识单元的集合，便于高效检索。

```python
knowledge = {
    "text": "知识文本",
    "embedding": "嵌入向量",
}
```

#### 📖 知识系统设计
- 🧠 提取知识 (`extract_knowledge`)：从文本中提取知识点。
- ➕ 添加知识 (`add_knowledge`)：将知识添加到知识库中。
- ❌ 删除知识 (`del_knowledge`)：按索引删除知识或清空所有知识。
- 👀 展示知识 (`show_knowledge`)：列出所有知识文本和相关信息。
- 🔍 搜索知识 (`search_knowledge`)：找到最匹配的知识。

### 🎭 角色配置说明

- 📛 `name`: 角色的姓名。
- 🌱 `seed_memory`: 角色的背景故事。
- 💬 `language_style`: 角色的语言风格。
- 🎮 `current_state`: 角色目前的活动或状态。
- 🧠 `basic_knowledge`: 角色的基础知识，初始为空数组。
- 📚 `memory_stream`: 角色的记忆流，初始为空数组。
- 😄 `mood_list`: fsm的可能状态列表。
- 😃 `emoji_list`: fsm状态对应的表情包列表。

### 📌 JSON映射到NPC的属性
```json
{
  "name": "角色名称",
  "seed_memory": "角色背景故事描述",
  "language_style": "角色语言风格描述",
  "current_state": "角色当前状态描述",
  "basic_knowledge": [],
  "memory_stream": [],
  "mood_list": [],
  "emoji_list": []
}
```

