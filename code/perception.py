import json
import PyPDF2
import apis

class Perception:
    """
    Perception 类，该类负责处理输入的文本数据，包括从 PDF 文件中读取文本、
    将文本分割成多个段落以确保连贯性和上下文的理解，并且可以生成知识列表以供智能代理的
    “大脑”处理。

    主要功能包括：
    - 读取 PDF 文件并提取文本内容。
    - 将长文本分割成带有上下文的多个段落。
    - 从文本段落中提取知识点，并生成知识列表。
    - 将知识列表保存为 JSON 文件。

    脚本使用了 PyPDF2 库来处理 PDF 文件，以及 apis 模块（自定义的）来请求文本的嵌入表示。
    这个脚本是一个命令行工具，没有图形用户界面。
    """
    def __init__(self):
        pass

    @classmethod
    def read_pdf(cls, pdf_path):
        """
        读取PDF文件并返回其文本内容。

        参数:
        pdf_path: PDF文档的文件路径。

        返回:
        一个包含PDF文档所有页面文本的字符串。
        """
        text_content = []

        try:
            # 以二进制读取模式打开 PDF 文件
            with open(pdf_path, 'rb') as file:
                # 创建 PDF 阅读器对象
                pdf_reader = PyPDF2.PdfFileReader(file)

                # 遍历 PDF 中的每一页
                for page_num in range(pdf_reader.numPages):
                    # 获取页面对象
                    page = pdf_reader.getPage(page_num)

                    # 提取页面文本
                    text = page.extractText()

                    # 将页面文本添加到列表中
                    text_content.append(text)

        except FileNotFoundError:
            print(f"文件未找到：{pdf_path}")
            return None
        except PyPDF2.utils.PdfReadError as e:
            print(f"读取 PDF 文件时发生错误：{e}")
            return None

        # 将所有页面的文本连接成一个字符串并返回
        return '\n'.join(text_content)

    @classmethod
    def split_text(cls, text, min_length=500, buffer_min_length=150):
        """
        将文本分割成带有上下文缓冲区的多个段落，确保连贯性。

        参数:
        text: 需要分割的输入文本。
        min_length: 每个文本段落的最小长度。
        buffer_min_length: 上下文缓冲区的最小长度。

        返回:
        一个文本段落的列表。
        """
        if len(text) < min_length:
            return [text]

        segments = []
        current_segment = ''
        buffer_sentences = ''
        cleared_buffer = ''

        # 将文本按句子分割
        sentences = text.split('。')

        for i, sentence in enumerate(sentences):
            # 检查是否是最后一句，如果不是，则添加句号
            if i < len(sentences) - 1:
                sentence += '。'

            # 将当前句子添加到缓冲区
            buffer_sentences += sentence

            # 检查缓冲区句子长度是否满足最小长度要求
            if len(buffer_sentences) >= buffer_min_length:
                # 将缓冲区句子添加到当前段落
                current_segment += buffer_sentences
                # 清空缓冲区句子
                cleared_buffer = buffer_sentences
                buffer_sentences = ''

            # 如果当前段落达到指定长度，或者已经是最后一句话了，则结束当前段落
            if len(current_segment) >= min_length or (i == len(sentences) - 1):
                if (i == len(sentences) - 1) and buffer_sentences:
                    current_segment += buffer_sentences
                segments.append(current_segment.replace(' ', '').replace('\n', ''))
                # 如果已经是最后一句话了，就不需要再设置 `cleared_buffer` 了
                if i < len(sentences) - 1:
                    current_segment = cleared_buffer
                else:
                    current_segment = ''

        if current_segment:
            segments.append(current_segment.replace(' ', '').replace('\n', ''))

        return segments

    @classmethod
    def generate_knowledge_units(cls, segments):
        """
        从文本段落中生成知识单元，并请求每个段落的嵌入表示，以供智能代理的大脑导入和处理。

        参数:
        segments: 文本段落的列表。

        返回:
        包含所有文本段落的知识单元列表。每个知识单元包含文本和对应的嵌入表示。
        """
        try:
            # 请求每个段落的嵌入表示
            embeddings = apis.request_embedding(segments)
        except Exception as e:
            print(f"请求嵌入表示时发生错误：{e}")
            return []

        knowledge_units = []
        for segment, embedding in zip(segments, embeddings):
            if embedding:  # 确保嵌入表示有效
                knowledge_unit = {
                    "text": segment,
                    "embedding": embedding,
                    "sub_knowledge": None  # 预留字段，用于未来可能的扩展
                }
                knowledge_units.append(knowledge_unit)

        # 尝试保存知识单元列表到 JSON 文件
        try:
            with open("../resource/knowledge_units.json", "w", encoding="utf-8") as f:
                json.dump(knowledge_units, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"保存知识单元列表时发生错误：{e}")

        return knowledge_units

if __name__ == "__main__":
    with open("../resource/xiaogong.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print(f"文本总长度:{len(text)}")
    segments = Perception.split_text(text,500,0)
    for segment in segments:
        print(f"文本段长度：{len(segment)}\n{segment}\n")

    segments = Perception.split_text(text,500,100)
    for segment in segments:
        print(f"文本段长度：{len(segment)}\n{segment}\n")