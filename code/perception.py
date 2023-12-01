import PyPDF2

class LucyAgent:
    """代表一个具有感知、大脑和行为能力的智能代理。"""
    def __init__(self, perception, brain, action):
        self.perception = perception
        self.brain = brain
        self.action = action

class Perception:
    """代表智能代理的感知能力。当前未实现具体功能。"""
    def __init__(self):
        pass

    def read_pdf(self,pdf_path):
        """
        读取 PDF 文件并返回其文本内容。

        参数:
        pdf_path: PDF 文件的路径。

        返回:
        一个包含 PDF 文件所有页面文本的字符串。
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

    def split_text_by_min_length(self, text, min_length=500, buffer_min_length=150):
        # 初始化变量
        segments = []
        current_segment = ''
        buffer_sentences = ''
        cleard_buffer = ''

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
                cleard_buffer = buffer_sentences
                buffer_sentences = ''

            # 如果当前段落达到指定长度，则结束当前段落
            if len(current_segment) >= min_length:
                # 将当前段落添加到段落列表
                segments.append(current_segment.strip())
                # 重置当前段落为最后一句
                current_segment = cleard_buffer

        # 确保最后一个段落被添加，即使它没有达到最小长度
        if current_segment:
            segments.append(current_segment.strip())

        return segments

if __name__ == "__main__":
    perception = Perception()
    pdf_text = perception.read_pdf('../resource/zzz.pdf')
    segments = perception.split_text_by_min_length(pdf_text)

    # 打印分割后的段落
    for segment in segments:
        print(segment)
        print("------")



