from google.cloud import translate_v2 as translate
from gpt_index import Ｄocument, SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, PromptHelper
from gpt_index import LLMPredictor as BaseLLMPredictor
from langchain import OpenAI
import gradio as gr
import sys
import os
import pandas as pd
import time
import openai
from transformers import MarianMTModel, MarianTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import unicodedata

os.environ["OPENAI_API_KEY"] = 'Your API Key'
openai.api_key = "Your API Key"

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

# # 初始化翻译模型和 tokenizer
# model_name = 'Helsinki-NLP/opus-mt-en-zh'
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# model = MarianMTModel.from_pretrained(model_name)


# # 定义一个函数将文本翻译成中文
# def translate(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True)
#     translated = model.generate(**inputs)
#     translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
#     return translated_text[0]

# # 定义一个函数将文档向量化
# def vectorize_text(text):
#     # 对文本进行翻译
#     translated_text = translate(text)

#     # 使用预训练的词向量模型将文本向量化
#     # 这里使用的是 FastText 多语言词向量模型
#     # 如果你使用的是其他的模型，请相应地调整代码
#     # 可以通过 Hugging Face 的 transformers 库来加载其他模型
#     vector_model_path = 'cc.en.300.bin'
#     if 'zh' in translated_text:
#         vector_model_path = 'cc.zh.300.bin'
#     # 加载词向量模型
#     vector_model = FastText.load(vector_model_path)
#     # 向量化文本
#     vector = np.mean([vector_model[word] for word in translated_text.split()], axis=0)
#     return vector


def is_chinese(string):
    string = string.replace("\n", "")
    for char in string:
        if 'CJK' in unicodedata.name(char):
            return True
    return False


def translate_text(text, target_language):
    translate_client = translate.Client()
    print("Translation: ", text, target_language)
    result = translate_client.translate(
        text, target_language, source_language='en')
    print("Translation result: ", result["translatedText"])
    return result["translatedText"]


def parse_excel(path):
    # read excel file as dataframe
    df = pd.read_excel(path)
    # convert dataframe to dictionary
    data = df.to_dict(orient="records")
    # create document object with file name as id and data as content
    return Document(id=path.name, content=data)


class CSVFileReader(SimpleDirectoryReader):
    def __init__(self, file_path, delimiter=',', encoding='utf-8'):
        self.file_path = file_path
        self.delimiter = delimiter
        self.encoding = encoding

    def load_data(self):
        data = pd.read_csv(
            self.file_path, delimiter=self.delimiter, encoding=self.encoding)
        documents = [Document(doc_id=str(i), text=str(record))
                     for i, record in enumerate(data.to_dict(orient='records'))]
        return documents


class LLMPredictor(BaseLLMPredictor):
    def __init__(self, llm, prompt_helper=None):
        super().__init__(llm)
        self.prompt_helper = prompt_helper

    def __call__(self, prompt):
        if self.prompt_helper:
            prompt = self.prompt_helper(prompt)

        time.sleep(1.5)  # Sleep for 1 second before calling OpenAI API

        response = self.llm(prompt)
        return response


def construct_index(csv_file_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(
        temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = CSVFileReader(csv_file_path).load_data()

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index


def contains_keywords(text, keywords):
    return any(keyword in text for keyword in keywords)


def ai_chat(input_text):
    # 關鍵字列表
    keywords = ['patent', '專利', '专利', 'สิทธิบัตร', 'bằng sáng chế']
    target_language = "zh-TW"
    if not is_chinese(input_text):
        input_text = translate_text(input_text, target_language)

    # 檢查字串是否包含列表中的任何元素
    result = contains_keywords(input_text.lower(), keywords)

    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, mode="embedding",
                           response_mode="compact")
    print("response:", response)

    # Unfortunately, without prior knowledge, it is not possible to answer this question
    # No answer can be provided without prior knowledge.
    if result and not ("without prior knowledge" in response.response or "Unfortunately" in response.response):
        # index = GPTSimpleVectorIndex.load_from_disk('index.json')
        # response = index.query(input_text, response_mode="compact")
        messages.append({"role": "assistant", "content": response.response})

        # 目标语言的代码（例如：en-英语）
        target_language = "zh-TW"
        print(response.response)

        # 调用翻译函数
        if not is_chinese(response.response):
            return translate_text(response.response, target_language)
        else:
            return response.response

    else:
        try:
            messages.append({"role": "user", "content": input_text})
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
            reply = chat.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})
            return reply
        except openai.api_errors.APIError as api_error:
            return f"API 錯誤: {api_error}"

        except Exception as error:
            return f"其他錯誤: {error}"


def chatbot(input, history=[]):
    output = ai_chat(input)
    history.append((input, output))
    return history, history


iface = gr.Interface(fn=chatbot,
                     inputs=["text", 'state'],
                     outputs=["chatbot", 'state'],
                     title="Custom-trained AI Chatbot")

# 檔案存在就不用重建
if not os.path.exists("index.json"):
    index = construct_index("docs/專利一般查詢2023-03-16.csv")
iface.launch(share=True)
