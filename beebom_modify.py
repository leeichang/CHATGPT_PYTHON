from gpt_index import Document,SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import gradio as gr
import sys
import os

import time
import pandas as pd

# 移除！沒有這個
# from openai import OpenAI
#from gpt import LLMPredictor

os.environ["OPENAI_API_KEY"] = 'Your API Key'

#後續補上引用
import csv
#載入csv文件
# def load_csv_documents(directory_path):
#     documents = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".csv"):
#             with open(os.path.join(directory_path, filename), 'r') as f:
#                 reader = csv.reader(f)
#                 for row in reader:
#                     documents.append('\n'.join(row))
#     return documents
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

#建立LLMPredictor 休息`1`秒
def make_llm_prediction(prompt):
    response = llm.predict(prompt)
    time.sleep(1) # 暫停1秒
    return response


#建立索引
def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    
    #修改成Gpt3.5 因為後續測試遇到每分鐘只能呼叫60次的限制，所以每做一次呼叫就休息1秒
    # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    
    llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs)
    llm_predictor = LLMPredictor(llm=llm)
    llm_predictor.make_prediction = make_llm_prediction


    # 讀取csv文件轉換為documents格式
    documents = CSVFileReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

#建立聊天機器人
def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

#實例化聊天機器人畫面
iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

#建立資料夾索引，判斷是否有index.json檔案，沒有則建立
#改成直接傳入csv檔案
if not os.path.exists('index.json'):
    index = construct_index("docs/專利一般查詢2023-03-16.csv")

#啟動聊天機器人
iface.launch(share=True)
