# chatgpt_python

### beebom_ori.py 
from the blog 『How to Train an AI Chatbot With Custom Knowledge Base Using ChatGPT API(https://beebom.com/how-train-ai-chatbot-custom-knowledge-base-chatgpt-api/)』
Tell you how to import your text data into gpt_index and predict by openAI API to create a LLM index that can be query

### beebom_modify.py
Personnel Change
1.use gpt-3.5-turbo.
2.Change from loading all text files in a folder to a single csv file.
3.Add wait one second when call openAI API for 60 times/min limit.
4.Don't always create an index

### chatgpt.py
funtion to connect openAI chat API with message history.

### app.py
Make the program perfect by adding the following functions
1.What to do if prompt is Chinese.
2.Translate to Chinese if the query result not Chinese.
3.Call gpt-index when determining the inclusion of the patent word; otherwise call chatGPT API
4.Add Message history

### docs 
The demo patent data.
