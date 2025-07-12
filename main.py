from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from langchain.schema import BaseOutputParser, StrOutputParser
class SelfParser(BaseOutputParser):
    def parse(self, text):
        return text.split(".")
load_dotenv()

from langchain.prompts import PromptTemplate
deepseek= ChatOpenAI(
    temperature=0,
    model="deepseek-chat",
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL')
)
prompt = PromptTemplate.from_template("{city}最著名的景点有哪些，举三个？不用具体介绍")
 
chain = prompt | deepseek | StrOutputParser()
print(chain.invoke({"city": "青海"}))