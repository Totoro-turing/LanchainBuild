from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
load_dotenv()


class LLMFactory:
    @staticmethod
    def create_llm() -> ChatOpenAI:
        return ChatOpenAI(
            temperature=0,
            model="deepseek-chat",
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url=os.getenv('DEEPSEEK_BASE_URL')
        )
