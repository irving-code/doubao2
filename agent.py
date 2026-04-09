from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# 1. 加载 .env 文件里的配置
load_dotenv()

# 2. 初始化大模型（完全适配火山引擎）
llm = ChatOpenAI(
    model=os.getenv("ENDPOINT_ID"),
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_URL"),
    temperature=0.3,
    timeout=60,
)

# 3. 最简单的测试：问一句你好
try:
    print("正在连接火山引擎...")
    response = llm.invoke("你好，如果你收到了，请回复'连接成功！'")
    print("\n🎉 大模型返回结果：")
    print(response.content)
except Exception as e:
    print("\n❌ 连接失败，错误信息：")
    print(e)