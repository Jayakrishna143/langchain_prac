from langchain.tools import tool
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import requests
load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # Or gemini-1.5-flash depending on your API access
    temperature=0,
    google_api_key=os.getenv("gemini")
)

@tool
def multiply(a:int,b:int) -> int:
    """Given 2 numbers a and b this tool returns their product"""
    return a*b

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """ 
    this function fetches the currency conversion factor between a given base currency and a target currency
    """
    url =f"https://v6.exchangerate-api.com/v6/846ff0b0966a9b0ed53d2c83/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()
result = get_conversion_factor.invoke({"base_currency":"USD","target_currency":"INR"})
print(result)

@tool
def convert(base_currency_value:float,conversion_rate:float)-> float:
    """
    given a currency conversion rate this function calculates the target cuurency value from a given base currency
    """
    return base_currency_value * conversion_rate
convert_res = convert.invoke({"base_currency_value":10.00,"conversion_rate":91.3})
print(convert_res)


# llm_with_tool = model.bind_tools([multiply])
# res = llm_with_tool.invoke("can you multiply 3 * 10 ")
# print(res.tool_calls[0])

# result =  multiply.invoke(res.tool_calls[0].get("args"))
# print(result)
