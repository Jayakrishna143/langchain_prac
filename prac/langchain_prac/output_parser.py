from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate
llm = ChatOllama(model="gemma3:1b", temperature=0)

# cpt = ChatPromptTemplate.from_messages ([HumanMessagePromptTemplate.from_template("Tell me tha pros of {topic}")])
# final_prompt = cpt.format(topic ="blackhole")
# res = llm.invoke(final_prompt)
# parser = CommaSeparatedListOutputParser()
# parsed_output = parser.parse(res.content)
# print(parsed_output)
parser = CommaSeparatedListOutputParser()
cpt = ChatPromptTemplate.from_messages ([SystemMessagePromptTemplate.from_template("You are a good assistan who will try to give the output in the below format and do not add explanation or extra text:{format} "),
                                        HumanMessagePromptTemplate.from_template("Tell me tha pros of {topic}")])
final_prompt = cpt.format(topic ="blackhole",format = parser.get_format_instructions())
response = llm.invoke(final_prompt)
parsed_output = parser.parse(response.content)
print(parsed_output)
