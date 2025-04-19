import os
from dotenv import load_dotenv

from tools.tools import get_profile_url_tavily


load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub

def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    template = "Given the full name {name_of_person} find their LinkedIn profile URL. Your answer should contain only the URL."

    prompt_template = PromptTemplate(template=template, input_variables=["name_of_person"])

    tools_for_agent = [
        Tool(
            name="Crawl google 4 linkedin page",
            func=get_profile_url_tavily,
            description="useful for when you need to find a LinkedIn profile URL for a person. The input should be the full name of the person."
        )
    ]

    react_prompt = hub.pull("hwchase17/react")  # Load the react prompt from LangChain Hub

    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)
    
    result = agent_executor.invoke(input={"input": prompt_template.format_prompt(name_of_person=name)})

    linked_in_profile_url = result["output"]
    return linked_in_profile_url

if __name__ == "__main__":
    likedin_url=lookup(name="Eden Marco Udemy Instructor")
    print(likedin_url)



