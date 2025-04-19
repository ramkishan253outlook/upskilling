from dotenv import load_dotenv
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.tools import render_text_description, Tool
from langchain_openai import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from typing import Union, List

from callbacks import AgentCallBackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_names(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
        raise ValueError(f"Toll with name {tool_name} not found")

if __name__ == "__main__":
    print("Hello ReAct Agent!!")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(temperature=0, callbacks=[AgentCallBackHandler()]).bind(stop=["Observation:"])
    intermediate_steps = []

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: x["agent_scratchpad"],
            }
            | prompt
            | llm
            | ReActSingleInputOutputParser()
        )
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "Give me the length of the word 'DOG'", "agent_scratchpad": intermediate_steps})
        print(agent_step)



        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_names(tools, tool_name)
            print(tool_to_use.name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))

