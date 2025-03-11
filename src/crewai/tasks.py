from crewai import Task
from agents import writer_agent

def writer_task(single_keyword, prompts, llm):
    return Task(
        description=prompts["task"]["description"].format(keyword=single_keyword),
        expected_output=prompts["task"]["expected_output"],
        agent=writer_agent(prompts, llm),
    )