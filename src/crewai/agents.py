from crewai import Agent

def writer_agent(prompts, llm):
    return Agent(
        role=prompts["agent"]["role"],
        goal=prompts["agent"]["goal"],
        backstory=prompts["agent"]["backstory"],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
