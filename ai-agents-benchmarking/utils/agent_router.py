from utils.config_loader import config

def route_to_agent(prompt):
    """Routes the prompt to the selected AI agent framework."""
    agent_framework = config["llm_execution"].get("agent_framework", "None")

    if agent_framework == "CrewAI":
        from frameworks.crewai_benchmark import execute_task_with_crewai
        return execute_task_with_crewai(lambda: prompt)
    elif agent_framework == "LangChain":
        from frameworks.langchain_benchmark import execute_task_with_langchain
        return execute_task_with_langchain(lambda: prompt)
    elif agent_framework == "LangGraph":
        from frameworks.langgraph_benchmark import execute_task_with_langgraph
        return execute_task_with_langgraph(lambda: prompt)
    elif agent_framework == "Swarm":
        from frameworks.swarm_benchmark import execute_task_with_swarm
        return execute_task_with_swarm(lambda: prompt)
    elif agent_framework == "AutoGen":
        from frameworks.autogen_benchmark import execute_task_with_autogen
        return execute_task_with_autogen(lambda: prompt)
    
    raise ValueError(f"Unsupported agent framework: {agent_framework}")
