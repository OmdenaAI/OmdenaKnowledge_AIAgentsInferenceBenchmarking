import autogen

def writer_agent(config_list, prompts):
    """Creates an AssistantAgent with dynamic role and prompts from config."""
    return autogen.AssistantAgent(
        name=prompts["agent"]["role"],  # Set name dynamically
        system_message=(          
            f"Goal: {prompts['agent']['goal']}\n"
            f"Backstory: {prompts['agent']['backstory']}"
        ),
        llm_config={"config_list": config_list},
    )