def generate_paragraph(state, llm, config):
    keyword = state["keyword"]

    # Extract prompt details from config
    role = config["prompts"]["agent"]["role"]
    goal = config["prompts"]["agent"]["goal"]
    backstory = config["prompts"]["agent"]["backstory"]
    task_description = config["prompts"]["task"]["description"].format(keyword=keyword)
    expected_output = config["prompts"]["task"]["expected_output"]

    # Construct prompt
    prompt = f"""
    Role: {role}
    Goal: {goal}
    Backstory: {backstory}

    Task Description: {task_description}

    Expected Output:
    {expected_output}
    """

    response = llm.invoke(prompt)
    return {"keyword": keyword, "response": response.content}
