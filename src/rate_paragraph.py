def rate_paragraph(paragraph: str, client, prompts, llm_config) -> str:
    """Sends a paragraph to Groq LLM and gets a rating from 1 to 10."""
    messages = [
        {"role": "system", "content": prompts["paragraph_rating"]["system"]},
        {"role": "user", "content": prompts["paragraph_rating"]["user"].format(paragraph=paragraph)},
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages, 
            model=llm_config["rating_model"],
        )
        rating = chat_completion.choices[0].message.content.strip()

        return rating if rating.isdigit() else "Invalid rating received"

    except Exception as e:
        return f"Error: {e}"