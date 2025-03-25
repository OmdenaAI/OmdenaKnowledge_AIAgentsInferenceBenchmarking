import pandas as pd

def save_results_to_csv(results, total_keywords, total_time_taken, throughput, total_tokens, csv_save, framework):
    # Select the correct filename based on the framework
    filename_key = f"{framework}_filename"  # Example: "crewai_filename"
    filename = csv_save.get(filename_key, "default_results.csv")  # Fallback if key not found

    df = pd.DataFrame(results, columns=csv_save["columns"])
    
    # Create a summary row
    summary_data = {
        csv_save["columns"][0]: csv_save["summary"]["keyword"],
        csv_save["columns"][1]: csv_save["summary"]["latency"].format(total_keywords=total_keywords),
        csv_save["columns"][2]: csv_save["summary"]["generated_paragraph"].format(total_time_taken=total_time_taken),
        csv_save["columns"][3]: csv_save["summary"]["response_ratings"].format(throughput=throughput),
        csv_save["columns"][4]: csv_save["summary"]["tokens_used"].format(total_tokens=total_tokens),
        csv_save["columns"][5]: csv_save["summary"]["peak_memory"],
        csv_save["columns"][6]: csv_save["summary"]["memory_delta"],
    }
    
    # Append summary row **after results**
    df = pd.concat([df, pd.DataFrame([summary_data])], ignore_index=True)

    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


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