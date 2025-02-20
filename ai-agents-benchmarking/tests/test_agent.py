# from litellm import completion

# response = completion(
#     model="ollama/deepseek-r1", 
#     messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
#     api_base="http://localhost:11434"
# )
# print(response)
# ================
# Example usage - Streaming
# from litellm import completion

# response = completion(
#     model="ollama/llama3", 
#     messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
#     api_base="http://localhost:11434",
#     stream=True
# )
# print(response)
# for chunk in response:
#     print(chunk['choices'][0]['delta'])
# =====