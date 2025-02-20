# benchmarks/complex_tasks.py

def multi_step_reasoning_task():
    """Complex multi-step reasoning task"""
    return {
        "prompt": "Solve the following problem step by step: A car travels at 60 mph for 2 hours, then increases speed to 80 mph for the next 3 hours. What is the total distance traveled?",
        "expected_answer": "360",  # 60*2 + 80*3 = 360
        "task_type": "multi_step_reasoning"
    }

def code_generation_task():
    """AI-generated code based on requirements"""
    return {
        "prompt": "Generate a Python function that calculates the factorial of a number using recursion.",
        "task_type": "code_generation"
    }

def advanced_text_summarization_task():
    """Summarization of long, complex text"""
    return {
        "prompt": "Summarize the following research abstract: Large language models have revolutionized NLP by demonstrating emergent capabilities in reasoning, translation, and generation. However, challenges such as bias, energy efficiency, and interpretability remain critical areas of improvement.",
        "task_type": "advanced_summarization"
    }

def logical_deduction_task():
    """Logical reasoning challenge"""
    return {
        "prompt": "If all cats are mammals and some mammals are nocturnal, does it follow that some cats are nocturnal? Explain.",
        "task_type": "logical_deduction"
    }

def api_data_extraction_task():
    """Simulate AI extracting relevant information from API responses"""
    return {
        "prompt": "Extract key financial insights from the following API response: { 'stock': 'AAPL', 'price': 145.67, 'change': '+1.2%', 'volume': 1000000 }",
        "expected_answer": "Stock: AAPL, Price: 145.67, Change: +1.2%, Volume: 1000000",
        "task_type": "api_extraction"
    }

# Dictionary mapping task names to functions
COMPLEX_TASKS = {
    "multi_step_reasoning": multi_step_reasoning_task,
    "code_generation": code_generation_task,
    "advanced_summarization": advanced_text_summarization_task,
    "logical_deduction": logical_deduction_task,
    "api_extraction": api_data_extraction_task,
}

if __name__ == "__main__":
    for task_name, task_func in COMPLEX_TASKS.items():
        task_details = task_func()
        print(f"Task: {task_name}")
        print(f"Details: {task_details}\n")
