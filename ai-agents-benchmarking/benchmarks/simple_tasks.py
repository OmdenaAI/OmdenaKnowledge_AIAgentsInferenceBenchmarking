# benchmarks/simple_tasks.py

def qa_task():
    """Simple Q&A Task"""
    return {
        "prompt": "What is the capital of France?",
        "expected_answer": "Paris",
        "task_type": "qa"
    }

def text_summarization_task():
    """Basic text summarization"""
    return {
        "prompt": "Summarize the following text: Artificial intelligence is transforming industries by automating tasks, improving decision-making, and enhancing user experiences.",
        "task_type": "summarization"
    }

def sentiment_analysis_task():
    """Simple sentiment analysis task"""
    return {
        "prompt": "Analyze the sentiment of the following review: 'This product is amazing! I love it so much.'",
        "task_type": "sentiment"
    }

def basic_math_task():
    """Basic mathematical computation"""
    return {
        "prompt": "What is 12 + 45?",
        "task_type": "math"
    }

def named_entity_recognition_task():
    """Basic Named Entity Recognition (NER)"""
    return {
        "prompt": "Identify named entities in this text: 'Elon Musk is the CEO of Tesla and SpaceX.'",
        "task_type": "ner"
    }

# Dictionary mapping task names to functions
SIMPLE_TASKS = {
    "qa": qa_task,
    "summarization": text_summarization_task,
    "sentiment": sentiment_analysis_task,
    "math": basic_math_task,
    "ner": named_entity_recognition_task,
}

if __name__ == "__main__":
    for task_name, task_func in SIMPLE_TASKS.items():
        task_details = task_func()
        print(f"Task: {task_name}")
        print(f"Details: {task_details}\n")
