import re

def calculate_accuracy(result, expected_answer):
    """
    Calculate accuracy for tasks with known answers.

    Args:
        result (str): The LLM's output.
        expected_answer (str): The correct answer to compare against.

    Returns:
        float: Accuracy score as a percentage.
    """
    # Sanitize result (strip special characters)
    result = re.sub(r'[^\w\s]', '', str(result)).lower().strip()
    expected_answer = str(expected_answer).lower().strip()

    # ✅ Numeric Match
    try:
        if result.replace('.', '', 1).isdigit() and expected_answer.replace('.', '', 1).isdigit():
            return 100.0 if abs(float(result) - float(expected_answer)) < 0.01 else 0.0
    except ValueError:
        pass

    # ✅ Exact or Partial Match
    if result == expected_answer:
        return 100.0
    elif expected_answer in result:
        return 75.0
    return 0.0
