YIELD_SYSTEM_PROMPT = """You are an agricultural yield prediction expert. Return ONLY the predicted yield as a number, no other text.

"""

YIELD_PREDICTION_PROMPT = """
Predict yield for {crop} based on these conditions:

### Current Measurements:
- Precipitation: {precipitation} mm/day
- Specific Humidity: {specific_humidity} g/kg
- Relative Humidity: {relative_humidity}%
- Temperature: {temperature}°C

### {historical_records_type}:
{historical_examples}

### Yield Statistics:
Utilize the following yield stats from all {crop} records for your prediction:
{yield_stats}.
- Your prediction must fall strictly within this range.
- Use the mean ({mean_yield:.2f}) as a central reference point.

### Instructions:
Base your prediction on the {similarity_type} historical records provided, ensuring it is constrained by the yield stats range. Your output must be a single number representing the predicted yield, with no additional text.
"""
