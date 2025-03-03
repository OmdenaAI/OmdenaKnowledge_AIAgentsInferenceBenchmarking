from decimal import Decimal, getcontext
import logging
import re

def extract_number(response: str, logger: logging.Logger) -> Decimal:
    """
    Extract numerical value from response with proper error handling
    
    Args:
        response: Response from LLM to parse
        
    Returns:
        Decimal: Extracted number or penalty value (1e6) if parsing fails
    """
    # Set high precision for scientific values
    getcontext().prec = 40
    PENALTY_VALUE = Decimal('1e6')
    value = PENALTY_VALUE

    if response:
        response = response.strip()
    
        try:
            # First try: direct conversion of scientific notation or regular number
            # Remove commas and spaces
            cleaned = response.replace(',', '').replace(' ', '')
            # Match scientific notation or regular numbers
            sci_match = re.search(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?', cleaned)
            if sci_match:
                value = Decimal(sci_match.group())
        
        except (ValueError, IndexError):
            logger.info(f"Could not extract number from response: {response}")
            
            try:
                # Second try: find any numbers including scientific notation
                numbers = re.findall(
                    r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?', 
                    cleaned
                )
                if numbers:
                    value = Decimal(numbers[-1])
                    logger.info(f"Extracted number with fallback from response: {response}")
                else:
                    logger.warning(f"No numbers found in response: {response}")
                    
            except Exception as e:
                logger.error(
                    f"Could not extract number with fallback from response: '{response}'. Error: {str(e)}"
                )
    
    return value