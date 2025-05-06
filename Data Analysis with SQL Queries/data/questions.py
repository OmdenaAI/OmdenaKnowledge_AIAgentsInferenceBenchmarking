# Dictionary with the question and the respective query, result and expected answer
queries = {
    "question_1": {
        "question": "What is the average rating, rounded to two decimal places, of the products in the category 'Electronics'?", 
        "write_query": {"query": "SELECT ROUND(AVG(rating), 2) AS average_rating FROM amazon_cleaned WHERE category = 'Electronics';"}, 
        "execute_query": {"result": "(4.08,)"}, 
        "generate_answer": {"answer": "The average rating, rounded to two decimal places, of the products in the category 'Electronics' is 4.08"}
    },
    "question_2": {
        "question": "How many unique product IDs are there in the table?", 
        "write_query": {"query": "SELECT COUNT(DISTINCT product_id) FROM amazon_cleaned"}, 
        "execute_query": {"result": "(1351,)"}, 
        "generate_answer": {"answer": "There are 1351 records in the database."}
    },
    "question_3": {
        "question": "Order the products by product_id and return the rating of the first product?", 
        "write_query": {"query": "SELECT rating FROM amazon_cleaned ORDER BY product_id LIMIT 1"}, 
        "execute_query": {"result": "[(4.1,)]"}, 
        "generate_answer": {"answer": "The rating of the first product in the table is 4.1."}
    },
    "question_4": {
        "question": "How many unique values for 'category' are in the database?", 
        "write_query": {"query": "SELECT COUNT(DISTINCT category) FROM amazon_cleaned"}, 
        "execute_query": {"result": "[(9,)]"}, 
        "generate_answer": {"answer": "There are 9 unique values for 'category' in the database."}
    },
    "question_5": {
        "question": "What is the maximum value in the column 'discount_percentage' in the table?", 
        "write_query": {"query": "SELECT MAX(discount_percentage) FROM amazon_cleaned"}, 
        "execute_query": {"result": "[('94%',)]"}, 
        "generate_answer": {"answer": "The maximum value in the column 'discount_percentage' in the table is 94%."}
    },
}