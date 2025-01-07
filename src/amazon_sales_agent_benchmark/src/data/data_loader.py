import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import json
import chromadb
from chromadb.config import Settings
from src.llm.base_llm_provider import BaseLLMProvider
import re  # Add to imports at top
import time

class DataLoader:
    """Handles data loading, validation, and ChromaDB operations."""

    def __init__(
        self, 
        collection_name: str,
        embedding_function,
        logger: logging.Logger,
        llm: BaseLLMProvider,
        config: Dict,
        persist_directory: Optional[str] = None
    ):
        """Initialize with all required dependencies."""
        self.llm = llm
        self.logger = logger
        self.config = config
        self.required_columns = config['required_columns']
        # Use model's max_length for chunk size
        self.chunk_size = config['model'].get('max_length', 2048)
        self.chunk_overlap = int(self.chunk_size * 0.1)  # 10% overlap

        # Setup ChromaDB with absolute path for persistence
        self.persist_directory = str(Path(persist_directory or ".chromadb").absolute())
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Using absolute persist directory: {self.persist_directory}")
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                allow_reset=True,
                is_persistent=True
            )
        )

        # Initialize collections with detailed logging
        product_collection_name = f"{collection_name}_products"
        metric_collection_name = f"{collection_name}_metrics"
        
        self.logger.info(f"Initializing collections with base name: {collection_name}")
        self.logger.info(f"Product collection name: {product_collection_name}")
        self.logger.info(f"Metric collection name: {metric_collection_name}")
        
        # List existing collections
        existing_collections = self.client.list_collections()
        self.logger.info(f"Existing collections: {[c.name for c in existing_collections]}")

        self.product_collection = self.client.get_or_create_collection(
            name=product_collection_name,
            embedding_function=embedding_function
        )
        self.metric_collection = self.client.get_or_create_collection(
            name=metric_collection_name,
            embedding_function=embedding_function
        )

        # Log collection sizes and details
        self.logger.info(f"Product collection name: {self.product_collection.name} Product collection size: {self.product_collection.count()}")
        self.logger.info(f"Metric collection name: {self.metric_collection.name} Metric collection size: {self.metric_collection.count()}")

        # Add categories attribute
        self.categories = set()

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate and clean the data, ensuring all required columns are properly formatted."""
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Amazon data missing required columns: {missing_columns}")

        # Validate data types and non-null values for critical columns
        for col in ['product_id', 'product_name', 'category']:
            if data[col].isnull().any():
                null_indices = data[data[col].isnull()].index.tolist()
                raise ValueError(f"Column '{col}' contains null values at indices: {null_indices}")

        # Validate and clean numeric columns
        numeric_cols = {
            'rating': lambda x: pd.to_numeric(x, errors='coerce').fillna(0.0),
            'rating_count': lambda x: pd.to_numeric(x.str.replace(',', '').str.strip(), errors='coerce').fillna(0).astype(int),
            'discounted_price': lambda x: pd.to_numeric(x.str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0.0),
            'actual_price': lambda x: pd.to_numeric(x.str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0.0),
            'discount_percentage': lambda x: pd.to_numeric(x.str.replace('%', '').str.strip(), errors='coerce').fillna(0.0)
        }

        for col, converter in numeric_cols.items():
            try:
                data[col] = converter(data[col])
            except Exception as e:
                self.logger.error(f"Error cleaning column '{col}': {str(e)}")
                raise ValueError(f"Column '{col}' contains invalid data.")

        # Final validation to check for remaining NaN or invalid entries
        for col in numeric_cols.keys():
            if data[col].isnull().any():
                null_indices = data[data[col].isnull()].index.tolist()
                raise ValueError(f"Column '{col}' still contains null values after cleaning at indices: {null_indices}")

    def needs_processing(self, file_path: Path) -> bool:
        """
        Determine if data needs to be reprocessed by comparing the source file's
        modification time with the last processed timestamp in the database metadata.
        Returns True if processing is needed, False otherwise.
        """
        needs_processing = True  # Default assumption: processing is needed

        try:
            # Log collection counts
            self.logger.info(f"Checking collections - Products: {self.product_collection.count()}, Metrics: {self.metric_collection.count()}")
            self.logger.info(f"Using persist directory: {self.persist_directory}")

            # Ensure collections are not empty
            if self.product_collection.count() > 0 and self.metric_collection.count() > 0:
                # Fetch metadata
                metadata = self.product_collection.get(
                    where={"document_type": "metadata"},
                    include=["metadatas"]
                )

                if metadata["ids"]:
                    # Retrieve last processed timestamp
                    last_processed = float(metadata["metadatas"][0].get("last_processed", 0))
                    data_modified = file_path.stat().st_mtime

                    # Log timestamps for debugging
                    self.logger.info(f"File modified time: {data_modified}, Last processed time: {last_processed}")

                    # Compare timestamps
                    if data_modified <= last_processed:
                        self.logger.info("No processing needed, database is up to date")
                        needs_processing = False
                    else:
                        self.logger.info("Data file has been modified, processing needed")
                else:
                    self.logger.info("No processing metadata found, processing needed")
            else:
                self.logger.info("Collections are empty, processing needed")
        except Exception as e:
            self.logger.error(f"Error checking processing status: {str(e)}")

        return needs_processing

    def load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load and validate CSV data."""
        try:
            self.logger.info(f"Loading data from {file_path}")
            
            df = pd.read_csv(file_path)
            
            # Validate required columns exist
            missing_columns = set(self.required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates(subset='product_id', keep='first')
            if len(df) < initial_rows:
                self.logger.warning(f"Removed {initial_rows - len(df)} duplicate rows")
            
            self._validate_data(df)

            # Store unique categories from the data
            self.categories = set(df['category'].unique())
            self.logger.info(f"Found categories: {self.categories}")

            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {str(e)}")
            raise

    def _clear_existing_data(self) -> None:
        """Reset all collections."""
        try:
            self.logger.info("Clearing all existing collections.")

            # Clear product-level collection
            self.client.delete_collection(name=self.product_collection.name)
            self.product_collection = self.client.create_collection(
                name=self.product_collection.name,
                embedding_function=self.product_collection._embedding_function
            )

            # Clear metric-level collection
            self.client.delete_collection(name=self.metric_collection.name)
            self.metric_collection = self.client.create_collection(
                name=self.metric_collection.name,
                embedding_function=self.metric_collection._embedding_function
            )

            self.logger.info("All collections cleared and reinitialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to clear collections: {str(e)}")
            raise

    def _format_metric_document(self, product: pd.Series, metric: str, value: float, category: str, direction: str) -> str:
        result = ""
        """Format document text based on metric type."""
        if metric == "discount_percentage":
            result = f"Product: '{product['product_name']}', Category: '{category}', Discount: {value}%"
        elif metric == "original_price":
            result = f"Product: '{product['product_name']}', Category: '{category}', Price: ₹{value}"
        elif metric == "rating":
            result = f"Product: '{product['product_name']}', Category: '{category}', Rating: {value}"
        elif metric == "rating_count":
            result = f"Product: '{product['product_name']}', Category: '{category}', Reviews: {int(value)}"
        else:
            result = f"Product: '{product['product_name']}', Category: '{category}', {metric}: {value}"

        return result

    def embed_and_store(self, df: pd.DataFrame) -> None:
        """Embed and store product data into collections."""
        try:
            self.logger.info("Starting embedding and storage process.")
            self._clear_existing_data()

            # Store product-level data
            documents = []
            metadatas = []
            ids = []

            for _, row in df.iterrows():
                # Default product-level document
                product_doc = (
                    f"Product: {row['product_name']}\n"
                    f"ID: {row['product_id']}\n"
                    f"Category: {row['category']}\n"
                    f"Discounted Price: ₹{row['discounted_price']}\n"
                    f"Original Price: ₹{row['actual_price']}\n"
                    f"Discount: {row['discount_percentage']}%\n"
                    f"Rating: {row['rating']}\n"
                    f"Rating Count: {row['rating_count']}"
                )
                documents.append(product_doc)
                metadatas.append(
                    {
                        "product_id": str(row["product_id"]),
                        "product_name": str(row["product_name"]),
                        "category": str(row["category"]),
                        "rating": float(row["rating"]),
                        "rating_count": int(row["rating_count"]),
                        "discounted_price": float(row["discounted_price"]),
                        "original_price": float(row["actual_price"]),  # Derived mapping
                        "discount_percentage": float(row["discount_percentage"]),
                    }
                )
                ids.append(str(row["product_id"]))

            self.product_collection.add(documents=documents, metadatas=metadatas, ids=ids)

            # Store metric-level data
            metrics = ["rating", "rating_count", "original_price", "discount_percentage"]
            metric_docs = {metric: {"highest": None, "lowest": None} for metric in metrics}

            for metric in metrics:
                for category, group in df.groupby("category"):
                    # Map `original_price` to `actual_price`
                    if metric == "original_price":
                        metric_column = group["actual_price"]
                    else:
                        metric_column = group[metric]

                    # Find all products with the highest/lowest values
                    max_value = metric_column.max()
                    min_value = metric_column.min()
                    
                    highest_products = group[metric_column == max_value]
                    lowest_products = group[metric_column == min_value]

                    # Store each highest product separately
                    if metric_docs[metric]["highest"] is None:
                        metric_docs[metric]["highest"] = []

                    for idx, (_, product) in enumerate(highest_products.iterrows()):
                        metric_docs[metric]["highest"].append({
                            "document": self._format_metric_document(
                                product=product,
                                metric=metric,
                                value=max_value,
                                category=category,
                                direction="highest"
                            ),
                            "metadata": {
                                "product_id": str(product["product_id"]),
                                "category": category,
                                "metric": metric,
                                "direction": "highest",
                                "document_key": f"{category}_{metric}_highest",
                                "value": float(max_value),
                                "rank": idx + 1
                            },
                            "id": f"{category}_highest_{metric}_{idx + 1}"
                        })

                    # Store each lowest product separately
                    if metric_docs[metric]["lowest"] is None:
                        metric_docs[metric]["lowest"] = []

                    for idx, (_, product) in enumerate(lowest_products.iterrows()):
                        metric_docs[metric]["lowest"].append({
                            "document": self._format_metric_document(
                                product=product,
                                metric=metric,
                                value=min_value,
                                category=category,
                                direction="lowest"
                            ),
                            "metadata": {
                                "product_id": str(product["product_id"]),
                                "category": category,
                                "metric": metric,
                                "direction": "lowest",
                                "document_key": f"{category}_{metric}_lowest",
                                "value": float(min_value),
                                "rank": idx + 1
                            },
                            "id": f"{category}_lowest_{metric}_{idx + 1}"
                        })

                # Add all metric documents to the metric collection
                for metric, data in metric_docs.items():
                    for direction, entries in data.items():
                        if entries:
                            # Log the document keys being added
                            for entry in entries:
                                document_key = entry["metadata"]["document_key"]
                                self.logger.info(f"Adding metric document with key: {document_key}")
                            
                            self.metric_collection.add(
                                documents=[entry["document"] for entry in entries],
                                metadatas=[entry["metadata"] for entry in entries],
                                ids=[entry["id"] for entry in entries]
                            )

            # Store processing metadata
            self.product_collection.add(
                documents=["metadata"],
                metadatas=[{
                    "document_type": "metadata",
                    "last_processed": str(time.time())
                }],
                ids=["processing_metadata"]
            )

            self.logger.info("Embedding and storage process completed successfully.")

        except Exception as e:
            self.logger.error(f"Failed to embed and store data: {str(e)}")
            raise

    def _build_where_clause(self, query_text: str, category: str) -> dict:
        """Build where clause based on query category."""
        try:
            # Base where clause
            where = {}
            
            if category == "price-related":
                if "product_id" in query_text.lower():  # Case-insensitive check
                    product_id = query_text.split("'")[1]
                    where["product_id"] = {"$eq": product_id}  # Exact match
                else:
                    product_name = query_text.split("'")[1]
                    where["product_name"] = {"$eq": product_name}  # Exact match

            elif category == "rating-related":
                product_id = query_text.split("'")[1]
                where["product_id"] = {"$eq": product_id}  # Exact match

            elif category == "product-specific metadata":
                product_name = query_text.split("'")[1]
                where["product_name"] = {"$eq": product_name}  # Exact match

            elif category == "exploratory":
                if "category" in query_text.lower():  # Case-insensitive check
                    category_name = query_text.split("'")[1]
                    where["category"] = {"$eq": category_name}  # Exact match

            return where

        except Exception as e:
            self.logger.error(f"Failed to build where clause: {str(e)}")
            raise

    def _extract_metric_and_direction(self, query_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract metric and direction (highest/lowest) from query text.

        Parameters:
            query_text (str): The input query text.

        Returns:
            Tuple[Optional[str], Optional[str]]: The extracted metric and direction if found; otherwise, (None, None).
        """
        metric = None
        direction = None

        # Define mappings for supported directions and metrics
        directions = {
            "highest": "highest", "largest": "highest", "most": "highest",
            "lowest": "lowest", "smallest": "lowest", "least": "lowest"
        }
        metrics = {
            "rating": "rating", "discount": "discount_percentage", 
            "price": "original_price", "rating_count": "rating_count"
        }

        # Convert query text to lowercase and split into words
        query_words = query_text.lower().split()

        # Search for a direction keyword in the query text
        for direction, normalized_direction in directions.items():
            if direction in query_words:
                # Find the position of the direction keyword
                index = query_words.index(direction)
                # Check if a metric keyword follows the direction keyword
                if index + 1 < len(query_words):  # Ensure there is a next word
                    next_word = re.sub(r'[^\w\s]', '', query_words[index + 1])
                    # Match the next word to a known metric
                    if next_word in metrics:
                        metric = metrics[next_word]
                        direction = normalized_direction
                        break

        # If no direction and metric are found, return None, None
        return metric, direction

    def _extract_category(self, query_text: str) -> Optional[str]:
        """
        Extract category from query text using regex and validate against known categories.
        
        Args:
            query_text (str): The query text containing a category in quotes
            
        Returns:
            Optional[str]: The validated category if found, None otherwise
        """
        try:
            # Look for text between single or double quotes
            matches = re.findall(r'[\'\"](.*?)[\'\"]', query_text)
            
            # Check if any matched text is a valid category
            for match in matches:
                if match in self.categories:
                    return match
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting category: {str(e)}")
            return None

    def retrieve_chunks(self, query_text: str, category: str, n_results: int = 5) -> List[str]:
        """Retrieve chunks based on query."""
        result = []

        try:
            self.logger.info(f"Retrieving chunks for query: {query_text}")

            if category == "exploratory":
                self.logger.info("Processing exploratory query")
                metric, direction = self._extract_metric_and_direction(query_text)

                if metric:
                    query_category = self._extract_category(query_text)

                    if query_category:
                        document_key = f"{query_category}_{metric}_{direction}"
                        self.logger.info(f"Searching for document key: {document_key}")
                        results = self.metric_collection.query(
                            query_texts=[query_text],
                            n_results=n_results,
                            where={"document_key": document_key}
                        )
                    else:
                        # If no specific category, search across all categories
                        results = self.metric_collection.query(
                            query_texts=[query_text],
                            n_results=n_results,
                            where={"metric": metric, "direction": direction}
                        )
                else:
                    # If no metric/direction found, query product collection instead
                    self.logger.info("No specific metric found, querying product collection")
                    where_clause = self._build_where_clause(query_text, category)
                    results = self.product_collection.query(
                        query_texts=[query_text],
                        n_results=n_results,
                        where=where_clause if where_clause else None
                    )
            else:
                # Non-exploratory queries use the product collection
                where_clause = self._build_where_clause(query_text, category)
                results = self.product_collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where_clause if where_clause else None
                )

            # Return documents from any valid query
            if results['documents'] and results['documents'][0]:
                result = results['documents'][0]

            return result

        except Exception as e:
            self.logger.error(f"Failed to retrieve chunks: {str(e)}")
            raise

    def load_questions(self, questions_path: Path) -> List[Dict]:
        """Load questions from JSONL file, skipping blank lines."""
        try:
            self.logger.info(f"Loading questions from {questions_path}")
            questions = []
            with questions_path.open() as f:
                for line_num, line in enumerate(f, 1):
                    # Skip blank lines
                    if not line.strip():
                        continue
                        
                    try:
                        question = json.loads(line)
                        questions.append({
                            'question': question['prompt'],
                            'expected_answer': question['completion'],
                            'category': question['category']
                        })
                    except json.JSONDecodeError:
                        self.logger.warning(f"Skipping invalid JSON at line {line_num}: {line.strip()}")
                        continue
                        
            self.logger.info(f"Loaded {len(questions)} questions")
            return questions
            
        except Exception as e:
            self.logger.error(f"Failed to load questions: {str(e)}")
            raise
