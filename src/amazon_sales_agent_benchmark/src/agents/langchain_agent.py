from typing import List, Optional
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from .base_agent import BaseAgent
from src.prompts.templates import PromptTemplates

class LangChainAgent(BaseAgent):
    """LangChain implementation of agent framework."""

    def __init__(self, llm, templates: dict, logger):
        """Initialize LangChain agent."""
        super().__init__(llm, templates, logger)
        try:
            self.llm = llm.langchain if hasattr(llm, 'langchain') else llm
            self.chains = {}
            self.logger.info("LangChain agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain agent: {str(e)}")
            raise

    def initialize(self) -> None:
        """Initialize LangChain chains for different prompt types."""
        try:
            self.logger.info("Initializing LangChain chains")
            
            if not self.templates:
                raise ValueError("No chains could be initialized from templates. Templates are required")
            
            # Create chains for each template
            for category, template_text in self.templates.items():
                if template_text:
                    # Handle judge template differently
                    if category == "judge_template":
                        prompt = PromptTemplate(
                            input_variables=["responses"],
                            template=template_text
                        )
                    else:
                        prompt = PromptTemplate(
                            input_variables=["context", "question"],
                            template=template_text
                        )
                    self.chains[category] = prompt | self.llm
                else:
                    self.logger.error(f"No template found for category: {category}")
                    raise ValueError(f"No template found for category: {category}")
            
            if not self.chains:
                raise ValueError("No chains could be initialized from templates")
            
            self.initialized = True
            self.logger.info(f"Available chains: {list(self.chains.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize chains: {str(e)}")
            raise RuntimeError(f"Chain initialization failed: {str(e)}")

    def execute_query(self, query: str, category: str, context: Optional[str] = None) -> str:
        """Execute query using appropriate chain based on category."""
        try:
            if not self.initialized:
                raise RuntimeError("Agent not properly initialized")
            
            template_name = PromptTemplates.get_category_mapping().get(category)
            chain = self.chains.get(template_name)
            
            if not chain:
                self.logger.error(f"No chain found for category: {category}")
                raise ValueError(f"No chain found for category: {category}")
            
            response = chain.invoke({
                "context": context or "",
                "question": query
            })

            return self._clean_response(response)  # Use base class method
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            raise

    def aggregate_results(self, results: List[str], original_prompt: str) -> str:
        """Aggregate multiple responses using LLM as judge."""
        try:
            self.logger.info("Aggregating results")
            
            if not results:
                self.logger.info("No results to aggregate")
                return "No results to aggregate."
                       
            judge_chain = self.chains.get("judge_template")
            if not judge_chain:
                self.logger.error("Judge template not found")
                raise ValueError("Judge template not found")
            
            formatted_responses = "\n".join([f"{i+1}. {r}" for i, r in enumerate(results)])
            
            final_response = judge_chain.invoke({
                "responses": formatted_responses,
                "original_prompt": original_prompt
            })
            
            self.logger.info("Results aggregated successfully")
            return self._clean_response(final_response)
            
        except Exception as e:
            self.logger.error(f"Results aggregation failed: {str(e)}")
            raise 