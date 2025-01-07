from typing import List, Dict, TypedDict, Optional
from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from .base_agent import BaseAgent
from src.prompts.templates import PromptTemplates

class AgentState(TypedDict):
    """State for query processing workflow."""
    query: str
    context: str
    response: Optional[str]
    template_type: str
    error: Optional[str]

class LangGraphAgent(BaseAgent):
    """LangGraph implementation focused on efficient query processing."""

    def __init__(self, llm, templates: dict, logger):
        """Initialize with standard components."""
        super().__init__(llm, templates, logger)
        self.llm = llm.langchain if hasattr(llm, 'langchain') else llm
        self.workflow = None
        self.category_mapping = None
        self.chains = {}
        self.logger.info("LangGraph agent initialized")

    def _create_chain(self, template: str) -> RunnableSequence:
        """Create a processing chain for a template."""
        prompt = PromptTemplate.from_template(template)
        return prompt | self.llm

    def initialize(self) -> None:
        """Initialize workflow graph."""
        try:
            self.logger.info("Initializing LangGraph workflow")
            self.category_mapping = PromptTemplates.get_category_mapping()
            
            # Initialize chains including judge template
            for name, template in self.templates.items():
                if name == 'judge_template':
                    # Create judge chain with specific prompt structure
                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["responses", "original_prompt"]
                    )
                    self.chains[name] = prompt | self.llm
                else:
                    # Create regular query chains
                    self.chains[name] = self._create_chain(template)
            
            workflow = StateGraph(AgentState)
            
            # Define process node function
            def process_query(state: AgentState) -> Dict:
                try:
                    template_type = state["template_type"]
                    
                    if template_type not in self.chains:
                        raise ValueError(f"Unknown template type: {template_type}")
                    
                    self.logger.info(f"Processing with template: {template_type}")
                    
                    response = self.chains[template_type].invoke({
                        "context": state["context"],
                        "question": state["query"]
                    })
                    
                    return {
                        **state,
                        "response": self._clean_response(response)
                    }
                    
                except Exception as e:
                    return {
                        **state,
                        "error": str(e)
                    }

            # Add nodes
            workflow.add_node("process", process_query)
            workflow.add_node("end", lambda x: x)

            # Connect nodes
            workflow.add_edge("process", "end")
            
            # Set entry and exit points
            workflow.set_entry_point("process")
            workflow.set_finish_point("end")

            self.workflow = workflow.compile()
            self.initialized = True
            self.logger.info(f"Workflow initialized with templates: {list(self.chains.keys())}")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def execute_query(self, query: str, category: str, context: Optional[str] = None) -> str:
        """Execute single query through workflow."""
        try:
            if not self.initialized:
                raise RuntimeError("Agent not initialized")

            template_name = self.category_mapping.get(category)
            if not template_name:
                raise ValueError(f"Unknown category: {category}")

            # Create initial state
            state = AgentState(
                query=query,
                context=context or "",
                response=None,
                template_type=template_name,
                error=None
            )

            # Process through workflow
            final_state = self.workflow.invoke(state)
            
            if final_state.get("error"):
                raise RuntimeError(final_state["error"])
                
            return final_state["response"]  # Already cleaned in process_response

        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            raise

    def aggregate_results(self, results: List[str], original_prompt: str) -> str:
        """
        Aggregate multiple responses using the judge template.

        Args:
            results (List[str]): List of responses to aggregate.
            original_prompt (str): The original question/prompt.

        Returns:
            str: The best response chosen by the judge.
        """
        if not results:
            return "No results to aggregate."
        
        if len(results) == 1:
            return results[0]
        
        judge_chain = self.chains.get("judge_template")
        if not judge_chain:
            self.logger.error("Judge template not found")
            raise ValueError("Judge template not found")

        # Prepare input to match template variables
        input_data = {
            "responses": "\n".join(f"{i + 1}. {response}" for i, response in enumerate(results)),
            "original_prompt": original_prompt
        }

        try:
            response = judge_chain.invoke(input_data)
            cleaned_response = self._clean_response(response)
            self.logger.info(f"Aggregated {len(results)} responses")
            return cleaned_response
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            raise
