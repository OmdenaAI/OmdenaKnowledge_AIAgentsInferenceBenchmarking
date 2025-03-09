from typing import Dict, Any, Optional
import logging
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from agents import DataAgentState, YieldAgentState, DataPreparationAgent, PredictionAgent
from simple_agent_common.utils import MemoryManager

class OrchestratorState(DataAgentState, YieldAgentState):
    """Combined state for the full workflow."""
    pass

def build_graph(
    config: Dict[str, Any],
    logger: logging.Logger,
    llm: Optional[ChatGroq] = None,
    memory_manager: Optional[MemoryManager] = None
) -> StateGraph:
    """Build the complete agent workflow graph.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        llm: Optional ChatGroq instance (will be created if not provided)
        
    Returns:
        StateGraph: Configured workflow graph
    """
    # Create LLM if not provided
    if llm is None:
        raise ValueError("LLM is not provided")
    
    # Initialize agents
    data_agent = DataPreparationAgent(config=config, logger=logger)
    yield_agent = PredictionAgent(llm=llm, config=config, logger=logger, memory_manager=memory_manager)
    
    # Create workflow graph
    workflow = StateGraph(OrchestratorState)
    
    # Add nodes for sequential processing
    workflow.add_node("prepare_data", data_agent.process_step)
    workflow.add_node("make_predictions", yield_agent.process_step)
    
    # Add edge for sequential flow
    workflow.add_edge("prepare_data", "make_predictions")
    
    # Set entry point
    workflow.set_entry_point("prepare_data")
    
        # Create initial state
    initial_state = create_initial_state()
    
    # Compile and run
    app = workflow.compile()

    return app

def create_initial_state() -> OrchestratorState:
    """Create initial state for the workflow.
    
    Returns:
        OrchestratorState: Initial workflow state
    """
    return OrchestratorState(
        messages=[],
        dataset=None,
        questions=None,
        predictions=[]
    )


def run_workflow(
    app: StateGraph
) -> OrchestratorState:
    """Run the complete workflow.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        llm: Optional ChatGroq instance
        
    Returns:
        OrchestratorState: Final workflow state with predictions
    """

    # Create initial state
    initial_state = create_initial_state()
    
    # Compile and run
    final_state = app.invoke(initial_state)
    
    return final_state 