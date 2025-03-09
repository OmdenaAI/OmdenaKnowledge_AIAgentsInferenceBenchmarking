from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """Base state definition for all agents."""
    messages: List[BaseMessage]  # Conversation history

class LanggraphAgentTemplate:
    """Base template for LangGraph agents with common functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the agent with configuration.
        
        Args:
            config: Dictionary containing agent configuration
        """
        self.config = config
    
    def create_graph(self) -> StateGraph:
        """Create the agent's state graph.
        
        Returns:
            StateGraph: The configured state graph for the agent
        """
        # Create a new graph
        workflow = StateGraph(AgentState)
        
        # Add single processing node
        workflow.add_node("process", self.process_step)
        
        # Set entry point
        workflow.set_entry_point("process")
        
        return workflow
    
    def process_step(self, state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
        """Process the agent's task.
        
        Args:
            state: Current state of the agent
            config: Optional configuration for the runnable
            
        Returns:
            AgentState: Updated state after processing
        """
        raise NotImplementedError("Subclasses must implement process_step")
    
    def run(self, initial_state: Optional[AgentState] = None) -> AgentState:
        """Run the agent workflow.
        
        Args:
            initial_state: Optional initial state for the agent
            
        Returns:
            AgentState: Final state after workflow completion
        """
        graph = self.create_graph()
        
        if initial_state is None:
            initial_state = AgentState(
                messages=[]
            )
        
        # Run the graph
        app = graph.compile()
        final_state = app.invoke(initial_state)
        
        return final_state 