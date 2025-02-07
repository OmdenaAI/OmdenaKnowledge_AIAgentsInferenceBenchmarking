from typing import Dict, Any, TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain.schema import BaseMessage, HumanMessage
import operator
import logging

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    final_answer: Dict[str, Any]

class MultiAgentOrchestrator:
    def __init__(self, router, agents):
        self.router = router
        self.agents = agents
        self.logger = logging.getLogger(__name__)
        self.graph = self._build_graph()

    def _build_graph(self) -> Graph:
        # Create state graph
        workflow = StateGraph(AgentState)

        # Add router node
        def route_question(state: AgentState) -> Dict[str, str]:
            question = state["messages"][-1].content
            router_result = self.router.determine_agent(question)
            return {"agent": router_result["agent"]}

        workflow.add_node("router", route_question)

        # Add agent nodes
        for agent_name, agent in self.agents.items():
            def create_agent_node(agent):
                def run_agent(state: AgentState) -> AgentState:
                    question = state["messages"][-1].content
                    result = agent.execute(question)
                    return {"final_answer": result}
                return run_agent

            workflow.add_node(f"agent_{agent_name}", create_agent_node(agent))

        # Add edges
        workflow.set_entry_point("router")
        
        # Route to appropriate agent based on router's decision
        workflow.add_conditional_edges(
            "router",
            lambda x: x["agent"],  # This function extracts the agent from state
            {
                "math": "agent_math",
                "physics": "agent_physics",
                "chemistry": "agent_chemistry",
                "biology": "agent_biology"
            }
        )

        # End states
        for agent_name in self.agents.keys():
            workflow.set_finish_point(f"agent_{agent_name}")

        return workflow.compile()


    def run(self, question: str) -> Dict[str, Any]:
        final_result = {}

        try:
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "current_agent": "",
                "final_answer": {}
            }
            result = self.graph.invoke(initial_state)
            final_result = {
                **result["final_answer"]
            }
        except Exception as e:
            self.logger.error(f"Error in orchestrator: {e}")
            raise

        return final_result