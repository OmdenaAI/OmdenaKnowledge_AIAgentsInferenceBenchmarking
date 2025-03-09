from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from simple_agent_common.utils import TokenCounter

class LangChainTokenCounter(TokenCounter):
    def count_messages(self, messages: List[BaseMessage]) -> Dict[str, int]:
        """Count tokens in a list of LangChain messages"""
        prompt_tokens = 0
        completion_tokens = 0
        
        for message in messages:
            if isinstance(message, AIMessage):
                completion_tokens += self.count_tokens(message.content)
            else:
                prompt_tokens += self.count_tokens(message.content)
                
        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
        } 