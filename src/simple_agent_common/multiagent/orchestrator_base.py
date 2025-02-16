from abc import abstractmethod
from typing import Dict, Any, List

class OrchestratorBase:

    @abstractmethod
    def run(self, question: str) -> Dict[str, Any]:
        pass