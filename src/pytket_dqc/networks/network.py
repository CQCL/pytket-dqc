from abc import ABC, abstractmethod
import networkx as nx  # type:ignore


class Network(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_nx(self) -> nx.Graph:
        pass

    @abstractmethod
    def draw(self):
        pass
