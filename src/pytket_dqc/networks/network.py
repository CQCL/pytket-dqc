from abc import ABC, abstractmethod
import networkx as nx  # type:ignore


class Network(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_server_nx(self) -> nx.Graph:
        pass

    @abstractmethod
    def get_full_nx(self) -> nx.Graph:
        pass

    @abstractmethod
    def draw(self):
        pass
