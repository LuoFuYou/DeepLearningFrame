import numpy as np
from bin.Debug.data_node import DataNode

class Variable():
    def __init__(self, graph) -> None:
        self.graph = graph
        self.data_node = DataNode(graph.NextId())
        graph.AddDataNode(self)
        
    def SetData(self, data: np.ndarray):
        self.data_node.SetData(data)