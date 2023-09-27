from variable import Variable
from cal_node import PyCalNode
from bin.Debug.graph import Graph

class PyGraph():
    def __init__(self) -> None:
        self.graph = Graph()
        
    def __lshift__(self, begin_node: Variable):
        self.graph.AddBeginNode(begin_node.data_node)
        
    def NextId(self) -> int:
        return self.graph.NextId()
    
    def AddDataNode(self, variable: Variable):
        self.graph.AddDataNode(variable.data_node)
        
    def AddCalNode(self, cal_node: PyCalNode):
        self.graph.AddCalNode(cal_node.cal_node)
    
    def Forward(self):
        self.graph.Forward()
        
    def Backward(self):
        self.graph.Backward()
        
    def ZeroGrad(self):
        self.graph.ZeroGrad()
        
    def UpdateParams(self, lr: float):
        self.graph.UpdateParams(lr)