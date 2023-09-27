from variable import Variable
from cal_node import PyCalNode
from bin.Debug.opt_node import CalNode
from bin.Debug.act_func import ReLU

class PyReLU(PyCalNode):
    def __init__(self, graph) -> None:
        self.graph = graph
        self.cal_node = ReLU(self.graph.NextId())
        graph.AddCalNode(self)
        
    def __call__(self, input: Variable) -> Variable:
        self.input = input
        self.cal_node.AddInput(self.input.data_node)
        self.input.data_node.AddConsumer(self.cal_node)
        
        self.output = Variable(self.graph)
        self.cal_node.SetOutput(self.output.data_node)
        self.output.data_node.SetProducer(self.cal_node)
        
        return self.output