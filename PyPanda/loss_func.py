from variable import Variable
from bin.Debug.opt_node import CalNode
from bin.Debug.loss_func import MSELoss, CELoss

class PyMSELoss():
    def __init__(self, graph) -> None:
        self.graph = graph
        self.cal_node = MSELoss(self.graph.NextId()) 
        graph.AddCalNode(self)
        
    def __call__(self, predict: Variable, target: Variable) -> Variable:
        self.predict = predict
        self.target = target
        self.cal_node.AddInput(self.predict.data_node)
        self.cal_node.AddInput(self.target.data_node)
        self.predict.data_node.AddConsumer(self.cal_node)
        self.target.data_node.AddConsumer(self.cal_node)
        
        self.output = Variable(self.graph)
        self.cal_node.SetOutput(self.output.data_node)
        self.output.data_node.SetProducer(self.cal_node)
        
        return self.output
    
class PyCELoss():
    def __init__(self, graph) -> None:
        self.graph = graph
        self.cal_node = CELoss(self.graph.NextId()) 
        graph.AddCalNode(self)
        
    def __call__(self, predict: Variable, target: Variable) -> Variable:
        self.predict = predict
        self.target = target
        self.cal_node.AddInput(self.predict.data_node)
        self.cal_node.AddInput(self.target.data_node)
        self.predict.data_node.AddConsumer(self.cal_node)
        self.target.data_node.AddConsumer(self.cal_node)
        
        self.output = Variable(self.graph)
        self.cal_node.SetOutput(self.output.data_node)
        self.output.data_node.SetProducer(self.cal_node)
        
        return self.output