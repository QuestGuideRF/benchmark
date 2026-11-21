from common_import import *
from activation import PaddleActivation ,TorchActivation
@benchmark_registry .register ("relu",reuse ="activation")
class ReluConfig (APIConfig ):
    def __init__ (self ):
        super (ReluConfig ,self ).__init__ ('relu')
        self .api_name ='relu'
        self .api_list ={
        'elu':'elu',
        'leaky_relu':'leaky_relu',
        'relu':'relu',
        'relu6':'relu6',
        'selu':'selu',
        }
        self .alias_name ="activation"
    def disabled (self ):
        if self .api_name in ["selu"]and self .x_dtype =="float16":
            print (
            "-- Предупреждение:\n"
            "  1. Эта конфигурация отключена, поскольку float16 не поддерживается для %s.\n"
            %(self .api_name ))
            return True
        return super (ReluConfig ,self ).disabled ()
@benchmark_registry .register ("relu")
class TFRelu (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =self .layers (config .api_name ,features =x )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])