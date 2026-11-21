from common_import import *
@benchmark_registry .register ("broadcast_tensors")
class BroadcastTensorsConfig (APIConfig ):
    def __init__ (self ):
        super (BroadcastTensorsConfig ,self ).__init__ ('broadcast_tensors')
        self .run_torch =False
@benchmark_registry .register ("broadcast_tensors")
class PaddleBroadcastTensors (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        inputs =[]
        for i in range (len (config .input_shape )):
            input =self .variable (
            name ='input_'+str (i ),
            shape =config .input_shape [i ],
            dtype =config .input_dtype [i ])
            inputs .append (input )
        result =paddle .broadcast_tensors (input =inputs )
        self .feed_list =[inputs ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,inputs )