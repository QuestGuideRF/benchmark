from common_import import *
@benchmark_registry .register ("multi_dot")
class PaddleMultiDot (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        inputs =[]
        for i in range (len (config .inputs_shape )):
            input_i =self .variable (
            name ='input_'+str (i ),
            shape =config .inputs_shape [i ],
            dtype =config .inputs_dtype [i ])
            inputs .append (input_i )
        result =paddle .linalg .multi_dot (inputs )
        self .feed_list =inputs
        self .fetch_list =[result ]
@benchmark_registry .register ("multi_dot")
class TorchMultiDot (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        inputs =[]
        for i in range (len (config .inputs_shape )):
            input_i =self .variable (
            name ='input_'+str (i ),
            shape =config .inputs_shape [i ],
            dtype =config .inputs_dtype [i ])
            inputs .append (input_i )
        result =torch .linalg .multi_dot (inputs )
        self .feed_list =inputs
        self .fetch_list =[result ]