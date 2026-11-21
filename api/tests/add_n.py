from common_import import *
@benchmark_registry .register ("add_n")
class PDAddN (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        inputs =[]
        for i in range (len (config .inputs_shape )):
            input_i =self .variable (
            name ='input_'+str (i ),
            shape =config .inputs_shape [i ],
            dtype =config .inputs_dtype [i ])
            inputs .append (input_i )
        result =paddle .add_n (inputs =inputs )
        self .feed_list =inputs
        self .fetch_list =[result ]
@benchmark_registry .register ("add_n")
class TorchAddN (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        input_list =[]
        input_0 =self .variable (
        name ='input_'+str (0 ),
        shape =config .inputs_shape [0 ],
        dtype =config .inputs_dtype [0 ])
        result =input_0
        input_list .append (input_0 )
        for i in range (1 ,len (config .inputs_shape )):
            input_i =self .variable (
            name ='input_'+str (i ),
            shape =config .inputs_shape [i ],
            dtype =config .inputs_dtype [i ])
            result =torch .add (result ,input_i )
            input_list .append (input_i )
        self .feed_list =input_list
        self .fetch_list =[result ]
@benchmark_registry .register ("add_n")
class TFAddN (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        inputs =[]
        for i in range (len (config .inputs_shape )):
            input_i =self .variable (
            name ='input_'+str (i ),
            shape =config .inputs_shape [i ],
            dtype =config .inputs_dtype [i ])
            inputs .append (input_i )
        result =tf .add_n (inputs =inputs )
        self .feed_list =inputs
        self .fetch_list =[result ]