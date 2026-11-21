from common_import import *
class PDElementwiseSum (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        inputs =[]
        for i in range (len (config .inputs_shape )):
            input_i =self .variable (
            name ='input_'+str (i ),
            shape =config .inputs_shape [i ],
            dtype =config .inputs_dtype [i ])
            inputs .append (input_i )
        result =fluid .layers .sums (input =inputs )
        self .feed_vars =inputs
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,inputs )
class TFElementwiseSum (TensorflowAPIBenchmarkBase ):
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
        if config .backward :
            self .append_gradients (result ,inputs )
if __name__ =='__main__':
    test_main (
    PDElementwiseSum (),
    TFElementwiseSum (),
    config =APIConfig ("elementwise_sum"))