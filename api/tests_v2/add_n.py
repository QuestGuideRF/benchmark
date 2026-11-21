from common_import import *
class PDAddN (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        inputs =[]
        for i in range (len (config .inputs_shape )):
            input_i =self .variable (
            name ='input_'+str (i ),
            shape =config .inputs_shape [i ],
            dtype =config .inputs_dtype [i ])
            inputs .append (input_i )
        result =paddle .add_n (inputs =inputs )
        self .feed_vars =inputs
        self .fetch_vars =[result ]
class TFAddN (TensorflowAPIBenchmarkBase ):
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
if __name__ =='__main__':
    test_main (PDAddN (),TFAddN (),config =APIConfig ("add_n"))