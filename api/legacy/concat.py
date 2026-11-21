from common_import import *
class PDConcat (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        inputs =[]
        for i in range (len (config .input_shape )):
            input_i =self .variable (
            name ='input_'+str (i ),
            shape =config .input_shape [i ],
            dtype =config .input_dtype [i ])
            inputs .append (input_i )
        result =fluid .layers .concat (input =inputs ,axis =config .axis )
        self .feed_vars =inputs
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,inputs )
class TFConcat (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        inputs =[]
        for i in range (len (config .input_shape )):
            input_i =self .variable (
            name ='input_'+str (i ),
            shape =config .input_shape [i ],
            dtype =config .input_dtype [i ])
            inputs .append (input_i )
        result =tf .concat (values =inputs ,axis =config .axis )
        self .feed_list =inputs
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,inputs )
if __name__ =='__main__':
    test_main (PDConcat (),TFConcat (),config =APIConfig ("concat"))