from common_import import *
class PDConcat (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        xs =[]
        for i in range (len (config .x_shape )):
            x_i =self .variable (
            name ='input_'+str (i ),
            shape =config .x_shape [i ],
            dtype =config .x_dtype [i ])
            xs .append (x_i )
        result =paddle .concat (x =xs ,axis =config .axis )
        self .feed_vars =xs
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,xs )
class TFConcat (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        xs =[]
        for i in range (len (config .x_shape )):
            x_i =self .variable (
            name ='x_'+str (i ),
            shape =config .x_shape [i ],
            dtype =config .x_dtype [i ])
            xs .append (x_i )
        result =tf .concat (values =xs ,axis =config .axis )
        self .feed_list =xs
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,xs )
if __name__ =='__main__':
    test_main (PDConcat (),TFConcat (),config =APIConfig ("concat"))