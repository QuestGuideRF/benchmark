from common_import import *
class PDStack (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        xs =[]
        for i in range (len (config .x_shape )):
            x_i =self .variable (
            name ='x_'+str (i ),
            shape =config .x_shape [i ],
            dtype =config .x_dtype [i ])
            xs .append (x_i )
        result =fluid .layers .stack (x =xs ,axis =config .axis )
        self .feed_vars =xs
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,xs )
class TFStack (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        values =[]
        for i in range (len (config .x_shape )):
            value_i =self .variable (
            name ='value_'+str (i ),
            shape =config .x_shape [i ],
            dtype =config .x_dtype [i ])
            values .append (value_i )
        result =tf .stack (values =values ,axis =config .axis )
        self .feed_list =values
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,values )
if __name__ =='__main__':
    test_main (PDStack (),TFStack (),config =APIConfig ("stack"))