from common_import import *
class PDAssign (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
        result =paddle .assign (data )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
class TFAssign (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .input_shape ,dtype =config .input_dtype )
        ref =self .variable (
        name ='target',
        shape =config .input_shape ,
        dtype =config .input_dtype ,
        value =np .zeros (config .input_shape ).astype (config .input_dtype ))
        result =ref .assign (data )
        self .feed_list =[data ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDAssign (),TFAssign (),config =APIConfig ("assign"))