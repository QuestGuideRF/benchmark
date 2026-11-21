from common_import import *
class MatmulConfig (APIConfig ):
    def __init__ (self ):
        super (MatmulConfig ,self ).__init__ ("matmul")
        self .feed_spec =[{"range":[-1 ,1 ]},{"range":[-1 ,1 ]}]
class PDMatmul (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =fluid .layers .matmul (
        x =x ,
        y =y ,
        transpose_x =config .transpose_x ,
        transpose_y =config .transpose_y ,
        alpha =config .alpha )
        self .feed_vars =[x ,y ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])
class TFMatmul (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =tf .matmul (
        a =x ,
        b =y ,
        transpose_a =config .transpose_x ,
        transpose_b =config .transpose_y ,
        adjoint_a =False ,
        adjoint_b =False ,
        a_is_sparse =False ,
        b_is_sparse =False )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])
if __name__ =='__main__':
    test_main (PDMatmul (),TFMatmul (),config =MatmulConfig ())