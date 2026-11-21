from common_import import *
class FloorDivideConfig (APIConfig ):
    def __init__ (self ):
        super (FloorDivideConfig ,self ).__init__ ("floor_divide")
        self .feed_spec =[{"range":[1 ,1000 ]},{"range":[1 ,1000 ]}]
class PDFloorDivide (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =paddle .floor_divide (x =x ,y =y )
        self .feed_vars =[x ,y ]
        self .fetch_vars =[result ]
class TFFloorDivide (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =tf .math .floordiv (x =x ,y =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDFloorDivide (),TFFloorDivide (),config =FloorDivideConfig ())