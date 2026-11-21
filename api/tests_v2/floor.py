from common_import import *
class FloorConfig (APIConfig ):
    def __init__ (self ):
        super (FloorConfig ,self ).__init__ ("floor")
        self .feed_spec ={"range":[-1 ,1 ]}
        self .alias_name ="activation"
class PDFloor (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .floor (x =x )
        self .feed_vars =[x ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
class TFFloor (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =tf .math .floor (x =x )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
if __name__ =='__main__':
    test_main (PDFloor (),TFFloor (),config =FloorConfig ())