from common_import import *
class RemainderConfig (APIConfig ):
    def __init__ (self ):
        super (RemainderConfig ,self ).__init__ ("remainder")
        self .feed_spec =[{"range":[-1000 ,1000 ]},{"range":[1 ,1000 ]}]
        self .alias_name ="elementwise"
    def disabled (self ):
        if self .x_dtype =="float16":
            print (
            "Warning:\n"
            "  1. Эта конфигурация отключена, поскольку float16 не поддерживается для %s.\n"
            %(self .api_name ))
            return True
        return super (RemainderConfig ,self ).disabled ()
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (RemainderConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if len (self .x_shape )>len (self .y_shape )and self .y_shape !=[1 ]:
            self .y_shape =unsqueeze_short (
            short =self .y_shape ,long =self .x_shape )
        elif len (self .x_shape )<len (self .y_shape )and self .x_shape !=[1 ]:
            self .x_shape =unsqueeze_short (
            short =self .x_shape ,long =self .y_shape )
class PDRemainder (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =paddle .remainder (x =x ,y =y )
        self .feed_vars =[x ,y ]
        self .fetch_vars =[result ]
class TFRemainder (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        result =tf .math .floormod (x =x ,y =y )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDRemainder (),TFRemainder (),config =RemainderConfig ())