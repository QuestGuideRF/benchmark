from common_import import *
@benchmark_registry .register ("psroi_pool")
class PsroiPoolConfig (APIConfig ):
    def __init__ (self ):
        super (PsroiPoolConfig ,self ).__init__ ('psroi_pool')
        self .run_torch =False
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (PsroiPoolConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        self .boxes_value =np .array (
        [[1 ,5 ,8 ,10 ],[4 ,2 ,6 ,7 ],[12 ,12 ,19 ,21 ]])
        self .boxes_num_value =np .array ([1 ,2 ])
@benchmark_registry .register ("psroi_pool")
class PaddlePsroiPool (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        boxes =self .variable (
        name ='boxes',
        shape =config .boxes_value .shape ,
        dtype =config .x_dtype ,
        value =config .boxes_value )
        boxes_num =self .variable (
        name ='boxes_num',
        shape =config .boxes_num_value .shape ,
        dtype ='int32',
        value =config .boxes_num_value )
        result =paddle .vision .ops .psroi_pool (
        x =x ,
        boxes =boxes ,
        boxes_num =boxes_num ,
        output_size =config .output_size ,
        spatial_scale =config .spatial_scale )
        self .feed_list =[x ,boxes ,boxes_num ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])