from common_import import *
from pad2d import PaddlePad2d ,TorchPad2d
@benchmark_registry .register ("pad3d",reuse ="pad2d")
class Pad3dConfig (APIConfig ):
    def __init__ (self ):
        super (Pad3dConfig ,self ).__init__ ('pad3d')
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (Pad3dConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if self .mode =="reflect":
            print (
            "Warning:\n"
            "  1. Режим отражения в torch.nn.functional.pad не реализован.\n"
            )
            self .run_torch =False
        if len (self .x_shape )!=5 :
            print ("Warning:\n"
            "  2. Размер входных данных должен быть 5.\n")
@benchmark_registry .register ("pad3d")
class TFPad3d (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        padding =[
        [0 ,0 ],
        [0 ,0 ],
        [config .pad [4 ],config .pad [5 ]],
        [config .pad [2 ],config .pad [3 ]],
        [config .pad [0 ],config .pad [1 ]],
        ]
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =tf .pad (x ,
        padding ,
        mode =str .upper (config .mode ),
        constant_values =config .value )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])