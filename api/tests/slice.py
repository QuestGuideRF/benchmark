from common_import import *
@benchmark_registry .register ("slice")
class SliceConfig (APIConfig ):
    def __init__ (self ):
        super (SliceConfig ,self ).__init__ ("slice")
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        super (SliceConfig ,self ).init_from_json (filename ,config_id ,
        unknown_dim )
        if len (self .axes )>1 :
            print (
            "Warning:\n"
            "  1. оси параметров длиной больше 1 не поддерживаются \n"
            "в torch.narrow() будет выполнен torch.narrow()")
            self .run_torch =False
        if len (self .starts )>1 :
            print (
            "Warning:\n"
            "  2. параметр, начинающийся с длины больше 1, не поддерживается \n"
            "в torch.narrow() будет выполнен torch.narrow()")
            self .run_torch =False
        if len (self .ends )>1 :
            print (
            "Warning:\n"
            "  3. параметр заканчивается длиной больше 1, не поддерживается \n"
            "в torch.narrow() будет выполнен torch.narrow()")
            self .run_torch =False
    def to_pytorch (self ):
        torch_config =super (SliceConfig ,self ).to_pytorch ()
        torch_config .length =self .ends [0 ]-self .starts [0 ]
        return torch_config
    def to_tensorflow (self ):
        tf_config =super (SliceConfig ,self ).to_tensorflow ()
        tf_config .begin =[]
        tf_config .size =[]
        for i in range (len (self .input_shape )):
            tf_config .begin .append (0 )
            tf_config .size .append (self .input_shape [i ])
        if len (self .starts )<len (self .input_shape ):
            for c ,i in enumerate (self .axes ):
                tf_config .begin [i ]=self .starts [c ]
                tf_config .size [i ]=self .ends [c ]-self .starts [c ]
        else :
            for j in range (len (self .starts )):
                tf_config .begin [i ]=self .starts [j ]
                if self .ends [j ]-self .starts [j ]<self .input_shape [j ]:
                    tf_config .size [j ]=self .ends [j ]-self .starts [j ]
        return tf_config
@benchmark_registry .register ("slice")
class PaddleSlice (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (
        name ='x',shape =config .input_shape ,dtype =config .input_dtype )
        result =paddle .slice (
        input =x ,axes =config .axes ,starts =config .starts ,ends =config .ends )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("slice")
class TorchSlice (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (
        name ='x',shape =config .input_shape ,dtype =config .input_dtype )
        result =torch .narrow (
        input =x ,
        dim =config .axes [0 ],
        start =config .starts [0 ],
        length =config .length ).contiguous ()
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("slice")
class TFSlice (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        result =tf .slice (input_ =input ,begin =config .begin ,size =config .size )
        self .feed_list =[input ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])