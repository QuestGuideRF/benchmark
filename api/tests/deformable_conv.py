from common_import import *
@benchmark_registry .register ("deformable_conv")
class DeformableConvConfig (APIConfig ):
    def __init__ (self ,op_type ="deformable_conv"):
        super (DeformableConvConfig ,self ).__init__ (op_type )
        self .run_tf =False
        self .feed_spec =[
        {
        "range":[0 ,1 ]
        },
        {
        "range":[0 ,1 ],
        },
        {
        "range":[0 ,1 ],
        },
        {
        "range":[0 ,1 ],
        }
        ]
    def to_pytorch (self ):
        torch_config =super (DeformableConvConfig ,self ).to_pytorch ()
        if isinstance (self .stride ,int ):
            torch_config .stride =[self .stride ,self .stride ]
        if isinstance (self .dilation ,int ):
            torch_config .dilation =[self .dilation ,self .dilation ]
        if isinstance (self .padding ,int ):
            torch_config .padding =[self .padding ,self .padding ]
        return torch_config
@benchmark_registry .register ("deformable_conv")
class PaddleDeformableConv (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ='weight',
        shape =config .weight_shape ,
        dtype =config .weight_dtype )
        offset =self .variable (
        name ='offset',
        shape =config .offset_shape ,
        dtype =config .offset_dtype )
        mask =self .variable (
        name ='mask',shape =config .mask_shape ,dtype =config .mask_dtype )
        result =paddle .vision .ops .deform_conv2d (
        x =x ,
        offset =offset ,
        weight =weight ,
        stride =config .stride ,
        dilation =config .dilation ,
        padding =config .padding ,
        deformable_groups =config .deformable_groups ,
        groups =config .groups ,
        mask =mask )
        self .feed_list =[x ,weight ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ])
@benchmark_registry .register ("deformable_conv")
class TorchDeformableConv (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        import torchvision
        input =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ='weight',shape =config .weight_shape ,dtype =config .x_dtype )
        offset =self .variable (
        name ='offset',
        shape =config .offset_shape ,
        dtype =config .offset_dtype )
        mask =self .variable (
        name ='mask',shape =config .mask_shape ,dtype =config .mask_dtype )
        result =torchvision .ops .deform_conv2d (
        input =input ,
        offset =offset ,
        weight =weight ,
        stride =config .stride ,
        padding =config .padding ,
        dilation =config .dilation ,
        mask =mask )
        self .feed_list =[input ,weight ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ,weight ])