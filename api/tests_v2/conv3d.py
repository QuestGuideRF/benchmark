from common_import import *
class Conv3dConfig (APIConfig ):
    def __init__ (self ,op_type ="conv3d"):
        super (Conv3dConfig ,self ).__init__ (op_type )
        self .feed_spec =[
        {
        "range":[-1 ,1 ]
        },
        {
        "range":[-1 ,1 ],
        "permute":[2 ,3 ,4 ,1 ,0 ]
        }
        ]
        self .run_tf =False
class PDConv3d (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        weight =self .variable (
        name ='weight',
        shape =config .weight_shape ,
        dtype =config .weight_dtype )
        result =paddle .nn .functional .conv3d (
        x =x ,
        weight =weight ,
        bias =None ,
        stride =config .stride ,
        padding =config .padding ,
        dilation =config .dilation ,
        groups =config .groups ,
        data_format =config .data_format )
        self .feed_vars =[x ,weight ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,weight ])
if __name__ =='__main__':
    test_main (PDConv3d (),config =Conv3dConfig ())