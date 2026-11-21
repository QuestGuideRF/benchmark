from common_import import *
class PDPad3d (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .nn .functional .pad (x =x ,
        pad =config .pad ,
        mode =config .mode ,
        value =config .value ,
        data_format =config .data_format )
        self .feed_vars =[x ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
class TFPad3d (TensorflowAPIBenchmarkBase ):
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
if __name__ =='__main__':
    test_main (PDPad3d (),TFPad3d (),config =APIConfig ("pad3d"))