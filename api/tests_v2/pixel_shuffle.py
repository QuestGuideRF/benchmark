from common_import import *
class PDPixelShuffle (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =paddle .nn .functional .pixel_shuffle (
        x =x ,
        upscale_factor =config .upscale_factor ,
        data_format =config .data_format )
        self .feed_vars =[x ]
        self .fetch_vars =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
class TFPixelShuffle (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        out =tf .nn .depth_to_space (
        input =x ,
        block_size =config .upscale_factor ,
        data_format =config .data_format )
        self .feed_list =[x ]
        self .fetch_list =[out ]
        if config .backward :
            self .append_gradients (out ,[x ])
if __name__ =='__main__':
    test_main (
    PDPixelShuffle (),TFPixelShuffle (),config =APIConfig ("pixel_shuffle"))