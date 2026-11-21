from common_import import *
class YoloBoxConfig (APIConfig ):
    def __init__ (self ):
        super (YoloBoxConfig ,self ).__init__ ("yolo_box")
        self .run_tf =False
class PDYoloBox (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        img_size =self .variable (
        name ='img_size',
        shape =config .img_size_shape ,
        dtype =config .img_size_dtype )
        boxes ,scores =paddle .fluid .layers .yolo_box (
        x =x ,
        img_size =img_size ,
        class_num =config .class_num ,
        anchors =config .anchors ,
        conf_thresh =config .conf_thresh ,
        downsample_ratio =config .downsample_ratio )
        self .feed_vars =[x ,img_size ]
        self .fetch_vars =[boxes ,scores ]
if __name__ =='__main__':
    test_main (PDYoloBox (),config =YoloBoxConfig ())