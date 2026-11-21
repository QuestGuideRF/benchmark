from common_import import *
class PDWhereIndex (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        value =paddle .fluid .layers .where (data )
        self .feed_vars =[data ]
        self .fetch_vars =[value ]
        print (self .fetch_vars )
class TFWhereIndex (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        value =tf .where (data )
        self .feed_list =[data ]
        self .fetch_list =[value ]
        print (self .fetch_list )
if __name__ =='__main__':
    test_main (PDWhereIndex (),TFWhereIndex (),config =APIConfig ("where_index"))