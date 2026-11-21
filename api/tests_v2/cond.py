from common_import import *
class PDCond (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        def true_fn ():
            return paddle .multiply (x ,y )
        def false_fn ():
            return paddle .divide (x ,y )
        ten_var =paddle .full (
        shape =config .input_shape ,dtype =config .input_dtype ,fill_value =10 )
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        pred =paddle .less_than (input ,ten_var )
        result =paddle .static .nn .cond (pred ,true_fn ,false_fn )
        self .feed_vars =[x ,y ,input ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])
class TFCond (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        def true_fn ():
            return tf .multiply (x ,y )
        def false_fn ():
            return tf .divide (x ,y )
        ten_var =tf .constant (
        10 ,shape =config .input_shape ,dtype =config .input_dtype )
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        pred =tf .less (input ,ten_var )
        result =tf .cond (tf .reshape (pred ,[]),true_fn ,false_fn )
        self .feed_list =[x ,y ,input ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])
if __name__ =='__main__':
    test_main (PDCond (),TFCond (),config =APIConfig ('cond'))