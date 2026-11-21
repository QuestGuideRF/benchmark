import os ,sys
import numpy as np
try :
    import paddle
    import paddle .fluid as fluid
except ImportError :
    sys .stderr .write (
    "Невозможно импортировать весло или Paddle.fluid, возможно, весло не установлено.\n"
    )
try :
    import tensorflow as tf
except ImportError :
    sys .stderr .write (
    "Невозможно импортировать tensorflow, возможно, tensorflow не установлен.\n")
package_path =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
sys .path .append (package_path )
from common .paddle_op_benchmark import PaddleAPIBenchmarkBase
from common .tensorflow_op_benchmark import TensorflowAPIBenchmarkBase
from common .api_param import APIConfig
from common .main import test_main ,test_main_without_json
def use_gpu ():
    return os .environ .get ("CUDA_VISIBLE_DEVICES",None )!=""