from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import matplotlib .image
from paddle import fluid
from .reader import process_image
import benchmark
core =fluid .core
logger =logging .getLogger (__name__ )
class ResNet50Benchmark (benchmark .Benchmark ):
    def __init__ (self ):
        super (ResNet50Benchmark ,self ).__init__ ("resnet50")
    def load_data (self ,filename ):
        image =process_image (filename ,224 )
        tensor =core .PaddleTensor ()
        print (len (image ))
        tensor .shape =[1 ,3 ,224 ,224 ]
        tensor .data =core .PaddleBuf (image .tolist ())
        tensor .dtype =core .PaddleDType .FLOAT32
        tensor .lod =[[0 ,1 ]]
        return [tensor ]