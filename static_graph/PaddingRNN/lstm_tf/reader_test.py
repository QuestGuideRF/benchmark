from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os .path
import reader
def testPtbProducer ():
    raw_data =[4 ,3 ,2 ,1 ,0 ,5 ,6 ,1 ,1 ,1 ,1 ,0 ,3 ,4 ,1 ]
    batch_size =3
    num_steps =2
    iter =reader .get_data_iter (raw_data ,batch_size ,num_steps )
    print (iter )
    for batch in iter :
        x ,y =batch
        print (x )
        print (y )
        print ("11")
testPtbProducer ()