import numpy as np
def get_out_scoremap (net ):
    return net .blobs ['score'].data [0 ].argmax (axis =0 ).astype (np .uint8 )
def feed_net (net ,in_ ):
    net .blobs ['data'].reshape (1 ,*in_ .shape )
    net .blobs ['data'].data [...]=in_
def segrun (net ,in_ ):
    feed_net (net ,in_ )
    net .forward ()
    return get_out_scoremap (net )
def fast_hist (a ,b ,n ):
    k =np .where ((a >=0 )&(a <n ))[0 ]
    bc =np .bincount (n *a [k ].astype (int )+b [k ],minlength =n **2 )
    if len (bc )!=n **2 :
        return 0
    return bc .reshape (n ,n )
def get_scores (hist ):
    acc =np .diag (hist ).sum ()/(hist .sum ()+1e-12 )
    cl_acc =np .diag (hist )/(hist .sum (1 )+1e-12 )
    iu =np .diag (hist )/(hist .sum (1 )+hist .sum (0 )-np .diag (hist )+1e-12 )
    return acc ,np .nanmean (cl_acc ),np .nanmean (iu ),cl_acc ,iu