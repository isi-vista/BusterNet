import numpy as np 
from matplotlib import pyplot

def simple_cmfd_decoder( busterNetModel, rgb ) :
    """A simple BusterNet CMFD decoder
    """
    # 1. expand an image to a single sample batch
    single_sample_batch = np.expand_dims( rgb, axis=0 )
    # 2. perform busterNet CMFD
    pred = busterNetModel.predict( single_sample_batch )[0]
    return pred

def visualize_result( rgb, gt, pred, figsize=(12,4), title=None ) :
    """Visualize raw input, ground truth, and BusterNet result
    """
    pyplot.figure( figsize=figsize )
    pyplot.subplot(131)
    pyplot.imshow( rgb )
    pyplot.title('input image')
    pyplot.subplot(132)
    pyplot.title('ground truth')
    pyplot.imshow(gt)
    pyplot.subplot(133)
    pyplot.imshow(pred)
    pyplot.title('busterNet pred')
    if title is not None :
        pyplot.suptitle( title )