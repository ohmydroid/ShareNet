# ShareNet
Two convolutional layer in the bottleneck block share parameters, which can be formulated as:  
$$Y=W^{T}A(WX)$$

Such pattern can be extended into FFN or MOE FFN of large language models to reduce parameters and accelerate model traning and inference. 
