# UNet

[UNET](https://arxiv.org/abs/1505.04597) was developed by Olaf Ronneberger, Philipp Fischer, and Thomas Brox for medical image segmentation.
The original paper presents the following architecture 

![UNet](images/unet.png)

## References

- [Understanding Semantic Segmentation with UNET](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)
- [U-Net For Segmenting Seismic Images With Keras](https://www.depends-on-the-definition.com/unet-keras-segmenting-images/)
- [Up-sampling with Transposed Convolution](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0)

## Todo

1. Current implementation is based on original UNet paper.
   Input layer is set to be of shape (572, 572, 3) to accept RGB images.
2. Loss function needs to be implemented. Also other metrics might be handy.
3. Score method for the UNet class needs to be implemented.
3. Prediction method for the UNet class needs to be implemented.

## Other things to read

- [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf)
- [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
