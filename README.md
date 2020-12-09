# Modified-Batch-Normalization

# Objective
Use strides in the input while calculating mean and variance and DepthwiseConv2D with kernel size 1 in the output to replace scale and shift parameters to decrease computation time.

OBN- Original Batch Normalization, MBN- Modified Batch Normalization
## Stride and Depthwise Conv Study
    1. We ran all the models with OBN and MBN
    2. We used strides 1, 2, 4, 8 and plotted time taken for batch normalization in all layers using Tensorboard profile (for both Forward and Back Propagation).
    3. We trained using Resnet9 on Cifar10 (32x32) and Resnet50 on Imagenette (224x224) dataset.
    4. Conclusion - No increase/decrease in computation time in BN layers
## Depthwise Conv Study
### Resnet9 Cifar10
    1. We ran the models with OBN and MBN
    2. We trained using Resnet9 on Cifar10 dataset with no stride and trained it for 24 epochs
    3. We trained it for 30 sessions on a Tesla P100 PCIe GPU
    4. Conclusion - No statistically significant increase/decrease in accuracy/time
### Resnet50 Imagenet
    1. We ran the models with OBN and MBN following the documentation provided in AWS 
    2. Conclusion - No significant increase/decrease in the accuracy, however, MBN model takes more time than the OBN Model.
### Resnet50 Imagenette
    1. We ran the models with OBN and MBN
    2. We trained using Resnet50 on Imagenette dataset with no stride and trained it for 10 epochs
    3. We trained it for 30 sessions on a Tesla P100 PCIe GPU
    4. Conclusion - No statistically significant increase/decrease in the accuracy, however, MBN model takes 1 second/epoch (on average) more than the OBN Model. 
	Reasons:
        a. We are calculating normed values of x there itself in our code and then passing this value to DepthwiseConv2D, whereas in original BatchNorm the mean and variance are passed to nn.fused_batch_norm which converts the complete batch normalization calculation in 1x1 form, which is ~12%-30% faster. (Reference1, Reference2). This was implemented as a default within the past 5-6 months.
### Mean Stddev Beta Gamma Variations
### Resnet9 Cifar10
    1. Same results for both OBN and MBN in first, middle and last BN layers.
### Conclusion: The idea we were trying to implement has been recently implemented as a default operation under nn.fused_batch_norm function.
