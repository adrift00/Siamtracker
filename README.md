# SiamTracker
## Implementation 
My Implementation  of siamrpn++, mainly refers to [pysot](https://github.com/STVIR/pysot). However, there are still some problems in the code:
1. The performence of my code is lower than the official implementation, the model which use mobilenetv2 as the backbone only achieves 0.3 EAO in VOT2018, but the official code achives 0.39 EAO in VOT2018. There maybe be still some bugs in the code.
2. There is a memory leak problem in the training process, the memory usage will increase in train process.

## Some useless improments
I try to make some improvements:
1. Add meta-update for siamrpn++ to update model's parameters online, the meta update algorithm is based on [Meta-tracker: Fast and Robust Online Adaptation for Visual Object Trackers](https://arxiv.org/pdf/1801.03049.pdf).
2. Use grad to update the template online, mainly refers to [GradNet: Gradient-Guided Network for Visual Object Tracking] (https://arxiv.org/pdf/1909.06800.pdf).

Unfortunately, these methods have no obvious effect when they are applied to siamrpn++.

## Model pruning
What's more, I apply the model pruning methods to the model to accelate the running speed of the model. I apply [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](http://arxiv.org/abs/1808.06866) and [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](http://arxiv.org/abs/1811.00250) on siamrpn++.

## Model deployment
I also deploy the siamrpn++ model to the android phone, the code is in my other repo [Siamtracker_Android](https://github.com/adrift00/Siamtracker_Android).



