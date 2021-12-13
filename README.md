# cifar10-quantization-demo

代码基于https://github.com/foolwood/pytorch-slimming改写

主要实现pytorch自带的量化，从float32量化到int8，网络很小，仅训练20个epoch。然后生成pt文件。

运行结果，跑10000张测试集，在x86的cpu上，推理速度加速一倍

```
Test set: Accuracy: 8118/10000 (81.2%)

float 32 inference time: 7.420151710510254

Test set: Accuracy: 8101/10000 (81.0%)

int 8 inference time: 3.5435214042663574
```







