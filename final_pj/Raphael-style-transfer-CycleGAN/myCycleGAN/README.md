## 摘要

基于CycleGAN官方源代码进行修改，实现Raphael画作的风格迁移。


## 代码说明

myCycleGAN /models/networks.py 中定义了MyPatchDiscriminator等，针对该任务，优化原本的CycleGAN。

myCycleGAN /options/base_options.py 与 train_options.py 进行了netD，netG，训练策略等的改动。

myCycleGAN /shuffle/shuffle.py 中定义了Patch Permutation Operation相关的函数

myCycleGAN /models/transformer.py 定义了使用全卷积网络实现的generator，可用于替代CycleGAN原本的ResNet和Unet作为生成网络

其余部分基本同CycleGAN源代码


