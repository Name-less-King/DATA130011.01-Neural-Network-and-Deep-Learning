小组成员： 陈乐偲 19307130195      梁之扬 19307130184

# Identify

## 摘要

使用先进的特征提取方法在尽量小的计算的情况下获取重要特征，用于鉴定。

使用 Geometric Tight Frame 以及 Gabor Wavelet 的方法抽取图像特征，并使用 Forward Stagewise技术选取重要的特征， 最后使用各种机器学习和深度学习方法进行真伪鉴定（二分类）。


## 文件说明

* tight_frame_feature.py gabor_wavelet.py 分别是Geometric Tight Frame 与 Gabor Wavelet 方法提取特征

*  feature_extract.py 是 tight_frame_feature.py 的辅助函数

* ml.py dl.py 分别对应机器学习和深度学习分类方法

* \utils 包括 

* prepare_data.py 用于数据预处理 

* forward_stagewise.py 用于特征选取 

* train_test_split_pickle.py 用于训练集测试集划分

  

# Raphael-style-transfer

## 摘要

使用基于卷积神经网络的相关方法实现风格迁移

将实现风格迁移的网络统一在基于generator(endoer-decoder) - discriminator的统一框架内，并且基于统一的架构，实现了一些经典论文中的方法，用于Raphael的风格迁移之中。

对于最基础的Gatys method，使用更高效的基于全卷积神经网络fully-convolution-network的generator替代原方法，加速风格迁移的训练过程。并且分别实现并探究了离线（off-line)和在线（on-line）的方法。并且基于CycleGAN中的循环一致的思想，为Gatys method添加逆向网络（inverse generator）。

针对Raphael画作的特点，使用精细的结构以支持更高的分辨率，并且使用灰度化、边缘增强等图像处理手段作为预处理过程增强风格迁移效果。

在相同的架构下，在encode和decoder之间加入控制风格的模块，可以实现任意风格迁移。任意风格迁移的关键，在于对储存图像风格的信息的理解。对其理解不同，也即SwapStyle和AdaIN两种不同的，任意风格迁移的方法。在统一的架构下，对比两种方法，旨在增强对图像风格信息的理解。


## 文件说明

* \net 定义网络模型
* \old 旧有代码，包括正则化探索，边缘提取增强等探索，使用PatchGAN的探索，以及用卷积作鉴定的部分尝试，一些旧有代码中运用的辅助函数等
* transfer.py off-line的Gatys风格迁移方法
* end2end_transfer.py on-line的Gatys风格迁移方法，在线方法也即端到端的训练方法
* cycle.py 在Gatys方法中加入逆向网络支持循环一致
* StyleSwap.py 任意风格迁移
* adain.py AdaIN任意风格迁移



# Raphael-style-transfer-CycleGAN

## 摘要

在CycleGAN的基础上，对其进行一定程度的改进，将其运用于Raphael风格迁移任务。

CycleGAN源代码对于小样本风格迁移（只有36张风格图片，8张内容图片—）的表现不佳。

为了优化CycleGAN的表现，改动了训练策略，并且经过对比试验后改进了Patch Discriminator的感受野（Patch大小），使得生成的图片既不至于像照片一样精细，又不至于过于抽象而难以理解，使生成的图片正好位于画的范畴之内。

同时，为了直面小样本的问题，尝试使用单一风格图片和单一内容图片进行CycelGAN的训练。在不使用任何数据扩增的前提下，我们采用了一些经典GAN模型中的策略，解决小样本CycleGAN训练问题。

对于单一风格图片的训练，引入SinGAN的思想和训练方法，使用多尺度的CycleGAN构成整体网络。并且尝试了三种训练多尺度网络的训练策略。

对于单一内容图片的训练，引入Patch Permutation GAN中的Patch Permutation模块。


## 文件说明

代码结构和CylceGAN源代码相同，对于改进的代码部分，详见myCycleGAN的README文件。

## 运行说明


使用优化后的CycleGAN模型训练

```
./myCycleGAN/train.py --dataroot {$PATH} --name {$NAME} --gpu_ids {$ID}
```

引入Patch Permutation 模块训练针对单一风格图片的风格迁移模型

```
./myCycleGAN/train_p2gan.py --dataroot {$PATH} --name {$NAME} --gpu_ids {$ID}
```

引入SinGAN的思想训练单一内容图片的风格迁移模型,但训练策略与原SinGAN不同，采用端到端的训练策略

```
./myCycleGAN/train_singan.py --dataroot {$PATH} --name {$NAME} --gpu_ids {$ID}
```

引入SinGAN的思想训练单一内容图片的风格迁移模型,但引入层级的特征融合模块（FFM），以使模型同时获得高尺度的内容信息和低尺度的风格信息

```
./myCycleGAN/train_higan.py --dataroot {$PATH} --name {$NAME} --gpu_ids {$ID}
```

引入SinGAN的思想训练单一内容图片的风格迁移模型,训练方式与原SinGAN相同，当每一尺度的GAN训练完后固定其参数，进行更高尺度的训练

```
./myCycleGAN/train_fixgan.py --dataroot {$PATH} --name {$NAME} --gpu_ids {$ID}
```

