# Combined Margin (caffe)

[English version](README.md)

- [Combined Margin (caffe)](#combined-margin-caffe)
  - [介绍](#%E4%BB%8B%E7%BB%8D)
  - [安装](#%E5%AE%89%E8%A3%85)
  - [update 2018-11-11 放出一个Resnet-36模型](#update-2018-11-11-%E6%94%BE%E5%87%BA%E4%B8%80%E4%B8%AAresnet-36%E6%A8%A1%E5%9E%8B)
  - [update 2018-12-10 放出训练log和solver等](#update-2018-12-10-%E6%94%BE%E5%87%BA%E8%AE%AD%E7%BB%83log%E5%92%8Csolver%E7%AD%89)

## 介绍

这是 Combined Margin Loss 的一个 caffe 实现版本。

本工程受到 [insightface](https://github.com/deepinsight/insightface) 和 [arcface-caffe](https://github.com/xialuxi/arcface-caffe) 的启发而成。

Combined Margin Loss 是由 [insightface](https://github.com/deepinsight/insightface) 提出的，旨在对 [Sphereface](https://github.com/wy1iu/sphereface)、 [AMSoftmax(CosineFace)](https://github.com/happynear/AMSoftmax) 和 Additive Angular Margin 进行改进。其中 Additive Angular Margin 即是 [insightface](https://github.com/deepinsight/insightface)。

本工程按照 [insightface](https://github.com/deepinsight/insightface) 中的算法进行实现,并且做了一些修改。主要是在计算输入 softmax 层的 logits 值时，对加入 combined margin 之后的角度 θ 做了边界处理。同时对 BP 时的梯度进行了边界保护，防止梯度过大或过小。

**请注意**：本工程中的实现方法并没有任何加速收敛的技巧，因此在训练时可能会出现 loss 值非常大的现象，猜想这是 Combined Margin 本身相比于 CosFace 和 ArcFace 更加 Hard 的原因。

Combined Margin 的实验效果可以参照 insightface 相关的实验内容，其人脸验证结果好于上述其他方法。参见 [Verification Results On Combined Margin](https://github.com/deepinsight/insightface#verification-results-on-combined-margin)。

如果你想要尝试其他方法，请参照 [insightface](https://github.com/deepinsight/insightface), [arcface-caffe](https://github.com/xialuxi/arcface-caffe) 和 [AMSoftmax(CosineFace)](https://github.com/happynear/AMSoftmax)

## 安装

1. 将 [toadd.proto](toadd.proto) 的内容加入到 caffe 的 caffe.proto 中，请按照 [toadd.proto](toadd.proto) 里介绍的方法进行操作。
2. 将本工程中所有 `.hpp` 文件复制到 `$CAFFE_ROOT/include/caffe/layers/` 文件夹，将所有 `.cpp` and `.cu` 文件复制到 `$CAFFE_ROOT/src/caffe/layers/` 文件夹。如果需要的话请覆盖同名文件。
3. 在 `$CAFFE_ROOT` 文件夹执行 `make all`。可能需要先运行 `make clean` 。
4. 现在你可以在训练中使用 Combined Margin 了。这里有一个例子 [example.prototxt](example.prototxt)，这个例子是从 [AMSoftmax's prototxt](https://github.com/happynear/AMSoftmax/blob/master/prototxt/face_train_test.prototxt) 修改得来的。你也可以通过将 **LabelSpecificAddLayer** 修改成 **CombinedMarginLayer** 来使用 Combined Margin，请不要忘记修改该层的相关参数配置。

有任何问题请开 issue，我会尽量回答。

## update 2018-11-11 放出一个Resnet-36模型

[这里](https://pan.baidu.com/s/18kCU-6rcyetI9UBHCJT_XQ)是一个 ResNet-36 模型 (password: 6sgx), 训练数据是 Deepglint 提供的 ms-celeb-1m 和 asian-celeb。这个模型在 LFW 上可以达到 99.75%。在 [BLUFR](http://www.cbsr.ia.ac.cn/users/scliao/projects/blufr/)上达到 99.69% VR@FAR0.1%, 99.53%	@FAR0.01%, 99.42% Top1@FAR1%。暂时没有做其他测试。

## update 2018-12-10 放出训练log和solver等

res36-training-config文件夹中是上述ResNet-36模型的训练log, 训练所用的solver.prototxt和train.prototxt(password: y672)。所用训练数据是 [Trillionpairs by Deepglint](http://trillionpairs.deepglint.com/data)。所有人脸都对齐到112x96。这一数据集提供了5点人脸特征点的信息，因此只需直接将人脸抠出来，而不需要再做人脸检测和特征点定位。