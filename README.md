# Combined Margin (caffe)

[中文版本README](README_zh.md)

## Introduction

In this repo, a caffe implementation of Combined Margin is introduced.

This repo was inspired by [insightface](https://github.com/deepinsight/insightface) and [arcface-caffe](https://github.com/xialuxi/arcface-caffe)

Combined Margin was proposed by [insightface](https://github.com/deepinsight/insightface) as the improvement of [Sphereface](https://github.com/wy1iu/sphereface), [AMSoftmax(CosineFace)](https://github.com/happynear/AMSoftmax), and Additive Angular Margin which was proposed by insightface before.

This implementation follows the method in [insightface](https://github.com/deepinsight/insightface), and do some modification. Mainly adding bounds for the logits after adding margin, so the logits value of ground truth won't get bigger after adding margin, instead of getting smaller which is our original purpose. And also add bound for the gradient.

Note that the combined margin in this implementation is rather hard without any tricks to help the converge while training.

According to insightface's experiments, the validation results of Combined Margin is better than the other methods mentioned above. see [Verification Results On Combined Margin](https://github.com/deepinsight/insightface#verification-results-on-combined-margin).

If you want to try other methods, please refer to [insightface](https://github.com/deepinsight/insightface), [arcface-caffe](https://github.com/xialuxi/arcface-caffe) and [AMSoftmax(CosineFace)](https://github.com/happynear/AMSoftmax)

## Installation

1. Merge [toadd.proto](toadd.proto) with caffe's caffe.proto, follow the instructions in [toadd.proto](toadd.proto).
2. Place all the `.hpp` files in `$CAFFE_ROOT/include/caffe/layers/`, and all the `.cpp` and `.cu` files in `$CAFFE_ROOT/src/caffe/layers/`. Replace the original files if necessary.
3. Go to `$CAFFE_ROOT` and `make all`.
   Maybe you need to do `make clean` first.
4. Now you can use Combined Margin Layer in your caffe training. Here's an [example.prototxt](example.prototxt) which is modified from [AMSoftmax's prototxt](https://github.com/happynear/AMSoftmax/blob/master/prototxt/face_train_test.prototxt). You can just change the **LabelSpecificAddLayer** into **CombinedMarginLayer**, and don't forget to change the layer parameters.

If you have any question, feel free to open an issue.

Anyone use the code, please site the original papers

### update 2018-11-11

[Here](https://pan.baidu.com/s/1bqClfIvSIcjFAWyExcvI1w)'s one ResNet-36 model (password: pzx2), trained on ms-celeb-1m and asian-celeb provided by Deepglint. This model can get 99.75% on LFW. And on [BLUFR](http://www.cbsr.ia.ac.cn/users/scliao/projects/blufr/), it gets 99.69% VR@FAR0.1%, 99.53%	@FAR0.01%, 99.42% Top1@FAR1%
. I didn't do other test.
