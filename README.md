# CACR-CL
Pytorch implementation of [**Contrastive Attraction and Contrastive Repulsion for Representation Learning**](https://arxiv.org/abs/2105.03746).

## CACR: Distributional Self-supervised learning
![motiv](assets/overview.png)

### Introduction
[CACR](https://arxiv.org/abs/2105.03746) is a distributional self-supervised learning method. Both positive samples and negative samples have their distribution in the representation space. CACR leverages a Bayesian strategy to align the positive distribution and distinguish the negative distribution.


### Structure of this repository
This repository contains three folders, respectively corresponds to our experiments on small-scale (on both balanced/imbalanced) datasets, large-scale standard dataset (ImageNet) and large-scale label-shifted dataset.

- To reproduce our results on small-scale experiments (CIFAR10/CIFAR100/STL10), please refer to *small_scale_experiments* folder.

- To reproduce our main results on standard large-scale experiments (ImageNet1K), please refer to *imagenet_pretraining* folder.

- To reproduce our results on small-scale experiments (ImageNet22K/Webvision -> ImageNet1K), please refer to *large_scale_shift_experiments* folder.

More details regarding the training configuration and running command are explained in the README under each subfolder.

### Main Results on ImageNet

#### ImageNet pretrained, performance of linear classification on ImageNet
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">model</th>
<th valign="center">pretrain<br/>epochs</th>
<th valign="center">linear<br/>acc</th>
<th valign="center">checkpoint</th>
<!-- TABLE BODY -->
<tr>
<td align="left">ResNet50</td>
<td align="right">1000</td>
<td align="center">74.7</td>
<td align="center"><a href="https://drive.google.com/file/d/17mE7obaWAG0-YMu3ffZLK-DkhNmn2hhB/view?usp=sharing">download</a></td>
</tr>
<tr>
<td align="left">ViT-Base</td>
<td align="right">300</td>
<td align="center">77.1</td>
<td align="center"><a href="https://drive.google.com/file/d/19BRxMvhdYcCLFHfgGQ-FZQibAN9IdurK/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

#### ImageNet pretrained, performance of linear classification on 20 Image in the Wild datasets
Please feel free to check our learned representation performance in [Image in the Wild Challenge](https://eval.ai/web/challenges/challenge-page/1832/leaderboard/4301).



### Citation
Please cite our work if you find it is helpful. Thank you!
```
@article{
  zheng2023contrastive,
  title={Contrastive Attraction and Contrastive Repulsion for Representation Learning},
  author={Huangjie Zheng and Xu Chen and Jiangchao Yao and Hongxia Yang and Chunyuan Li and Ya Zhang and Hao Zhang and Ivor Tsang and Jingren Zhou and Mingyuan Zhou},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=f39UIDkwwc},
}
```
