# Omni-Seg: A Scale-aware Dynamic Network for Pathological Image Segmentation

### [[GPU Accelerated Pipeline Docker]](https://github.com/MASILab/SLANTbrainSeg/tree/master/python) [[project page]](https://https://github.com/ddrrnn123/Omni-Seg/)   [[Journal paper]](https://arxiv.org/pdf/2206.13632v1.pdf) [[MIDL 2022 paper]](https://openreview.net/pdf?id=v-z4Zxkt9Ex) [[SPIE 2023 Paper]](https://arxiv.org/pdf/2206.13632v1.pdf)<br />


This is the official implementation of Omni-Seg: A Scale-aware Dynamic Network for Pathological Image Segmentation. <br />


![Overview](https://github.com/ddrrnn123/Omni-Seg/blob/main/GithubFigure/Overview1.png)<br />
![Docker](https://github.com/ddrrnn123/Omni-Seg/blob/main/GithubFigure/Overview2.png)<br />

Journal Paper <br />
[A Scale-aware Dynamic Network for Pathological Image Segmentation](https://arxiv.org/pdf/2206.13632v1.pdf) <br />
Ruining Deng, Quan Liu, Can Cui, Tianyuan Yao, Jun Long, Zuhayr Asad, R. Michael Womick, Zheyu Zhu, Agnes B. Fogo, Shilin Zhao, Haichun Yang, Yuankai Huo. <br />
*(Under review)* <br />

MIDL Paper <br />
[Omni-Seg: A Single Dynamic Network for Multi-label Renal Pathology Image Segmentation using Partially Labeled Data](https://openreview.net/pdf?id=v-z4Zxkt9Ex) <br />
Ruining Deng, Quan Liu, Can Cui, Zuhayr Asad, Haichun Yang, Yuankai Huo. <br />
*MIDL 2022* <br />

SPIE Paper <br />
[An Accelerated Pipeline for Multi-label Renal Pathology Image Segmentation at the Whole Slide Image Level](https://arxiv.org/pdf/2206.13632v1.pdf)<br />
Haoju Leng*, Ruining Deng*, Zuhayr Asad, R. Michael Womick, Haichun Yang, Lipeng Wan, and Yuankai Huo.<br />
*(Under review)* <br />

```diff
+ We release an accelerated pipeline as a single [Docker](https://github.com/MASILab/SLANTbrainSeg/tree/master/python).
```

## Abstract
Comprehensive semantic segmentation on renal pathological images is challenging due to the heterogeneous scales of the objects. For example, on a whole slide image (WSI), the cross-sectional areas of glomeruli can be 64 times larger than that of the peritubular capillaries, making it impractical to segment both objects on the same patch, at the same scale. To handle this scaling issue, we propose the Omni-Seg network, a scale-aware dynamic neural network that achieves multi-object (six tissue types) and multi-scale (5$\times$ to 40$\times$ scale) pathological image segmentation via a single neural network.<br /> 

The contribution of this paper is three-fold: 
(1) a novel scale-aware controller is proposed to generalize the dynamic neural network from single-scale to multi-scale; <br />
(2) semi-supervised consistency regularization of pseudo-labels is introduced to model the inter-scale correlation of unannotated tissue types into a single end-to-end learning paradigm;<br />
(3) superior scale-aware generalization is evidenced by directly applying a model trained on human kidney images to mouse kidney images, without retraining. 


cite: 
```
@inproceedings{
deng2022omniseg,
title={Omni-Seg: A Single Dynamic Network for Multi-label Renal Pathology Image Segmentation using Partially Labeled Data},
author={Ruining Deng and Quan Liu and Can Cui and Zuhayr Asad and Haichun Yang and Yuankai Huo},
booktitle={Medical Imaging with Deep Learning},
year={2022},
url={https://openreview.net/forum?id=v-z4Zxkt9Ex}
}
```

Journal version: Under review  
arxiv: https://arxiv.org/abs/2206.13632v1  
