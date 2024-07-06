# Omni-Seg: A Scale-aware Dynamic Network for Pathological Image Segmentation

### [[Accelerated Pipeline Docker]](https://hub.docker.com/repository/docker/lengh2/omni_seg) [[Project Page]](https://https://github.com/ddrrnn123/Omni-Seg/)   [[IEEE TBME Paper]](https://ieeexplore.ieee.org/document/10079171) [[MIDL 2022 paper]](https://openreview.net/pdf?id=v-z4Zxkt9Ex) [[SPIE 2023 Paper]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12471/124710Q/An-accelerated-pipeline-for-multi-label-renal-pathology-image-segmentation/10.1117/12.2653651.full)<br />


This is the official implementation of Omni-Seg: A Scale-aware Dynamic Network for Pathological Image Segmentation. <br />


![Overview](https://github.com/ddrrnn123/Omni-Seg/blob/main/GithubFigure/Overview1.png)<br />
![Docker](https://github.com/ddrrnn123/Omni-Seg/blob/main/GithubFigure/Overview2.png)<br />

**IEEE TBME Paper** <br />
> [Omni-Seg: A Scale-aware Dynamic Network for Pathological Image Segmentation](https://ieeexplore.ieee.org/document/10079171) <br />
> Ruining Deng, Quan Liu, Can Cui, Tianyuan Yao, Jun Long, Zuhayr Asad, R. Michael Womick, Zheyu Zhu, Agnes B. Fogo, Shilin Zhao, Haichun Yang, Yuankai Huo. <br />
> *IEEE Transactions on Biomedical Engineering* <br />

**MIDL Paper** <br />
> [Omni-Seg: A Single Dynamic Network for Multi-label Renal Pathology Image Segmentation using Partially Labeled Data](https://openreview.net/pdf?id=v-z4Zxkt9Ex) <br />
> Ruining Deng, Quan Liu, Can Cui, Zuhayr Asad, Haichun Yang, Yuankai Huo. <br />
> *MIDL 2022* <br />

**SPIE Paper** <br />
> [An Accelerated Pipeline for Multi-label Renal Pathology Image Segmentation at the Whole Slide Image Level](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12471/124710Q/An-accelerated-pipeline-for-multi-label-renal-pathology-image-segmentation/10.1117/12.2653651.full)<br />
> Haoju Leng*, Ruining Deng*, Zuhayr Asad, R. Michael Womick, Haichun Yang, Lipeng Wan, and Yuankai Huo.<br />
> *SPIE 2023* <br />

```diff
+ We release an accelerated pipeline as a single Docker.
```

## Abstract
Comprehensive semantic segmentation on renal pathological images is challenging due to the heterogeneous scales of the objects. For example, on a whole slide image (WSI), the cross-sectional areas of glomeruli can be 64 times larger than that of the peritubular capillaries, making it impractical to segment both objects on the same patch, at the same scale. To handle this scaling issue, we propose the Omni-Seg network, a scale-aware dynamic neural network that achieves multi-object (six tissue types) and multi-scale (5$\times$ to 40$\times$ scale) pathological image segmentation via a single neural network.<br /> 

The contribution of this paper is three-fold: <br />
(1) a novel scale-aware controller is proposed to generalize the dynamic neural network from single-scale to multi-scale; <br />
(2) semi-supervised consistency regularization of pseudo-labels is introduced to model the inter-scale correlation of unannotated tissue types into a single end-to-end learning paradigm;<br />
(3) superior scale-aware generalization is evidenced by directly applying a model trained on human kidney images to mouse kidney images, without retraining. 

## Quick Start
#### Get our docker image

```
sudo docker pull lengh2/omni_seg
```
#### Run Omni-Seg
You can run the following command or change the `input_dir`, then you will have the final segmentation results in `output_dir`. Please refer to "New data" section at [DEVELOP.md](https://github.com/ddrrnn123/Omni-Seg/blob/main/DEVELOP.md) for input data format requirement and data arrangement.
```
# you need to specify the input directory. 
export input_dir=/home/input_dir   
# make that directory
sudo mkdir $input_dir
# set output directory
export output_dir=$input_dir/output
#run the docker
sudo nvidia-docker run --shm-size 64G -it --rm -v $input_dir:/INPUTS/ -v $output_dir:/OUTPUTS lengh2/omni_seg 
```
<!---
## Google Colab (need modify)
-->


## Installation
Please refer to [INSTALL.md](https://github.com/ddrrnn123/Omni-Seg/blob/main/INSTALL.md) for installation instructions of the segmentation.

## Model
Pretrained model can be found [here](https://github.com/ddrrnn123/Omni-Seg/tree/main/Omni_seg_pipeline_gpu/snapshots_2D)

## Data
The training data can be found [here](http://haeckel.case.edu/data/KI_data/)

The example dataset for the pipeline which contains a .SVS input file and three .PNG files with different magnifications generated from the .SVS file can be found [here](https://drive.google.com/drive/folders/1Nx2fSltW3HyPGXiCwNVA4Nci8c_wJ7RZ)

## Omni-Seg - Region Image Demo
Omni-Seg can easily be run on a single image.

Below is an example input of region image.

<img src='GithubFigure/region_input.png' align="center" height="230px"> 

- The entire pipeline is at the [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder
- Create three empty folders named as "40X", "10X", and "5X" under [Omni_seg_pipeline_gpu/svs_input](Omni_seg_pipeline_gpu/svs_input) folder. Put 40X, 10X and 5X PNG files of the region image into these folders correspondingly. Each folder must contain only one file when running.
- Create three empty folders in the [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder (before running, these three folders must be empty to remove any previous data): 
  1. "clinical_patches" folder
  2. "segmentation_merge" folder
  3. "final_merge" folder
- Run the python scripts as following orders:
  1. [1024_Step1_GridPatch_overlap_padding.py](Omni_seg_pipeline_gpu/1024_Step1_GridPatch_overlap_padding.py)
  ```
  python 1024_Step1_GridPatch_overlap_padding.py
  ```
  2. [1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py](Omni_seg_pipeline_gpu/1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py)
  ```
  python 1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py
  ```
  3. [Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py](Omni_seg_pipeline_gpu/Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py)
  ```
  python Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py --reload_path 'snapshots_2D/fold1_with_white_UNet2D_ns_normalwhole_1106/MOTS_DynConv_fold1_with_white_UNet2D_ns_normalwhole_1106_e89.pth'
  ```
  4. [step3.py](Omni_seg_pipeline_gpu/step3.py)
  ```
  python step3.py
  ```
  5. [step4.py](Omni_seg_pipeline_gpu/step4.py)
  ```
  python step4.py
  ```
- The output will be stored at "final_merge" folder.

If set up correctly, the output should look like

<img src='GithubFigure/region_output.png' align="center" height="230px"> 

## Omni-Seg - Whole Slide Image Demo
CircleNet can also be run on Whole Slide Images in *.svs file format.

Please download the following file:
- [Human Kidney WSI (3d90_PAS.svs)](https://vanderbilt.box.com/s/sskcgbvz15bcfuh9sra96u1dy1hzqm6o)

We need to annotate and convert data into *.png file format first.

- Annotate the WSI rectangularly to remove most of the empty background. Recommend to use ImageScope and save the .xml file for annotation information. 
- Convert the svs file into PNG files and saved into 40X, 10X and 5X magnifications. Please refer to [Omni_seg_pipeline_gpu/svs_input/svs_to_png.py](Omni_seg_pipeline_gpu/svs_input/svs_to_png.py) for an example to convert svs format to PNG format and resize to different magnifications.
- Create three empty folders named as "40X", "10X", and "5X" under [Omni_seg_pipeline_gpu/svs_input](Omni_seg_pipeline_gpu/svs_input) folder. Put 40X, 10X and 5X PNG files into these folders correspondingly. Each folder must contain only one file when running. 

After annotation, the inputs should be like the following image with three different magnifications

<img src='GithubFigure/WSI_input.png' align="center" height="350px"> 

Please create three empty folders in the [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder (before running, these three folders must be empty to remove any previous data): 
  1. "clinical_patches" folder
  2. "segmentation_merge" folder
  3. "final_merge" folder
  
To run the Omni-Seg pipeline, please go to [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder and run the python scipts as following orders:
  1. [1024_Step1_GridPatch_overlap_padding.py](Omni_seg_pipeline_gpu/1024_Step1_GridPatch_overlap_padding.py)
  ```
  python 1024_Step1_GridPatch_overlap_padding.py
  ```
  2. [1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py](Omni_seg_pipeline_gpu/1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py)
  ```
  python 1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py
  ```
  3. [Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py](Omni_seg_pipeline_gpu/Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py)
  ```
  python Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py --reload_path 'snapshots_2D/fold1_with_white_UNet2D_ns_normalwhole_1106/MOTS_DynConv_fold1_with_white_UNet2D_ns_normalwhole_1106_e89.pth'
  ```
  4. [step3.py](Omni_seg_pipeline_gpu/step3.py)
  ```
  python step3.py
  ```
  5. [step4.py](Omni_seg_pipeline_gpu/step4.py)
  ```
  python step4.py
  ```

The output will be stored at "final_merge" folder.

If set up correctly, the output should look like

<img src='GithubFigure/WSI_output.png' align="center" height="350px"> 





## Develop
Please refer to [DEVELOP.md](https://github.com/ddrrnn123/Omni-Seg/blob/main/DEVELOP.md) to train Omni-Seg on a new dataset, design a new architecture based on Omni-Seg.

## Previous Versions
#### Google Colab
A Google Colab version of the Oracle pipeline can be found [here](https://drive.google.com/drive/folders/1vKeDMYI3Xcm6s9yAy5stBhqKWjOFvUoy?usp=sharing). The code demonstrates the patch-wise segmentation of the Oracle pipeline. 

## Acknowledgments
This code is inspired by [DoDNet](https://github.com/jianpengz/DoDNet).

## MIT License
Copyright <2024> Biomedical Data Representation and Learning Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Citation
If you are using our pipeline or code, please cite:
```

@article{deng2023omni,
  title={Omni-Seg: A Scale-Aware Dynamic Network for Renal Pathological Image Segmentation},
  author={Deng, Ruining and Liu, Quan and Cui, Can and Yao, Tianyuan and Long, Jun and Asad, Zuhayr and Womick, R Michael and Zhu, Zheyu and Fogo, Agnes B and Zhao, Shilin and others},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2023},
  publisher={IEEE}
}

@inproceedings{deng2022single,
  title={Single Dynamic Network for Multi-label Renal Pathology Image Segmentation},
  author={Deng, Ruining and Liu, Quan and Cui, Can and Asad, Zuhayr and Huo, Yuankai and others},
  booktitle={International Conference on Medical Imaging with Deep Learning},
  pages={304--314},
  year={2022},
  organization={PMLR}
}

@inproceedings{leng2023accelerated,
  title={An accelerated pipeline for multi-label renal pathology image segmentation at the whole slide image level},
  author={Leng, Haoju and Deng, Ruining and Asad, Zuhayr and Womick, R Michael and Yang, Haichun and Wan, Lipeng and Huo, Yuankai},
  booktitle={Medical Imaging 2023: Digital and Computational Pathology},
  volume={12471},
  pages={174--179},
  year={2023},
  organization={SPIE}
}


```
