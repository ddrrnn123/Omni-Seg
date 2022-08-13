# Develop

This document provides tutorials to develop Omni-Seg.

## New data
Basically there are three steps:

- Annotate the WSI rectangularly. Recommend to use ImageScope and save the .xml file for annotation information. 
- Convert the svs file into PNG files and saved into 40X, 10X and 5X magnifications. Please refer to [Omni_seg_pipeline_gpu/svs_input/svs_to_png.py](Omni_seg_pipeline_gpu/svs_input/svs_to_png.py) for an example to convert svs format to PNG format and resize to different magnifications.
- Create three empty folders named as "40X", "10X", and "5X" under [Omni_seg_pipeline_gpu/svs_input](Omni_seg_pipeline_gpu/svs_input) folder. Put 40X, 10X and 5X PNG files into these folders correspondingly. Each folder must contain only one file when running. 

## New model

You will need to pass "--reload_path" argument when running [Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py](Omni_seg_pipeline_gpu/Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py). Otherwise, the default model will be used without passing in any argument.

For example:
```
$python3 Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py --reload_path new_model.pth
```

## Run Whole Slide Image segmentation pipeline locally without docker
- The entire pipeline is at the [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder
- Create three empty folders in the [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder (before running, these three folders must be empty to remove any previous data): 
  1. "clinical_patches" folder
  2. "segmentation_merge" folder
  3. "final_merge" folder
- Run python scripts in [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder as following orders: 
  1. [1024_Step1_GridPatch_overlap_padding.py](Omni_seg_pipeline_gpu/1024_Step1_GridPatch_overlap_padding.py)
  2. [1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py](Omni_seg_pipeline_gpu/1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py)
  3. [Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py](Omni_seg_pipeline_gpu/Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py)
  4. [step3.py](Omni_seg_pipeline_gpu/step3.py)
  5. [step4.py](Omni_seg_pipeline_gpu/step4.py)
- The output will be stored at "final_merge" folder.
