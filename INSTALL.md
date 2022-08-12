## Installation

We used PyTorch 1.9.1 on [Ubuntu 22.04 LTS](https://releases.ubuntu.com/22.04/) with [Anaconda](https://www.anaconda.com/download) Python 3.7.

1. [Optional but recommended] Create a new Conda environment. 

    ~~~
    conda create --name omni_seg python=3.7
    ~~~
    
    And activate the environment.
    
    ~~~
    conda activate omni_seg
    ~~~

2. Clone the Omni_Seg repo

3. Install the [requirements](https://github.com/ddrrnn123/Omni-Seg/blob/main/Omni_seg_pipeline_gpu/requirements.txt)

4. Install [apex](https://github.com/NVIDIA/apex):
    ~~~
    cd Omni_Seg/Omni_seg_pipeline_gpu/apex
    python3 setup.py install
    cd ..
    ~~~
