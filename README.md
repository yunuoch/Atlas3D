# Atlas3D: Physically Constrained Self-Supporting Text-to-3D for Simulation and Fabrication

### [[Project Page](https://yunuoch.github.io/Atlas3D/)] [[arXiv](https://arxiv.org/abs/2405.18515)]

Yunuo Chen<sup>1</sup>\*, Tianyi Xie<sup>1</sup>\*, Zeshun Zong<sup>1</sup>\*, Xuan Li<sup>1</sup>, Feng Gao<sup>2</sup>, Yin Yang<sup>3</sup>, Ying Nian Wu<sup>1</sup>, Chenfanfu Jiang<sup>1</sup><br>
<sup>1</sup>University of California, Los Angeles, <sup>2</sup>Amazon<small><small><small>(This work is not related to F. Gao’s position at Amazon.)</small></small></small>, <sup>3</sup>University of Utah <br>
*Equal contributions

![teaser.jpg](assets/teaser.jpg)

Abstract: *Existing diffusion-based text-to-3D generation methods primarily focus on producing visually realistic shapes and appearances, often neglecting the physical constraints necessary for downstream tasks. Generated models frequently fail to maintain balance when placed in physics-based simulations or 3D printed. This balance is crucial for satisfying user design intentions in interactive gaming, embodied AI, and robotics, where stable models are needed for reliable interaction. Additionally, stable models ensure that 3D-printed objects, such as figurines for home decoration, can stand on their own without requiring additional supports. To fill this gap, we introduce Atlas3D, an automatic and easy-to-implement method that enhances existing Score Distillation Sampling (SDS)-based text-to-3D tools. Atlas3D ensures the generation of self-supporting 3D models that adhere to physical laws of stability under gravity, contact, and friction. Our approach combines a novel differentiable simulation-based loss function with physically inspired regularization, serving as either a refinement or a post-processing module for existing frameworks. We verify Atlas3D's efficacy through extensive generation tasks and validate the resulting 3D models in both simulated and real-world environments.*


# Quick Start

## Install dependency
```shell
# Tested on Ubuntu 20.04 + Python 3.9.18 + Pytorch 2.1.0+cu121
git clone https://github.com/yunuoch/Atlas3D.git
cd Atlas3D
conda create -n atlas3d python=3.9
conda activate atlas3d
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
# Install warp (please note that the official release of warp is not supported yet)
pip install dependency/warp/
```

## Train on pretrained coarse stage
To use pretrained coarse stage results, please download ([link](https://drive.google.com/drive/folders/1-EMI-16smTvKgs9Ru2c6tZXiPh_9W27W?usp=sharing)), uncompress and place them in ./outputs/magic3d-coarse-if/

You can run with the corresponding config file as follows:
```shell
python launch.py --config configs/egg.yaml --train --gpu 0
```


## Train from scratch
```shell
# Run Magic3D as baseline coarse stage model
python launch.py --config configs/magic3d-coarse-if.yaml --train --gpu 0 system.prompt_processor.prompt="your prompt"

# Run refine stage with Atlas3D
python launch.py --config configs/atlas3d-refine-sd.yaml --train --gpu 0 system.prompt_processor.prompt="your prompt" system.geometry_convert_from=path/to/coarse/stage/trial/dir/ckpts/last.ckpt
```

## Acknowledgement
This code is built on [threestudio-project](https://github.com/threestudio-project/threestudio). We thank the maintainers for their contributions to the community.


## Citation
```
@article{chen2024atlas3d,
  title={Atlas3D: Physically Constrained Self-Supporting Text-to-3D for Simulation and Fabrication},
  author={Chen, Yunuo and Xie, Tianyi and Zong, Zeshun and Li, Xuan and Gao, Feng and Yang, Yin and Wu, Ying Nian and Jiang, Chenfanfu},
  journal={arXiv preprint arXiv:2405.18515},
  year={2024}
}
```