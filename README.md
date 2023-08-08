# myESANet

This repository contains code which is a paper implementation of "Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis" ([IEEE Xplore](https://ieeexplore.ieee.org/document/9561675),  [arXiv](https://arxiv.org/pdf/2011.06961.pdf)).

Based on the paper, I carefully implemented the network architecture on [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and [SUNRGB-D](https://rgbd.cs.princeton.edu/) and perform inference and test performance on edge devices(NVIDIA Jeton Nano).

This repository contains the code for training, evaluating, inference and deploying ESANet. As for deploying ESANet, I decide to deploy model in edge device by TensorRT and ONNX, as well as for measuring the inference time.

## Setup

## Content

## Training

## Evaluation

## Inference

## License and Citations

The source code is published under MIT license, see [license file](LICENSE) for details. 

The network architecture, is based on ESANet proposed by Seichter in 2021, the paper is as following:

>Seichter, D., Köhler, M., Lewandowski, B., Wengefeld T., Gross, H.-M.
*Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis*
in IEEE International Conference on Robotics and Automation (ICRA), pp. 13525-13531, 2021.

```bibtex
@inproceedings{esanet2021icra,
  title={Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis},
  author={Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021},
  volume={},
  number={},
  pages={13525-13531}
}

@article{esanet2020arXiv,
  title={Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis},
  author={Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael},
  journal={arXiv preprint arXiv:2011.06961},
  year={2020}
}
```
