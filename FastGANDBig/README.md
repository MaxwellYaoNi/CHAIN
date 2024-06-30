# Code For CHAIN+FastGANDBig

### Download [Inception model](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) to path
```
data/inception_model
```
The inception model is converted from Tensorflow to Pytorch by work [StyleGAN2-ADA-Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).

### Preparing your Data
 you can download [data](https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view) provided by [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch?tab=readme-ov-file) and extract it into 
```
data/[yourdataset]
```

### Preparing the moments used for calculating FID
```
python3 calculate_moments.py --data_path data/shells/img --moments_path data/torch_real_moments/shells_moments.npz
```

### Runing the code
```
python3 train.py \
--path data/shells/img --dataset_name shells --moments_path data/torch_real_moments/shells_moments.npz \
--chain --chain_blocks 12345 --tau 0.5 --lbd 20
```

### some experience that could be used for better performance. 
1. You can adjust --chain_blocks to be like 123456, 2345, 23456, 12345, large data requires less blocks
2. 

# Citations
If you find this useful, please cite the paper!

```
@InProceedings{Ni_2024_CVPR,
    author    = {Ni, Yao and Koniusz, Piotr},
    title     = {CHAIN: Enhancing Generalization in Data-Efficient GANs via lipsCHitz continuity constrAIned Normalization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {6763-6774}
}
```

```
@inproceedings{liu2020towards,
  title={Towards faster and stabilized gan training for high-fidelity few-shot image synthesis},
  author={Liu, Bingchen and Zhu, Yizhe and Song, Kunpeng and Elgammal, Ahmed},
  booktitle={International conference on learning representations},
  year={2020}
}
```

```
@article{karras2020training,
  title={Training generative adversarial networks with limited data},
  author={Karras, Tero and Aittala, Miika and Hellsten, Janne and Laine, Samuli and Lehtinen, Jaakko and Aila, Timo},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={12104--12114},
  year={2020}
}
```
