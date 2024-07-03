# Code For CHAIN+FastGANDBig

## 1. Download inception model and datasets
```
mkdir -p data/inception_model
mkdir -p data/torch_real_moments
```

### Download [Inception model](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) provided by [StyleGAN2-ADA-Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) to path
```
data/inception_model/inception-2015-12-05.pt
```
The inception model is converted from the [Tensorflow weights](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz) to Pytorch.

### Download datasets
Download [data](https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view) provided by [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch?tab=readme-ov-file) and extract it into 
```
data/[yourdataset]
```
## 2. prepare moments and train GANs (using the shells dataset as an example)

### Prepare the moments for calculating FID
```
python3 calculate_moments.py --data_path data/shells/img --moments_path data/torch_real_moments/shells_moments.npz
```

### Run the code to train GANs
```
python3 train.py \
--path data/shells/img --dataset_name shells --moments_path data/torch_real_moments/shells_moments.npz \
--chain_type chain --chain_blocks 12345 --tau 0.5 --lbd 20 --lbd_p0 0.1
```

## 3. Notes
1. For datasets with very few images, like Shells, Skulls, and AnimeFace, it is recommended to use ```--lbd_p0 0.1```. For datasets with more images, like Pokemon and ArtPainting, use ```--lbd_p0=0``` and more training iterations for convergence and better performance.
```
python3 train.py \
--path [datapath] --dataset_name [name] --moments_path [momentspath] \
--chain_type chain --chain_blocks 12345 --tau 0.5 --lbd 20 --iter 200000
```
2. To use the batch version of chain, change ```--chain_type chain_batch```. 
3. When working with new datasets, adjust ```--chain_blocks``` to settings like ```2345``` or ```1234``` and vary ```--tau``` for better performance. It is recommanded to set ```--lbd``` to  ```20```.
4. We tested the code on PyTorch 2.2.2, but it should also work on other versions if the required modules are installed.

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
