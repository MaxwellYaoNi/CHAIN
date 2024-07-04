import argparse, os

import torch
from torchvision import transforms
from PIL import Image
from operation import EasyDict, MinSizeCenterCrop
from torch_inception_metrics import ImageWrapper,TorchInceptionMetrics


def read_all_images(data_path, resolution, center_crop):
    frame = []
    img_names = os.listdir(data_path)
    img_names.sort()
    for i in range(len(img_names)):
        image_path = os.path.join(data_path, img_names[i])
        if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg' or image_path[-4:] == '.tif':
            frame.append(image_path)
        else:
            print(image_path)
    transform_list = []
    if center_crop: transform_list.append(MinSizeCenterCrop())
    transform_list += [
        transforms.Resize((int(resolution), int(resolution))),
        transforms.PILToTensor()
    ]
    transform = transforms.Compose(transform_list)
    images = []
    for image_path in frame:
        img = Image.open(image_path).convert('RGB')
        img = transform(img)
        images.append(img)
    return torch.stack(images)


if __name__ == "__main__":
    """
    example:
    python3 calculate_moments.py --data_path ../data/few-shot-images/shells/img --moments_path ../data/torch_real_moments/shells_moments.npz 
    python3 calculate_moments.py --data_path ../data/few-shot-images/skulls/img --moments_path ../data/torch_real_moments/skulls_moments.npz
    python3 calculate_moments.py --data_path ../data/few-shot-images/anime-face/img --moments_path ../data/torch_real_moments/anime-face_moments.npz
    python3 calculate_moments.py --data_path ../data/few-shot-images/art-painting/img --moments_path ../data/torch_real_moments/art-painting_moments.npz
    python3 calculate_moments.py --data_path ../data/few-shot-images/pokemon/img --moments_path ../data/torch_real_moments/pokemon_moments.npz
    python3 calculate_moments.py --data_path ../data/few-shot-images/BreCaHAD/images --moments_path ../data/torch_real_moments/BreCaHAD_moments.npz
    python3 calculate_moments.py --data_path ../data/few-shot-images/metface/images --moments_path ../data/torch_real_moments/metface_moments.npz
    python3 calculate_moments.py --data_path ../data/few-shot-images/messidorset1/img --moments_path ../data/torch_real_moments/messidorset1_moments.npz --center_crop 1
    
    you may need a machine with at least 64GB CPU memory, otherwise you need to modify the code
    """
    parser = argparse.ArgumentParser(description='calculate fid moments for datasets')
    parser.add_argument('--inception_path', type=str, default='../data/inception_model')
    parser.add_argument('--moments_path',   type=str, default='../data/torch_real_moments/shells_moments.npz')
    parser.add_argument('--data_path',      type=str, default='../data/few-shot-images/shells/img')
    parser.add_argument('--resolution',     type=int, default=1024)
    parser.add_argument('--center_crop',    type=int, default=0)
    config = EasyDict(vars(parser.parse_args()))
    images = read_all_images(config.data_path, config.resolution, config.center_crop)
    images = images.int()
    image_wrapper = ImageWrapper(images)
    num_gpus = 1
    metrics_calculator = TorchInceptionMetrics(config.inception_path, batch_size=100, num_gpus=num_gpus, torch_for_fid=False, eps=1e-6)
    metrics_calculator.load_train_fid_moments(config.moments_path, image_wrapper, verbose=True)
    _, _, FID = metrics_calculator.get_IS_tFID(image_wrapper, verbose=True)
    # images = images + (torch.rand(*images.shape)*5).int()
    # image_wrapper.set_images_torch(images, None)
    print('FID:%f' % FID)

