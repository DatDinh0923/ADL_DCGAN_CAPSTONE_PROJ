import os
import torch
import random
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
from torch_fidelity import calculate_metrics

FAKE_DIR = "/home/insomnia/1Code_Workspace/ADL/fake/fake_test_some_methods"     
REAL_DIR = "/home/insomnia/1Code_Workspace/ADL/dataset/images/img_align_celeba" 
NUM_PAIRS = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_msssim_for_diversity(img_dir, num_pairs=1000):
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
    if len(img_files) < 2:
        raise ValueError("MS-SSIM metrics need atleast 2 images")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    total_msssim = 0.0
    
    for _ in range(num_pairs):
        img1_path, img2_path = random.sample(img_files, 2)
        img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
        img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)
        score = ms_ssim(img1, img2, data_range=1.0, size_average=True) # Calculate MSSSIM for this image pair
        total_msssim += score.item()

    avg_msssim = total_msssim / num_pairs
    return avg_msssim

def compute_fid_kid(fake_dir, real_dir):    
    metrics_dict = calculate_metrics(
        input1=fake_dir, 
        input2=real_dir, 
        cuda=(device.type == 'cuda'),
        isc=False,
        fid=True,   # Compute FID     
        kid=True,   # Compute KID
        verbose=False   
    )
    return metrics_dict['frechet_inception_distance'], metrics_dict['kernel_inception_distance_mean']

if __name__ == '__main__':    
    avg_msssim = compute_msssim_for_diversity(FAKE_DIR, NUM_PAIRS) # Calculate MSSSIM
    fid_score, kid_score = compute_fid_kid(FAKE_DIR, REAL_DIR) # Calculate KID, FID
    
    print("="*50)
    print(f"1. MS-SSIM : {avg_msssim:.4f}")
    print(f"2. FID Score : {fid_score:.2f}")
    print(f"3. KID Score : {kid_score:.5f}")
    print("="*50)

"""
NOTE:
MS-SSIM: should be around 0.2~0.3, so that the images generated are diverse, the lower the better, too high (e.g 0.6) signified that the model collapsed
FID: tell the realism and of the generated iamges, should be low, high mean images look fake
    < 10 -> very good
    10–50 -> decent
    > 100 -> poor
KID: just like FID, but more stable, unbias cuz FID is likely to be more bias with smaller database

Low FID but high MS-SSIM --> mode collapse cuz memorizing
Low MS-SSIM but high FID --> diverse in generated images but poor quality
"""