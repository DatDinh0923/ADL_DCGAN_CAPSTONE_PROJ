import torch
import torchvision.utils as vutils
import os
from generator import Generator
from config import nz, ngpu

# This file just to generate image for evaluating, don't need to read this

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WEIGHT_PATH = "/home/insomnia/1Code_Workspace/ADL/weights/netGenerator.pth"
OUTPUT_DIR = "./fake/fake_test_some_methods"
os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_IMAGES = 2000 
BATCH_SIZE = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load(WEIGHT_PATH))
netG.eval()

count = 0
with torch.no_grad():
    while count < NUM_IMAGES:
        noise = torch.randn(BATCH_SIZE, nz, 1, 1, device=device)
        fake_images = netG(noise).cpu()
        
        # Lưu TỪNG bức ảnh ra file
        for i in range(fake_images.size(0)):
            if count >= NUM_IMAGES:
                break
            vutils.save_image(fake_images[i], f"{OUTPUT_DIR}/img_{count}.png", normalize=True)
            count += 1
            