import torch.nn as nn
from torch.nn.utils import spectral_norm
from config import nc, ndf

# ### ORIGINAL D IMPLEMENTATION
# class Discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is ``(nc) x 64 x 64``
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf) x 32 x 32``
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*2) x 16 x 16``
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*4) x 8 x 8``
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*8) x 4 x 4``
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         return self.main(input)

# ### Apply Spectral Norm and Dropout on the OG D, retrain D to see if the result is better or not
# class Discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         p_dropout = 0.3
#         self.main = nn.Sequential(
#             spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)), # Apply Spectral norm here instead of BatchNorm
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(p_dropout),
#             spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(p_dropout),
#             spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(p_dropout),
#             spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(p_dropout),
#             spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         return self.main(input)

## 128px DCGAN D implementation, Apply Spectral Norm and Dropout
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        p_dropout = 0.3
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p_dropout),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p_dropout),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p_dropout),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p_dropout),
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p_dropout),
            spectral_norm(nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)