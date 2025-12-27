import numpy as np
import torchvision
from torchvision import transforms
import time
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from glob import glob
from PIL import Image
from os.path import join
from scipy import ndimage
from scipy.ndimage import gaussian_filter  # **Import gaussian_filter**
import math
from torch.utils.data import Dataset

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.models as models
from pytorch_msssim import ssim as pytorch_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

class Inc(nn.Module):
    def __init__(self,in_channels,filters):
        super(Inc, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1),dilation=1,padding=(1-1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(3, 3), stride=(1, 1),dilation=1,padding=(3-1) // 2),
            nn.LeakyReLU(),
            )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1),dilation=1,padding=(1-1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(5, 5), stride=(1, 1),dilation=1,padding=(5-1) // 2),
            nn.LeakyReLU(),
            )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1),dilation=1),
            nn.LeakyReLU(),

        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1),dilation=1),
            nn.LeakyReLU(),
        )
    def forward(self,input):
        o1 = self.branch1(input)
        o2 = self.branch2(input)
        o3 = self.branch3(input)
        o4 = self.branch4(input)
        return torch.cat([o1,o2,o3,o4],dim=1)

def swish(x):
    return x * x.sigmoid()

def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6

def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)

class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)

def _make_divisible(v, divisor=8, min_value=None):  ## 将通道数变成8的整数倍
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0,),
                nn.ReLU(),
                nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
                HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSConvBlock, self).__init__()
        self.DW = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, groups=in_channels, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(in_channels)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.BN2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        a = self.HS(self.BN1(self.DW(x)))
        a = self.HS(self.BN2(self.PW(a)))
        return a

class ConvBlock1(nn.Module):
    def __init__(self):
        super(ConvBlock1, self).__init__()
        self.DW = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, groups=16, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(16)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock2(nn.Module):
    def __init__(self):
        super(ConvBlock2, self).__init__()
        self.DW = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, groups=32, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(32)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(64)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock3(nn.Module):
    def __init__(self):
        super(ConvBlock3, self).__init__()
        self.DW = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, groups=64, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(64)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock4(nn.Module):
    def __init__(self):
        super(ConvBlock4, self).__init__()
        self.DW = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, groups=80, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(80)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=80, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)
        self.SE = SELayer(80, 80)

    def forward(self, x):

        a = self.HS(self.BN(self.DW(x)))
        a = self.SE(a)
        a = self.HS(self.BNN(self.PW(a)))
        return a

class Mynet(nn.Module):
    def __init__(self, num_layers=3):
        super(Mynet, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)  ## 第一层卷积
        self.output = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = ConvBlock1()
        self.block2 = ConvBlock2()
        self.block3 = ConvBlock3()
        self.block4 = ConvBlock4()

    def forward(self, x):
        x = self.input(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        # x2 = torch.cat((x, x2), 1)
        x3 = self.block3(x2)
        x3 = torch.cat((x, x1, x3), 1)
        x4 = self.block4(x3)
        out = self.output(x4)
        return out

from torchvision.models import VGG19_Weights
# Define the complete model with additional DS Conv blocks and ConvBlock4
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.inception_block_r = Inc(in_channels=1, filters=64)
        self.inception_block_g = Inc(in_channels=1, filters=64)
        self.inception_block_b = Inc(in_channels=1, filters=64)
        self.se_layer_r = SELayer(inp=256, oup=256)
        self.se_layer_g = SELayer(inp=256, oup=256)
        self.se_layer_b = SELayer(inp=256, oup=256)
        self.ds_conv1 = DSConvBlock(in_channels=768, out_channels=256)
        self.ds_conv2 = DSConvBlock(in_channels=256, out_channels=128)
        self.ds_conv3 = DSConvBlock(in_channels=128, out_channels=16)
        self.ds_conv4 = DSConvBlock(in_channels=16, out_channels=32)
        self.ds_conv5 = DSConvBlock(in_channels=32, out_channels=64)
        self.ds_conv6 = DSConvBlock(in_channels=64, out_channels=32)
        self.conv_block4 = ConvBlock4()
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Split the input into R, G, B channels
        r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]

        # Process each channel independently through Inception and SE layers
        r = self.se_layer_r(self.inception_block_r(r))
        g = self.se_layer_g(self.inception_block_g(g))
        b = self.se_layer_b(self.inception_block_b(b))

        # Concatenate the outputs along the channel dimension (dim=1)
        x = torch.cat([r, g, b], dim=1)  # Shape: (1, 768, 256, 256)

        # Pass through the initial depthwise separable convolution blocks
        x = self.ds_conv1(x)  # Output shape: (1, 256, 256, 256)
        x = self.ds_conv2(x)  # Output shape: (1, 128, 256, 256)
        x = self.ds_conv3(x)  # Output shape: (1, 16, 256, 256)

        # Apply additional DS Conv blocks
        x1 = self.ds_conv4(x)  # Output shape: (1, 32, 256, 256)
        x2 = self.ds_conv5(x1)  # Output shape: (1, 64, 256, 256)
        x3 = self.ds_conv6(x2)  # Output shape: (1, 32, 256, 256)

        # Concatenate all outputs along the channel dimension (dim=1)
        x = torch.cat([x, x1, x2, x3], dim=1)  # Shape: (1, 16 + 32 + 64 + 32, 256, 256) = (1, 144, 256, 256)
        x = x[:,0:80,:,:]  # Output shape: (1, 80, 256, 256)
        # Adjust channels before passing through ConvBlock4
        x = self.conv_block4(x)  # Output shape: (1, 32, 256, 256)
        # Apply final 1x1 Conv with sigmoid
        x = self.final_conv(x)  # Output shape: (1, 3, 256, 256)
        x = self.sigmoid(x)  # Output shape: (1, 3, 256, 256)
        return x

class UIEDataset(Dataset):
    def __init__(self, raw_dir, reference_dir, transform=None):
        self.raw_dir = raw_dir
        self.reference_dir = reference_dir
        self.transform = transform
        self.image_names = [img for img in os.listdir(raw_dir) if img in os.listdir(reference_dir)]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        raw_image_path = os.path.join(self.raw_dir, self.image_names[idx])
        reference_image_path = os.path.join(self.reference_dir, self.image_names[idx])

        raw_image = Image.open(raw_image_path).convert("RGB")
        reference_image = Image.open(reference_image_path).convert("RGB")

        if self.transform:
            raw_image = self.transform(raw_image)
            reference_image = self.transform(reference_image)

        return raw_image, reference_image

# SSIM and PSNR Calculation
def getSSIM(X, Y):
    assert (X.shape == Y.shape), "Image patches provided have different dimensions"
    nch = 1 if X.ndim == 2 else X.shape[-1]
    mssim = []
    for ch in range(nch):
        Xc, Yc = X[..., ch].astype(np.float64), Y[..., ch].astype(np.float64)
        mssim.append(compute_ssim(Xc, Yc))  # **Updated call to compute_ssim**
    return np.mean(mssim)

def compute_ssim(X, Y):
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 5

    ux = gaussian_filter(X, sigma)  # **Update for SSIM calculation**
    uy = gaussian_filter(Y, sigma)  # **Update for SSIM calculation**

    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)
    vx = (uxx - ux * ux) * unbiased_norm
    vy = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim / D
    mssim = SSIM.mean()

    return mssim

def getPSNR(X, Y):
    target_data = np.array(X, dtype=np.float64)
    ref_data = np.array(Y, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    if rmse == 0: return 100
    else: return 20 * math.log10(255.0 / rmse)


# UIQM Calculation
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    weight = (1 / (K - T_a_L - T_a_R))
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    val = sum(x[s:e])
    return weight * val

def s_a(x, mu):
    return sum(math.pow((pixel - mu), 2) for pixel in x) / len(x)

def _uicm(x):
    R = x[:, :, 0].flatten()
    G = x[:, :, 1].flatten()
    B = x[:, :, 2].flatten()
    RG = R - G
    YB = ((R + G) / 2) - B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt(math.pow(mu_a_RG, 2) + math.pow(mu_a_YB, 2))
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)

def sobel(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag

def eme(x, window_size):
    k1 = int(x.shape[1] / window_size)
    k2 = int(x.shape[0] / window_size)
    w = 2. / (k1 * k2)
    x = x[:window_size * k2, :window_size * k1]
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            max_ = np.max(block)
            min_ = np.min(block)
            if min_ == 0.0 or max_ == 0.0: 
                val += 0
            else:
                val += math.log(max_ / min_)
    return w * val

def _uism(x):
    R = x[:, :, 0]
    G = x[:, :, 1]
    B = x[:, :, 2]
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)

def plip_g(x, mu=1026.0):
    return mu - x

def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k * ((g1 - g2) / (k - g2))

def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1 + g2 - ((g1 * g2) / gamma)

def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g / gamma)), c))

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))

def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta))

def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)

def _uiconm(x, window_size):
    plip_lambda = 1026.0
    plip_gamma = 1026.0
    plip_beta = 1.0
    plip_mu = 1026.0
    plip_k = 1026.0
    k1 = int(x.shape[1] / window_size)
    k2 = int(x.shape[0] / window_size)
    w = -1. / (k1 * k2)
    x = x[:window_size * k2, :window_size * k1]
    alpha = 1
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_ - min_
            bot = max_ + min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val += 0.0
            else:
                val += alpha * math.pow((top / bot), alpha) * math.log(top / bot)
    return w * val

def getUIQM(x):
    x = x.astype(np.float32)
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 10)
    return (c1 * uicm) + (c2 * uism) + (c3 * uiconm)

def test(config, test_dataloader, test_model):
    with torch.no_grad():
        for i, (input, target) in enumerate(test_dataloader):
            input = input.to(config['device'])
            output = test_model(input)
            
            for j in range(output.size(0)):
                # Assuming name[j] is part of the test_dataloader output and is a string filename
                name = test_dataloader.dataset.image_names[i * config['batch_size'] + j]  # Get the image name

                output_image = output[j].cpu().clamp(0, 1)  # Clamp the output to the range [0, 1]
                output_image = transforms.ToPILImage()(output_image)

                output_path = os.path.join(config['output_images_path'], name)  # Combine the output path and image name
                output_image.save(output_path)

    print("Testing completed.")



def setup(config):
    if torch.cuda.is_available():
        config['device'] = "cuda"
    else:
        config['device'] = "cpu"

    # Load the entire model
    model = torch.load(config['snapshot_path'], map_location=config['device'])
    model.to(config['device'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config['resize'], config['resize'])),
        transforms.ToTensor()
    ])
    
    # Ensure both raw and reference directories are passed
    test_dataset = UIEDataset(config['test_images_path'], config['label_images_path'], transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print("Test Dataset Reading Completed.")
    return test_dataloader, model



def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256, 256)):
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssims, psnrs = [], []
    
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        r_im = Image.open(gtr_path).resize(im_res)
        g_im = Image.open(gen_path).resize(im_res)
        
        # SSIM calculation
        ssim = getSSIM(np.array(r_im), np.array(g_im))
        ssims.append(ssim)
        
        # PSNR calculation
        r_im = r_im.convert("L")
        g_im = g_im.convert("L")
        psnr = getPSNR(np.array(r_im), np.array(g_im))
        psnrs.append(psnr)
        
    # Calculate averages and standard deviations
    avg_ssim = np.mean(ssims)
    avg_psnr = np.mean(psnrs)
    std_ssim = np.std(ssims)
    std_psnr = np.std(psnrs)
    
    return avg_ssim, avg_psnr, std_ssim, std_psnr

def measure_UIQMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    
    # Calculate averages and standard deviations
    avg_uiqm = np.mean(uqims)
    std_uiqm = np.std(uqims)
    
    return avg_uiqm, std_uiqm

# Kaggle specific setup
if __name__ == '__main__':
    config = {
        'snapshot_path': "/home4/qaiser.khan/UnderWaterImgEnh/EUVP_MODEL/final_model.pt",
        'test_images_path': "/home4/qaiser.khan/UnderWaterImgEnh/ImgEnh-2/Test/Raw",
        'output_images_path': "/home4/qaiser.khan/UnderWaterImgEnh/Gen-output-5",
        'batch_size': 1,
        'resize': 256,
        'calculate_metrics': True,
        'label_images_path': "/home4/qaiser.khan/UnderWaterImgEnh/ImgEnh-2/Test/Reference"
    }

    if not os.path.exists(config['output_images_path']):
        os.mkdir(config['output_images_path'])

    start_time = time.time()
    ds_test, model = setup(config)
    test(config, ds_test, model)
    print("Total testing time:", time.time() - start_time)

    # Calculate metrics if specified
    if config['calculate_metrics']:
        gen_uqims, _ = measure_UIQMs(config['output_images_path'])
        avg_uiqm, _ = measure_UIQMs(config['output_images_path'])

        avg_ssim, avg_psnr, _, _ = SSIMs_PSNRs(config['label_images_path'], config['output_images_path'])
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average UIQM: {avg_uiqm:.4f}")
        

      
