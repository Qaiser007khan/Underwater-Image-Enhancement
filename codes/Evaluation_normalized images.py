#Evaluation code from github with normalizing images 
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
from torchvision.utils import save_image

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load saved model
model = torch.load('/kaggle/input/data2-finalmodels/final_model.pt')
model.to(device)  # Ensure the model is on the correct device (GPU or CPU)
print("Full model loaded successfully!")

# Optional: Check some of the model's parameters to confirm loading
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # Print the first two values
    break  # Remove or modify this to print more layers or parameters

# Define the RawImageDataset
class RawImageDataset(Dataset):
    def __init__(self, raw_dir, transform=None):
        self.raw_dir = raw_dir
        self.transform = transform
        self.raw_image_paths = [os.path.join(raw_dir, img) for img in os.listdir(raw_dir)]

    def __len__(self):
        return len(self.raw_image_paths)

    def __getitem__(self, idx):
        raw_image_path = self.raw_image_paths[idx]
        raw_image = Image.open(raw_image_path).convert("RGB")
        
        if self.transform:
            raw_image = self.transform(raw_image)
        
        return raw_image, os.path.basename(raw_image_path)  # Return the image and its filename
   
    # Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Adjust the path for the raw test images
test_dataset = RawImageDataset('/kaggle/input/imgenh-2/dataset-2/Test/Raw', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Save all enhanced test images with the same names as input images
os.makedirs('enhanced_test_images', exist_ok=True)
model.eval()
with torch.no_grad():
    for i, (images, image_names) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        for j in range(images.size(0)):
            # Save the output image with the same name as the input image
            save_image(outputs[j].cpu(), os.path.join('enhanced_test_images', image_names[j]))

print('All enhanced test images saved!')

#EVALUATION METRICS CODE
#UIQM FILE CONTENT
import os
import cv2
import numpy as np
from skimage import data, color
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import math
from skimage.util import img_as_ubyte, img_as_float64
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from skimage.io import imread
import warnings
warnings.filterwarnings('ignore')
import skimage
from numpy import load
from numpy import expand_dims
import matplotlib
from matplotlib import pyplot
import sys
import PIL
from PIL import Image
import pandas as pd
import numpy as np
import scipy.misc
import imageio
import glob
import os
import cv2
!pip install sewar
import sewar
import math
from math import log2, log10
from scipy import ndimage
import skimage
from skimage import color
from skimage.metrics import structural_similarity
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag

def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    # weight
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:int(blocksize_y*k2), :int(blocksize_x*k1)]
    val = 0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            # bound checks, can't do log(0)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val

def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    # get eme for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def plip_g(x,mu=1026.0):
    return mu-x

def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k*((g1-g2)/(k-g2))

def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1+g2-((g1*g2)/(gamma))

def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g/gamma) ), c) )

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    #return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))

def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));

def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)

def _uiconm(x, window_size):
    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    # weight
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:int(blocksize_y*k2), :int(blocksize_x*k1)]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w*val

##########################################################################################

def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    #c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm


def getUCIQE(rgb_in):
    # calculate Chroma
    rgb_in = cv2.normalize(rgb_in, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    (l,a,b)=cv2.split(rgb_in)
    Chroma = np.sqrt(a*a + b*b)
    StdVarianceChroma = np.std(np.reshape(Chroma[:,:],(-1,1)))

    hsv = skimage.color.rgb2hsv(rgb_in)
    Saturation = hsv[:,:,2]
    MeanSaturation = np.mean(np.reshape(Saturation[:,:],(-1,1)))

    ContrastLuminance = max(np.reshape(l[:,:],(-1,1))) - min(np.reshape(l[:,:],(-1,1)))
    UCIQE = 0.4680 * StdVarianceChroma + 0.2745 * ContrastLuminance + 0.2576 * MeanSaturation
    return float(UCIQE)

def improve_contrast_image_using_clahe(bgr_image):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
import numpy as np
!pip install libsvm-official
from libsvm.svmutil import *
import matplotlib.pyplot as plt
import cv2
import os
import skimage
import imageio as iio

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

generated = []
gt = []

# SET THE TEST IMAGE PATH IN gt_addrr
gt_addrr = "/kaggle/input/imgenh-2/dataset-2/Test/Reference"

# SET THE ENHANCED TEST IMAGE PATH IN addrr
addrr = "/kaggle/working/enhanced_test_images"

# Ensure both lists of images have the same order by sorting the filenames
gt_filenames = sorted(os.listdir(gt_addrr))
generated_filenames = sorted(os.listdir(addrr))

# Load the images, resize them, and append them to the lists
for item in generated_filenames:
    if item.endswith(".jpg"):
        image_path = os.path.join(addrr, item)
        image = cv2.imread(image_path)
        if image is not None:  # Ensure the image was loaded
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32")
            image = cv2.resize(image, (256, 256))  # Resize to a common size
            generated.append(image)
        else:
            print(f"Warning: Could not load image {image_path}")

for item in gt_filenames:
    if item.endswith(".jpg"):
        image_path = os.path.join(gt_addrr, item)
        image = cv2.imread(image_path)
        if image is not None:  # Ensure the image was loaded
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32")
            image = cv2.resize(image, (256, 256))  # Resize to a common size
            gt.append(image)
        else:
            print(f"Warning: Could not load image {image_path}")

# Ensure both lists have the same length before proceeding
if len(generated) != len(gt):
    raise ValueError("The number of generated images and ground truth images do not match!")

# Initialize lists to store the metrics
SSIM_results = []
PSNR_results = []
UIQM = []
UCIQE = []
MSE = []

# Calculate metrics for each image pair
for i in range(len(generated)):
    print(f"Processing image pair {i+1}/{len(generated)}")
    
    # Normalize images
    norm_generated = NormalizeData(generated[i])
    norm_gt = NormalizeData(gt[i])
    
    # Calculate and store UIQM and UCIQE metrics
    UIQM.append(getUIQM(norm_generated))
    UCIQE.append(getUCIQE(norm_generated))
    
    # Calculate and store PSNR and SSIM with explicit win_size, channel_axis, and data_range
    PSNR_results.append(peak_signal_noise_ratio(norm_generated, norm_gt, data_range=1.0))
    SSIM_results.append(structural_similarity(norm_generated, norm_gt, win_size=7, channel_axis=-1, data_range=1.0))
    
    # Calculate and store MSE
    MSE.append(mean_squared_error(norm_generated, norm_gt))

# Print the average of the metrics
print(f"Average SSIM: {np.mean(SSIM_results):.4f}")
print(f"Average PSNR: {np.mean(PSNR_results):.4f} dB")
print(f"Average MSE: {np.mean(MSE):.4f}")
print(f"Average UIQM: {np.mean(UIQM):.4f}")
print(f"Average UCIQE: {np.mean(UCIQE):.4f}")
