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


# Loss Functions
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        ssim_loss = 1 - pytorch_ssim(img1, img2, data_range=1, size_average=True)
        return ssim_loss

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.vgg = nn.Sequential(*list(vgg)[:36]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = self.criterion(x_vgg, y_vgg)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.ssim_loss = SSIMLoss()
        self.vgg_loss = VGGLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        loss = self.mse_loss(output, target) + self.ssim_loss(output, target) + self.vgg_loss(output, target)
        return loss


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import pytorch_msssim
from tqdm import tqdm  # Progress bar for caching

class StandardDataset(Dataset):
    def __init__(self, raw_dir, reference_dir, transform=None):
        self.raw_dir = raw_dir
        self.reference_dir = reference_dir
        self.transform = transform

        self.raw_images = sorted(os.listdir(self.raw_dir))
        self.reference_images = sorted(os.listdir(self.reference_dir))

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, idx):
        raw_image_path = os.path.join(self.raw_dir, self.raw_images[idx])
        reference_image_path = os.path.join(self.reference_dir, self.reference_images[idx])

        raw_image = Image.open(raw_image_path).convert("RGB")
        reference_image = Image.open(reference_image_path).convert("RGB")

        if self.transform:
            raw_image = self.transform(raw_image)
            reference_image = self.transform(reference_image)

        return raw_image, reference_image

# Ensure calculations like PSNR are performed on the GPU
def calculate_psnr(outputs, targets):
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    psnrs = [psnr(outputs_np[i].transpose(1, 2, 0), targets_np[i].transpose(1, 2, 0), data_range=1.0)
             for i in range(outputs.size(0))]
    return np.mean(psnrs)

# Calculate Mean Square Error (MSE) on GPU
def calculate_mse(outputs, targets):
    mse_loss = nn.MSELoss().to(outputs.device)
    mse_value = mse_loss(outputs, targets)
    return mse_value.item()

def train_model(resume_training=True, checkpoint_path='/home4/qaiser.khan/UnderWaterImgEnh/Dataset2_model_state_dict.pth'):
    # Hyperparameters
    learning_rate = 0.0002
    batch_size = 4
    num_epochs = 100
    start_epoch = 10  # Will update if resuming training

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, Loss, Optimizer
    model = CustomModel().to(device)
    criterion = CombinedLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Resume training from checkpoint if provided
    if resume_training:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print(f"Resuming training from saved model weights...")

    # Dataset and DataLoader
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_dataset = StandardDataset(
        raw_dir='/home4/qaiser.khan/UnderWaterImgEnh/ImgEnh-2/Train/Raw',
        reference_dir='/home4/qaiser.khan/UnderWaterImgEnh/ImgEnh-2/Train/Reference',
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # For plotting
    epoch_losses = []
    epoch_psnrs = []
    epoch_ssims = []
    epoch_mses = []

    best_psnr = float('-inf')
    best_mse = float('inf')
    best_epoch = -1

    print(f"Number of images in the dataset: {len(train_dataset)}")
    print("Training started...")

    total_start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_mse = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_psnr += calculate_psnr(outputs, targets)
            epoch_ssim += pytorch_msssim.ssim(outputs, targets).item()
            epoch_mse += calculate_mse(outputs, targets)

        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = epoch_psnr / len(train_loader)
        avg_ssim = epoch_ssim / len(train_loader)
        avg_mse = epoch_mse / len(train_loader)

        epoch_losses.append(avg_loss)
        epoch_psnrs.append(avg_psnr)
        epoch_ssims.append(avg_ssim)
        epoch_mses.append(avg_mse)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_duration:.2f}s, '
              f'Average Loss: {avg_loss:.4f}, Average PSNR: {avg_psnr:.4f}, '
              f'Average SSIM: {avg_ssim:.4f}, Average MSE: {avg_mse:.4f}')

        # Save best weights every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save(model.state_dict(), f'best_model_weights_epoch_{epoch+1}.pth')
            print(f'Best model weights saved at epoch {epoch + 1}')

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f'Total Training Time: {total_training_time:.2f}s')

    # Save the final model
    torch.save(model, 'final_model.pt')
    print('Entire model saved!')
    torch.save(model.state_dict(), 'final_model_state_dict.pth')
    print('Model state dict saved!')

    # Save enhanced images
    os.makedirs('enhanced_images', exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            outputs = model(images)
            for j in range(images.size(0)):
                save_image(outputs[j].cpu(), f'enhanced_images/enhanced_image_{i}_{j}.png')
    print('All enhanced images saved!')

    # Plot training metrics
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.plot(range(1, num_epochs + 1), epoch_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')

    plt.subplot(1, 4, 2)
    plt.plot(range(1, num_epochs + 1), epoch_psnrs, label='PSNR', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR over Epochs')

    plt.subplot(1, 4, 3)
    plt.plot(range(1, num_epochs + 1), epoch_ssims, label='SSIM', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Epochs')

    plt.subplot(1, 4, 4)
    plt.plot(range(1, num_epochs + 1), epoch_mses, label='MSE', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE over Epochs')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

# To resume training
train_model(resume_training=True)

# To start training from scratch
#train_model()
