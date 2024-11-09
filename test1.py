import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

class UpsampleResNet(nn.Module):
    def __init__(self, in_channels=3, num_blocks=8, hidden_channels=96):
        super(UpsampleResNet, self).__init__()
        
        # 初始特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        )
        
        # 残差块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )
        
        # 上采样前的处理
        self.pre_upscale = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        
        # 上采样
        self.upscale = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(True)
        )
        
        # 最终处理
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 确保输出在 [0,1] 范围内
        )

    def forward(self, x):
        initial = self.initial_conv(x)
        res = self.res_blocks(initial)
        res = res + initial  # 全局残差连接
        up = self.pre_upscale(res)
        up = self.upscale(up)
        out = self.final(up)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + res

def pad_image(img, block_size=32):
    c, h, w = img.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    padding = (0, pad_w, 0, pad_h)
    return F.pad(img, padding, mode='reflect'), pad_h, pad_w

def split_image(img, block_size=32, overlap=None):
    if overlap is None:
        overlap = block_size // 2

    c, h, w = img.shape
    blocks = []
    positions = []
    
    stride = block_size - overlap
    
    # Offsets for different passes
    offsets = [(0, 0), (0, stride // 2), (stride // 2, 0), (stride // 2, stride // 2)]
    
    for offset_h, offset_w in offsets:
        h_blocks = (h + stride - 1) // stride
        w_blocks = (w + stride - 1) // stride
        
        for i in range(h_blocks):
            for j in range(w_blocks):
                start_h = min(i * stride + offset_h, h - block_size)
                start_w = min(j * stride + offset_w, w - block_size)
                
                if i == h_blocks - 1:
                    start_h = h - block_size
                if j == w_blocks - 1:
                    start_w = w - block_size
                
                block = img[:, start_h:start_h+block_size, start_w:start_w+block_size]
                blocks.append(block)
                positions.append((start_h, start_w))
    
    return torch.stack(blocks), positions, (h_blocks, w_blocks)

def merge_blocks(blocks, positions, original_shape, scale_factor=2, overlap=16):
    c, h, w = original_shape
    scaled_h, scaled_w = h * scale_factor, w * scale_factor
    output = torch.zeros((c, scaled_h, scaled_w), device=blocks[0].device)
    weight = torch.zeros((scaled_h, scaled_w), device=blocks[0].device)
    
    block_size = blocks[0].shape[-1]
    overlap_size = overlap * scale_factor
    
    mask = torch.ones((block_size, block_size), device=blocks[0].device)
    if overlap_size > 0:
        mask[:overlap_size, :] = torch.linspace(0, 1, overlap_size).view(-1, 1)
        mask[-overlap_size:, :] = torch.linspace(1, 0, overlap_size).view(-1, 1)
        mask[:, :overlap_size] *= torch.linspace(0, 1, overlap_size).view(1, -1)
        mask[:, -overlap_size:] *= torch.linspace(1, 0, overlap_size).view(1, -1)
    
    for block, (start_h, start_w) in zip(blocks, positions):
        start_h *= scale_factor
        start_w *= scale_factor
        
        end_h = min(start_h + block_size, scaled_h)
        end_w = min(start_w + block_size, scaled_w)
        block_part = block[:, :(end_h-start_h), :(end_w-start_w)]
        mask_part = mask[:(end_h-start_h), :(end_w-start_w)]
        
        output[:, start_h:end_h, start_w:end_w] += block_part * mask_part.unsqueeze(0)
        weight[start_h:end_h, start_w:end_w] += mask_part
    
    weight = weight.clamp(min=1e-8)
    output = output / weight.unsqueeze(0)
    
    return output

def inference(model, image_path, output_path, block_size=32, batch_size=4):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    
    img_padded, pad_h, pad_w = pad_image(img_tensor, block_size)
    original_shape = img_padded.shape
    
    blocks, positions, (h_blocks, w_blocks) = split_image(img_padded, block_size)
    
    model.eval()
    device = next(model.parameters()).device
    
    upscaled_blocks = []
    bicubic_blocks = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(blocks), batch_size), desc="Processing blocks"):
            batch_blocks = blocks[i:i+batch_size].to(device)
            batch_upscaled = model(batch_blocks)
            batch_upscaled = torch.clamp(batch_upscaled, 0, 1)
            batch_bicubic = F.interpolate(batch_blocks, 
                                          scale_factor=2, 
                                          mode='bicubic',
                                          align_corners=False)
            upscaled_blocks.extend(batch_upscaled.cpu())
            bicubic_blocks.extend(batch_bicubic.cpu())
    
    output = merge_blocks(upscaled_blocks, positions, original_shape)
    bicubic_output = merge_blocks(bicubic_blocks, positions, original_shape)
    
    if pad_h > 0 or pad_w > 0:
        h, w = original_shape[1:]
        output = output[:, :h*2-pad_h*2, :w*2-pad_w*2]
        bicubic_output = bicubic_output[:, :h*2-pad_h*2, :w*2-pad_w*2]
    
    output = torch.clamp(output, 0, 1)
    bicubic_output = torch.clamp(bicubic_output, 0, 1)
    
    output_img = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bicubic_img = (bicubic_output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    Image.fromarray(output_img).save(f"{output_path}_model.png")
    Image.fromarray(bicubic_img).save(f"{output_path}_bicubic.png")
    
    print(f"Results saved to {output_path}_model.png and {output_path}_bicubic.png")

if __name__ == "__main__":
    model = UpsampleResNet()
    model.load_state_dict(torch.load('ver1.pth'))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    input_path = "haruhi08_16360.png"
    output_path = "haruhi08_16360_ver1"
    
    inference(model, input_path, output_path, block_size=32, batch_size=4)