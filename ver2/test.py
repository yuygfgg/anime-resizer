import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from network import UpsampleResNet

def pad_image(img, block_size=32, overlap=16):
    """
    Pad the image so that its dimensions accommodate overlapping blocks.
    """
    c, h, w = img.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    pad_h += overlap  # Additional padding for overlap
    pad_w += overlap
    padding = (0, pad_w, 0, pad_h)
    img_padded = F.pad(img, padding, mode='reflect')
    return img_padded, pad_h, pad_w

def split_image(img, block_size=32, overlap=16):
    """
    Split the image into overlapping blocks of size (block_size x block_size).
    The window moves with a stride of (block_size - overlap).
    """
    c, h, w = img.shape
    stride = block_size - overlap
    blocks = []
    positions = []

    for i in range(0, h - overlap, stride):
        for j in range(0, w - overlap, stride):
            block = img[:, i:i+block_size, j:j+block_size]
            # Handle edge blocks
            if block.shape[1] < block_size:
                block = F.pad(block, (0, 0, 0, block_size - block.shape[1]), mode='reflect')
            if block.shape[2] < block_size:
                block = F.pad(block, (0, block_size - block.shape[2], 0, 0), mode='reflect')
            blocks.append(block)
            positions.append((i, j))
    
    blocks = torch.stack(blocks)
    return blocks, positions

def merge_blocks(blocks, positions, original_shape, scale_factor=2, block_size=32, overlap=16):
    """
    Merge the upscaled blocks into a single image using averaging in the overlapping regions.
    """
    c, h, w = original_shape
    scaled_h, scaled_w = h * scale_factor, w * scale_factor
    output = torch.zeros((c, scaled_h, scaled_w), device=blocks.device)
    weight = torch.zeros((c, scaled_h, scaled_w), device=blocks.device)

    block_size_scaled = block_size * scale_factor
    overlap_scaled = overlap * scale_factor
    stride_scaled = (block_size - overlap) * scale_factor

    for block, (i, j) in zip(blocks, positions):
        scaled_i = i * scale_factor
        scaled_j = j * scale_factor

        # Determine the actual block size, especially for edge blocks
        actual_block_h = min(block_size_scaled, scaled_h - scaled_i)  # Avoid going out of bounds in height
        actual_block_w = min(block_size_scaled, scaled_w - scaled_j)  # Avoid going out of bounds in width

        # Insert the block into the correct position in the output image
        output[:, scaled_i:scaled_i+actual_block_h, scaled_j:scaled_j+actual_block_w] += block[:, :actual_block_h, :actual_block_w]
        weight[:, scaled_i:scaled_i+actual_block_h, scaled_j:scaled_j+actual_block_w] += 1.0

    # Avoid division by zero
    weight[weight == 0] = 1.0
    output /= weight

    return output

def inference(model, image_path, output_path, batch_size=4, block_size=32, scale_factor=2, num_shift=1):
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0

    original_h, original_w = img_tensor.shape[1], img_tensor.shape[2]
    img_padded, pad_h, pad_w = pad_image(img_tensor, block_size)

    accumulated = None
    weight_sum = None

    # 多次平移（shift）操作
    for shift in range(num_shift):
        # 对图像进行平移处理（如果有平移操作）

        # 分割图像
        blocks, positions = split_image(img_padded, block_size)

        model.eval()
        device = next(model.parameters()).device

        upscaled_blocks = []

        with torch.no_grad():
            for i in tqdm(range(0, len(blocks), batch_size), desc=f"Processing blocks (shift {shift+1}/{num_shift})"):
                batch_blocks = blocks[i:i+batch_size].to(device)
                batch_upscaled = model(batch_blocks)
                batch_upscaled = torch.clamp(batch_upscaled, 0, 1)
                upscaled_blocks.append(batch_upscaled.cpu())

        upscaled_blocks = torch.cat(upscaled_blocks, dim=0)

        # 合并放大后的块
        output = merge_blocks(upscaled_blocks, positions, img_tensor.shape, scale_factor)

        # 裁剪到原始图像的2倍尺寸
        desired_h = original_h * scale_factor
        desired_w = original_w * scale_factor
        output = output[:, :desired_h, :desired_w]

        # 如果累计的张量为空，初始化它
        if accumulated is None:
            accumulated = torch.zeros_like(output)
            weight_sum = torch.zeros_like(output)

        # 确保累加时张量尺寸一致
        if accumulated.shape != output.shape:
            min_h = min(accumulated.shape[1], output.shape[1])
            min_w = min(accumulated.shape[2], output.shape[2])
            accumulated = accumulated[:, :min_h, :min_w]
            output = output[:, :min_h, :min_w]
            weight_sum = weight_sum[:, :min_h, :min_w]

        # 累加输出
        accumulated += output
        weight_sum += 1  # 每次叠加时增加权重计数

    # 计算平均值
    output = accumulated / weight_sum
    output = torch.clamp(output, 0, 1)

    # 保存最终输出图像
    output_img = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(output_img).save(f"{output_path}_model.png")
    print(f"Results saved to {output_path}_model.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Upscaling with UpsampleResNet using Multi-sampling Averaging")
    parser.add_argument('--input', type=str, required=True, help="Path to the input image file")
    parser.add_argument('--output', type=str, required=True, help="Path for saving the output image file")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for processing image blocks")
    parser.add_argument('--shift_times', type=int, default=4, help="Number of shifts for multi-sampling")
    args = parser.parse_args()

    model = UpsampleResNet()
    model.load_state_dict(torch.load('ver2.pth', map_location='cpu'))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    inference(model, args.input, args.output, batch_size=args.batch_size, num_shift=args.shift_times)