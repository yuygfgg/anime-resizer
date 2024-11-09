import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
from network import UpsampleResNet

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class ImageBlockDataset(Dataset):
    def __init__(self, image_folder, block_size=32, scale_factor=2):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.npy')]
        self.block_size = block_size
        self.scale_factor = scale_factor
        self.sample_info = []
        
        for img_file in self.image_files:
            img_path = os.path.join(image_folder, img_file)
            img_hr = np.load(img_path)
            h_hr, w_hr = img_hr.shape[:2]
            
            h_blocks = (h_hr // 2 - block_size) // block_size + 1
            w_blocks = (w_hr // 2 - block_size) // block_size + 1
            
            self.sample_info.append({
                'img_file': img_file,
                'h_blocks': h_blocks,
                'w_blocks': w_blocks
            })
        
        # 计算整个数据集的总样本数
        self.total_samples = sum([info['h_blocks'] * info['w_blocks'] for info in self.sample_info])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 找到对应的图片和块
        for info in self.sample_info:
            num_blocks = info['h_blocks'] * info['w_blocks']
            if idx < num_blocks:
                h_idx = idx // info['w_blocks']
                w_idx = idx % info['w_blocks']
                
                # 延迟加载图片
                img_path = os.path.join(self.image_folder, info['img_file'])
                img_hr = np.load(img_path)
                img_hr = torch.from_numpy(img_hr).permute(2, 0, 1).float() / 255.0
                
                img_lr = F.interpolate(img_hr.unsqueeze(0), scale_factor=1/self.scale_factor, mode='bilinear', align_corners=False).squeeze(0)
                
                hr_block = img_hr[:, h_idx*self.block_size*2:(h_idx+1)*self.block_size*2, 
                                    w_idx*self.block_size*2:(w_idx+1)*self.block_size*2]
                lr_block = img_lr[:, h_idx*self.block_size:(h_idx+1)*self.block_size, 
                                    w_idx*self.block_size:(w_idx+1)*self.block_size]
                return lr_block, hr_block
            
            idx -= num_blocks
        
        raise IndexError("Index out of range")

def pad_image(img, block_size=32):
    c, h, w = img.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    padding = (0, pad_w, 0, pad_h)
    return F.pad(img, padding, mode='reflect'), pad_h, pad_w

def split_image(img, block_size=32, overlap=6):
    c, h, w = img.shape
    blocks = []
    positions = []
    
    stride = block_size - overlap
    
    h_blocks = (h + stride - 1) // stride
    w_blocks = (w + stride - 1) // stride
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            start_h = min(i * stride, h - block_size)
            start_w = min(j * stride, w - block_size)
            
            if i == h_blocks - 1:
                start_h = h - block_size
            if j == w_blocks - 1:
                start_w = w - block_size
            
            block = img[:, start_h:start_h+block_size, start_w:start_w+block_size]
            blocks.append(block)
            positions.append((start_h, start_w))
    
    return torch.stack(blocks), positions, (h_blocks, w_blocks)

def merge_blocks(blocks, positions, original_shape, scale_factor=2, overlap=6):
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

def test_and_save(model, test_img_tensor, epoch, step):
    model.eval()
    with torch.no_grad():
        img_padded, pad_h, pad_w = pad_image(test_img_tensor, block_size=32)
        original_shape = img_padded.shape
        blocks, positions, (h_blocks, w_blocks) = split_image(img_padded, block_size=32)
        
        upscaled_blocks = []
        bilinear_blocks = []
        
        for i in range(0, len(blocks), 4):
            batch_blocks = blocks[i:i+4].to(device)
            batch_upscaled = model(batch_blocks)
            batch_bilinear = F.interpolate(batch_blocks, scale_factor=2, mode='bilinear', align_corners=False)
            
            upscaled_blocks.extend(batch_upscaled.cpu())
            bilinear_blocks.extend(batch_bilinear.cpu())
        
        output = merge_blocks(upscaled_blocks, positions, original_shape)
        bilinear_output = merge_blocks(bilinear_blocks, positions, original_shape)
        
        if pad_h > 0 or pad_w > 0:
            h, w = original_shape[1:]
            output = output[:, :h*2-pad_h*2, :w*2-pad_w*2]
            bilinear_output = bilinear_output[:, :h*2-pad_h*2, :w*2-pad_w*2]
        
        output = torch.clamp(output, 0, 1)
        bilinear_output = torch.clamp(bilinear_output, 0, 1)
        
        output_img = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bilinear_img = (bilinear_output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        os.makedirs('test_results', exist_ok=True)
        Image.fromarray(output_img).save(f'test_results/epoch{epoch+1}_step{step+1}_model.png')
        Image.fromarray(bilinear_img).save(f'test_results/epoch{epoch+1}_step{step+1}_bilinear.png')
    
    model.train()

def train_model():
    train_dataset = ImageBlockDataset(image_folder="../data", 
                                    block_size=32, 
                                    scale_factor=2)
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=16,
                            shuffle=True, 
                            num_workers=12,
                            pin_memory=True)

    model = UpsampleResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 4
    best_loss = float('inf')
    
    test_img = Image.open("input.png").convert('RGB')
    test_img_np = np.array(test_img)
    test_img_tensor = torch.from_numpy(test_img_np).float().permute(2, 0, 1) / 255.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for step, (lr_blocks, hr_blocks) in enumerate(train_loader):
            lr_blocks = lr_blocks.to(device, non_blocking=True)
            hr_blocks = hr_blocks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(lr_blocks)
            loss = F.l1_loss(outputs, hr_blocks)

            bilinear_outputs = F.interpolate(lr_blocks, scale_factor=2, mode='bilinear', align_corners=False).to(device)
            bilinear_loss = F.l1_loss(bilinear_outputs, hr_blocks)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Step [{step+1}/{len(train_loader)}] "
                      f"Model Loss: {loss.item():.4f} "
                      f"bilinear Loss: {bilinear_loss.item():.4f}")
            
            if (step + 1) % 2000 == 0:
                test_and_save(model, test_img_tensor.to(device), epoch, step)

             # 在 1500 step 后减小学习率
            if (step + 1) == 1500 and epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print("Reduced learning rate by 10 times.")
                
            # 在 15000 step 后减小学习率
            if (step + 1) == 15000 and epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print("Reduced learning rate by 10 times.")
            
            # 在 35000 step 后减小学习率
            if (step + 1) == 35000 and epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print("Reduced learning rate by 2 times.")
            
            # 在 50000 step 后减小学习率
            if (step + 1) == 50000 and epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print("Reduced learning rate by 2 times.")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("New best model saved!")

    torch.save(model.state_dict(), 'final_model.pth')
    print("Training completed.")

if __name__ == '__main__':
    train_model()