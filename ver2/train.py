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
        
        self.total_samples = sum([info['h_blocks'] * info['w_blocks'] for info in self.sample_info])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        for info in self.sample_info:
            num_blocks = info['h_blocks'] * info['w_blocks']
            if idx < num_blocks:
                h_idx = idx // info['w_blocks']
                w_idx = idx % info['w_blocks']
                
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

            if step % 100 == 0:
                # 计算 bicubic 放大的损失
                bicubic_outputs = F.interpolate(lr_blocks, scale_factor=2, mode='bicubic', align_corners=False).to(device)
                bicubic_loss = F.l1_loss(bicubic_outputs, hr_blocks)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Step [{step+1}/{len(train_loader)}] "
                      f"Model Loss: {loss.item():.4f} "
                      f"bicubic Loss: {bicubic_loss.item():.4f}"
                )

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