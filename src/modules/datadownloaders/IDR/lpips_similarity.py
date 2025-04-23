'''Code for LPIPS similarity. Takes all images inside the folders that are inside the main folder.
This is class agnostic. Some modifications can be made to result into class specific metrics, but this doesnt concern us.'''

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from lpips import LPIPS
from tqdm import tqdm
import os
from PIL import Image

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

# Define the transformation
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

# Define the dataset
dataset = ImageFolderDataset(root='/projects/imagesets/imagenet/RawImages/valid', transform=transform)

# Create a data loader
dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)

# Initialize LPIPS
loss_fn = LPIPS(net='alex')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
loss_fn.to(device)

lpips_scores = [] # Store the LPIPS scores here
num_comparisons = 0

# For each batch of images
for batch_idx, images1 in tqdm(enumerate(dataloader), desc='Processing batches'):
    images1 = images1.to(device)

    # Select three channels randomly for images1 with shape (batch_size, num_channels, height, width)
    num_channels = images1.size(1) 
    if num_channels < 3:
        # Duplicate channels to have three
        images1 = images1.repeat(1, 3 // num_channels + 1, 1, 1)[:, :3, :, :]
    else:
        # Randomly select three channels
        channel_indices = random.sample(range(num_channels), 3)
        images1 = images1[:, channel_indices, :, :]

    # For each other batch of images starting from batch_idx
    for batch_idx2, images2 in enumerate(dataloader):
        images2 = images2.to(device)
        
        # Skip if comparing the same batch or different sizes
        if batch_idx == batch_idx2 or images1.size(0) != images2.size(0):
            continue
        
        # Select three channels for images2 with shape (batch_size, num_channels, height, width)
        num_channels = images2.size(1)
        if num_channels < 3:
            images2 = images2.repeat(1, 3 // num_channels + 1, 1, 1)[:, :3, :, :]
        else:
            channel_indices = random.sample(range(num_channels), 3)
            images2 = images2[:, channel_indices, :, :]
        
        # Calculate the LPIPS between the two batches of images
        lpips_values = loss_fn(images1, images2)

        # Store the LPIPS values
        lpips_scores.append(lpips_values.detach().cpu().numpy())

        # Count the number of comparisons
        num_comparisons += lpips_values.numel()

# Convert list of all LPIPS scores to PyTorch tensor
lpips_scores = torch.tensor(lpips_scores)

# Calculate and print the mean and standard deviation of LPIPS scores
mean_lpips = torch.mean(lpips_scores)
std_lpips = torch.std(lpips_scores)
print('Mean LPIPS score:', mean_lpips.item())
print('Standard deviation of LPIPS scores:', std_lpips.item())
print(num_comparisons)