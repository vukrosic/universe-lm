import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, dataset_name, split="train", image_size=128):
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)

def get_vqvae_dataloader(dataset_name="lambdalabs/pokemon-blip-captions", batch_size=32, image_size=128):
    dataset = ImageDataset(dataset_name, split="train", image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader
