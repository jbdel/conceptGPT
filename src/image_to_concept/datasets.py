from torch.utils.data import Dataset
from torchvision.transforms import *
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate as pytorch_default_collate
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from image_to_concept.utils import create_one_hot_vector


class ImageDataset(Dataset):
    def __init__(self,
        annotations,
        image_dir='.',
        concept_type=None,
        split=None,
        filtered_concepts=None,
        size=224,
        **kwargs
    ):
        assert split in ["train", "validate", "test"]

        image_path = os.path.join(image_dir, f"{split}.image.tok")
        self.images = [line.strip().split(',')[0] for line in open(image_path).readlines()]
        self.labels = torch.tensor(
            [create_one_hot_vector(filtered_concepts, c) for c in annotations[split][concept_type]]).long()

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.images or [])

    def __getitem__(self, index):
        return {"images": self.transform(Image.open(self.images[index]).convert('RGB')),
                "labels": self.labels[index]}


def fetch_data(filtered_concepts, concept_type, annotations, image_dir):
  train_dataset = ImageDataset(
    split="train",
    image_dir=image_dir,
    annotations=annotations,
    concept_type=concept_type,
    filtered_concepts=filtered_concepts
  )
  train_sampler = BatchSampler(
    RandomSampler(train_dataset),
    batch_size=8,
    drop_last=False
  )

  train_dataloader = DataLoader(
    train_dataset,
    num_workers=2,
    batch_sampler=train_sampler,
    pin_memory=True
  )

  val_dataset = ImageDataset(
    split="validate",
    image_dir=image_dir,
    annotations=annotations,
    concept_type=concept_type,
    filtered_concepts=filtered_concepts
  )

  val_sampler = BatchSampler(
    SequentialSampler(val_dataset),
    batch_size=8,
    drop_last=False
  )

  val_dataloader = DataLoader(
    val_dataset,
    num_workers=2,
    batch_sampler=val_sampler,
    pin_memory=True
  )

  return train_dataloader, val_dataloader