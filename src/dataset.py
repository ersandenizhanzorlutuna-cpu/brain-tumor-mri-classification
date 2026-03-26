
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms 

CLASS_NAMES=['glioma','meningioma','pituitary','notumor']

class BrainTumorDataset(Dataset):
  def __init__(self, root_dir, transform=None):

    self.root_dir=root_dir
    self.transform=transform
    self.samples=[]


    for labels, class_name in enumerate(CLASS_NAMES):
      class_dir=os.path.join(root_dir, class_name)
      for filename in os.listdir(class_dir):
        if filename.lower().endswith(('.jpg','.jpeg','.png')):
          path=os.path.join(class_dir,filename)
          self.samples.append((path,labels))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    path,labels=self.samples[idx]
    image=Image.open(path).convert('RGB')
    if self.transform:
      image=self.transform(image)
      return image, labels
       
      



