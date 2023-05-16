import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler, Subset
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import BatchSampler, RandomSampler


######### preprocess images, convert to tensor ############

# the training transforms
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

#################### Task Splits #########################
#class_0 is the same as task_0 just fyi
'''Train data'''
trainclass_0 = 'data/old_datasets/conf_data_new/train/class_0'
trainclass_1 = 'data/old_datasets/conf_data_new/train/class_1'
trainclass_2 = 'data/old_datasets/conf_data_new/train/class_2'

'''Validity data'''
valclass_0 = 'data/old_datasets/conf_data_new/val/class_0'
valclass_1 = 'data/old_datasets/conf_data_new/val/class_1'
valclass_2 = 'data/old_datasets/conf_data_new/val/class_2'

'''Load the Image folders for DataLoader'''
traindataset0 = datasets.ImageFolder(root=trainclass_0, transform=transformer)
valdataset0 = datasets.ImageFolder(root=valclass_0, transform=transformer)

traindataset1 = datasets.ImageFolder(root=trainclass_1, transform=transformer)
valdataset1 = datasets.ImageFolder(root=valclass_1, transform=transformer)

traindataset2 = datasets.ImageFolder(root=trainclass_2, transform=transformer)
valdataset2 = datasets.ImageFolder(root=valclass_2, transform=transformer)
''''''

