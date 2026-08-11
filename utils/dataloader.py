import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import ToTensor
import random



    

class Normalize(object):
    def __init__(self,mean=0.5,std=0.5):
        self.mean =mean
        self.std = std
    
    def __call__ (self, input):
        input = (input -self.mean)/self.std
        
        return input

    
    
class PickelDatasetv2_new(Dataset):
    def __init__(self, root_dir, set_name, data_firstname ="FBP_",label_firstname ="MBIR_", data_dicname="fbp",label_dicname="mbir",patch_size = (128,128), z_slice = None, transform=None):
        # data_dir: 데이터 파일들이 있는 디렉토리
        self.data_root = os.path.join(root_dir,set_name)

        self.transform=transform
        self.subject_path_all = sorted([item.path for item in os.scandir(self.data_root)])
        self.label_firstname = label_firstname
        self.data_firstname = data_firstname
        self.patch_size = patch_size      
        self.data_dicname = data_dicname
        self.label_dicname = label_dicname
        self.label_pickle = ""
        self.to_tensor = ToTensor()        
        
        self.z_slice_all = [int(len(os.listdir(path))/2) for path in self.subject_path_all] # data와 label이 있으므로 2로 나누자
        self.len = sum(self.z_slice_all)
        
        self.z_slices = []
        if z_slice is None:
            # 슬라이스 갯수가 샘플 마다 다르다            
            for index, count in enumerate(self.z_slice_all):
                # 인덱스를 'count' 횟수만큼 반복하여 추가
                self.z_slices.extend([index] * count)

        self.z_slice = z_slice
    
    def __len__(self):
        
        #return len(self.subject_path_all) * self.z_slice
        return self.len        
    
    def __getitem__(self, idx):
        
        if self.z_slice is None:
            # 슬라이스 갯수가 샘플 마다 다르다
            subject_idx = self.z_slices[idx]
            slice_idx = idx % self.z_slice_all[subject_idx]  
            z_slice = self.z_slice_all[subject_idx]  
        else:
            subject_idx = idx // self.z_slice  # 672개의 슬라이스를 가지므로, 파일 인덱스를 구함
            slice_idx = idx % self.z_slice  # 파일 내에서의 슬라이스 인덱스를 구함
            z_slice = self.z_slice
        
        subject_path = self.subject_path_all[subject_idx]
        subject_name = subject_path.split("/")[-1]

        label_pickle_path = os.path.join(subject_path, f"{self.label_firstname}{subject_name}_{slice_idx:03}.pkl")
        input_pickle_path = os.path.join(subject_path, f"{self.data_firstname}{subject_name}_{slice_idx:03}.pkl")
                
        with open(label_pickle_path, 'rb') as file:
            label_dic = pickle.load(file)
            
        with open(input_pickle_path, 'rb') as file:
            input_dic = pickle.load(file)
        
        input = np.stack((input_dic[f'{self.data_dicname}_l'],input_dic[f'{self.data_dicname}_h']),axis=-1)  # (256, 256) 형태의 2D 이미지
        label = np.stack((label_dic[f'{self.label_dicname}_l'],label_dic[f'{self.label_dicname}_h']),axis=-1)
        # 필요한 경우, transform 적용 (e.g., 데이터 증강, 정규화 등)
        input = self.to_tensor(input)
        label = self.to_tensor(label)
        
        if self.transform:
            input,label = self.transform(input,label)
            
        return input, label, subject_name, subject_idx, z_slice
    

class PickelDatasetv2(Dataset):
    def __init__(self, root_dir, set_name, data_firstname ="FBP_",label_firstname ="MBIR_", data_dicname="fbp",label_dicname="mbir",patch_size = (128,128), z_slice = 672, transform=None):
        # data_dir: 데이터 파일들이 있는 디렉토리
        self.data_root = os.path.join(root_dir,set_name)
        self.z_slice = z_slice
        self.transform=transform
        #self.label_pickle=sorted([label.path for label in os.scandir(self.data_root) if label.name.startswith(label_firstname)])
        #self.input_pickle=sorted([input.path for input in os.scandir(self.data_root) if input.name.startswith(data_firstname)])
        self.subject_path_all =sorted([item.path for item in os.scandir(self.data_root)])
        self.label_firstname = label_firstname
        self.data_firstname = data_firstname
        self.patch_size = patch_size      
        self.data_dicname = data_dicname
        self.label_dicname = label_dicname
        self.label_pickle = ""
        self.to_tensor = ToTensor()

    
    
    def __len__(self):
        
        return len(self.subject_path_all) * self.z_slice # item * z_slice
    
    def __getitem__(self, idx):
        
        subject_idx = idx // self.z_slice  # 672개의 슬라이스를 가지므로, 파일 인덱스를 구함
        slice_idx = idx % self.z_slice  # 파일 내에서의 슬라이스 인덱스를 구함
        
        subject_path = self.subject_path_all[subject_idx]
        subject_name = subject_path.split("/")[-1]

        label_pickle_path = os.path.join(subject_path, f"{self.label_firstname}{subject_name}_{slice_idx:03}.pkl")
        input_pickle_path = os.path.join(subject_path, f"{self.data_firstname}{subject_name}_{slice_idx:03}.pkl")
                
        with open(label_pickle_path, 'rb') as file:
            label_dic = pickle.load(file)
            
        with open(input_pickle_path, 'rb') as file:
            input_dic = pickle.load(file)
        
        input = np.stack((input_dic[f'{self.data_dicname}_l'],input_dic[f'{self.data_dicname}_h']),axis=-1)  # (256, 256) 형태의 2D 이미지
        label = np.stack((label_dic[f'{self.label_dicname}_l'],label_dic[f'{self.label_dicname}_h']),axis=-1)
        # 필요한 경우, transform 적용 (e.g., 데이터 증강, 정규화 등)
        input = self.to_tensor(input)
        label = self.to_tensor(label)
        
        if self.transform:
            input,label = self.transform(input,label)
            
        return input, label, subject_name, subject_idx, self.z_slice
    
    
    
class RandomCrop(object):
    def __init__(self, size=(128,128)):
        self.size = size

    def __call__(self, image, label):

        h, w = image.shape[-2:]
        i, j = torch.randint(0, h - self.size[0], ()).item(), torch.randint(0, w - self.size[1], ()).item()

        image = image[:, i:i+self.size[0], j:j+self.size[1]]
        label = label[:, i:i+self.size[0], j:j+self.size[1]]

        return image, label
    
class RandomHorizontalFlip(object):
    def __init__(self):
        pass
    def __call__(self,image, label):
        if torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
        return image, label

class RandomVerticalFlip(object):
    def __init__(self):
        pass
    def __call__( self,image, label):
        if torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[1])
        return image, label
    
class RandomRotation(object):
    def __init__(self, degrees=90.0):
        self.degrees = degrees

    def __call__(self, image, label):

        if torch.rand(1) < 0.5:
            angle = random.uniform(-self.degrees, self.degrees)
            image = torchvision.transforms.functional.rotate(image, angle)
            label = torchvision.transforms.functional.rotate(label, angle)

        return image, label
    
    
class Dual_img_Compose:

    def __init__(self, transforms):

        self.transforms = transforms

    def __call__(self, input, label):

        for t in self.transforms:
            input, label = t(input,label)
        
        return input,label
