from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = Image_Generator(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = Image_Generator(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

def create_datasets(args):
    desi_training_path = f'/data3/DESI_Spectra/desi_spectra_train.hdf5'
    desi_validation_path = f'/data3/DESI_Spectra/desi_spectra_val.hdf5'
    ground_truth = f'DESI_redshift'
    
    train_dataset = Image_Generator(  
        desi_training_path,
        X_key='flux_norm_ivar',             
        y_key= ground_truth, # From your HDF5 structure
        scaler=False,
        augmenter = args.augment,
        labels_encoding=False,
        mode='train'
    )
    
    val_dataset = Image_Generator(
        desi_validation_path,
        X_key='flux_norm_ivar',
        y_key=ground_truth,
        scaler=False, 
        labels_encoding=False,
        mode='validation'
    )
    
    return train_dataset, val_dataset