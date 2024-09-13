import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.ndimage import center_of_mass
from flows_utils import logit_trafo


class ZDCDataset(Dataset):
    def __init__(self, x, y, alpha, apply_logit=True, with_noise=True, noise_mul=1):
        self.x = x
        self.y = y
        self.alpha = alpha
        self.apply_logit = apply_logit
        self.with_noise = with_noise
        self.noise_mul = noise_mul
        print(f"Noise mul: {noise_mul}")

    def __len__(self):
        # assuming file was written correctly
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[idx]
        y = self.y[idx]

        if self.with_noise:
            x = add_noise(x, self.noise_mul)

        x = x / (x.sum(axis=(-1, -2), keepdims=True) + 1e-16)

        if self.apply_logit:
            x = logit_trafo(x, self.alpha)

        sample = {'img': x.to(torch.float32), 'conds': y.to(torch.float32)}

        return sample


def add_noise(input_tensor, noise_mul):
    noise = noise_mul * np.random.rand(*input_tensor.shape)
    return input_tensor + noise


def get_dataloader(x, y, alpha, device, full, batch_size=64, apply_logit=True, with_noise=False, noise_mul=1,
                   y_scaler_fit=None):

    kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}
    dataset_kwargs = {'with_noise': with_noise, "noise_mul": noise_mul}

    if full:
        dataset = ZDCDataset(x, y, alpha, apply_logit=apply_logit)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    else:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        print(f"\nUsed scaler: {scaler.__str__()}\n")
        if y_scaler_fit is not None:
            scaler.fit(y_scaler_fit)
            print("Scaler fit to the original particles data")
        else:
            scaler.fit(y_train)
        y_train = scaler.transform(y_train)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = scaler.transform(y_test)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        train_dataset = ZDCDataset(X_train, y_train, alpha, apply_logit=apply_logit, **dataset_kwargs)
        test_dataset = ZDCDataset(X_test, y_test, alpha, apply_logit=apply_logit, **dataset_kwargs)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        return train_dataloader, test_dataloader, scaler


def setup_center_of_mass_coords(images):
    coms = np.array([center_of_mass(images[i]) for i in range(images.shape[0])])
    return coms
