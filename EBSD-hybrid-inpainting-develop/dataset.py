import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


def get_files(dir: str, pattern: str) -> list[str]:
    return sorted(list(map(os.path.abspath, Path(dir).glob(pattern))))


def load_ctf_to_tensor(ctf_file: str) -> Tensor:
    with torch.no_grad():
        try:
            data = np.loadtxt(ctf_file, skiprows=15, usecols=(1, 2, 5, 6, 7)); print(data)
        except Exception:
            data = np.loadtxt(ctf_file, skiprows=17, usecols=(1, 2, 5, 6, 7)); print(data)
        x_coords = data[:, 0]
        y_coords = data[:, 1]
        width = len(np.unique(x_coords))
        height = len(np.unique(y_coords))
        euler_angles = (data[:, 2:5].reshape(
            height, width, 3).transpose(2, 0, 1).astype(np.float32))
        return torch.tensor(euler_angles)


def save_ctf_as_pt(ctf_file: str) -> str:
    pt_file = ctf_file[:-4] + '.pt'
    if not os.path.exists(pt_file):
        torch.save(load_ctf_to_tensor(ctf_file), pt_file)
    return pt_file


def normalize(euler_angles: Tensor) -> Tensor:
    normalization_factor = 1 / torch.tensor([[[360]], [[180]], [[360]]])
    return euler_angles * normalization_factor


class EBSDDataset(Dataset):

    def create_pt_files(self, ctf_files: list[str]) -> list[str]:
        pt_files: list[str] = []
        for ctf_file in tqdm(ctf_files, desc='Creating/checking .pt files'):
            pt_file = save_ctf_as_pt(ctf_file)
            pt_files.append(pt_file)
        return pt_files

    def load_pt_to_tensor(self, pt_file: str) -> Tensor:
        data: Tensor = torch.load(pt_file, map_location=self.device)
        data = normalize(data)
        return data

    def __init__(self, data_dir: str, device: torch.device | str,
                 damage_density: float = 0.25, val: bool = False,
                 missing_region_size: int = 20) -> None:
        super().__init__()
        self.device = device
        self.damage_density = damage_density
        self.val = val
        self.missing_region_size = missing_region_size
        self.generator = torch.Generator(self.device)
        self.noisy_ctf_files = get_files(data_dir, '*noisy.ctf')
        self.clean_ctf_files = get_files(data_dir, '*clean.ctf')
        assert (len(self.noisy_ctf_files) == len(self.clean_ctf_files))
        # Create PyTorch (.pt) files which are faster to load than parsing CTF
        # files.
        self.noisy_pt_files = self.create_pt_files(self.noisy_ctf_files)
        self.clean_pt_files = self.create_pt_files(self.clean_ctf_files)

    def __len__(self) -> int:
        return len(self.noisy_pt_files)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        noisy_data = self.load_pt_to_tensor(self.noisy_pt_files[index])
        clean_data = self.load_pt_to_tensor(self.clean_pt_files[index])
        if self.val:
            # Ensure consistent mask generation for each image.
            self.generator = torch.Generator(self.device)
            self.generator.manual_seed(index)
        missing_region_top_left = torch.randint(0,
                                                noisy_data.shape[1]
                                                - self.missing_region_size,
                                                (2,),
                                                generator=self.generator)
        mask = torch.ones_like(noisy_data)
        mask[:,
             missing_region_top_left[0]:missing_region_top_left[0] + self.missing_region_size,
             missing_region_top_left[1]:missing_region_top_left[1] + self.missing_region_size] = 0
        return noisy_data, mask, clean_data
