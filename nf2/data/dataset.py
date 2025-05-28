import os

import numpy as np
import torch
from torch.utils.data import Dataset

from nf2.data.util import spherical_to_cartesian
from astropy import units as u

class BatchesDataset(Dataset):

    def __init__(self, batches_file_paths, batch_size=2 ** 13, **kwargs):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.batches_file_paths = batches_file_paths
        self.batch_size = int(batch_size)

    def __len__(self):
        ref_file = list(self.batches_file_paths.values())[0]
        n_batches = np.ceil(np.load(ref_file, mmap_mode='r').shape[0] / self.batch_size)
        return n_batches.astype(np.int32)

    def __getitem__(self, idx):
        # lazy load data
        data = {k: np.copy(np.load(bf, mmap_mode='r')[idx * self.batch_size: (idx + 1) * self.batch_size])
                for k, bf in self.batches_file_paths.items()}
        return data

    def clear(self):
        [os.remove(f) for f in self.batches_file_paths.values()]

class ImageDataset(Dataset):

    def __init__(self, cube_shape, norm, z=0):
        coordinates = np.stack(np.mgrid[:cube_shape[0],
                               :cube_shape[1]], -1)
        self.coordinates = coordinates
        self.coordinates_flat = coordinates.reshape((-1, 2))
        self.norm = norm
        self.z = z / self.norm

    def __len__(self, ):
        return self.coordinates_flat.shape[0]

    def __getitem__(self, idx):
        coord = self.coordinates_flat[idx]
        scaled_coord = [coord[0] / self.norm,
                        coord[1] / self.norm,
                        self.z]
        return np.array(scaled_coord, dtype=np.float32)


class CubeDataset(Dataset):

    def __init__(self, coord_range, ds_per_pixel=1 / 128, batch_size=2 ** 13):
        x_resolution = int((coord_range[0, 1] - coord_range[0, 0]) / ds_per_pixel)
        y_resolution = int((coord_range[1, 1] - coord_range[1, 0]) / ds_per_pixel)
        z_resolution = int((coord_range[2, 1] - coord_range[2, 0]) / ds_per_pixel)
        coords = np.stack(np.meshgrid(np.linspace(coord_range[0, 0], coord_range[0, 1], x_resolution, dtype=np.float32),
                                      np.linspace(coord_range[1, 0], coord_range[1, 1], y_resolution, dtype=np.float32),
                                      np.linspace(coord_range[2, 0], coord_range[2, 1], z_resolution, dtype=np.float32),
                                      indexing='ij'), -1)
        self.coords_shape = coords.shape[:-1]
        coords = coords.reshape(-1, 3)
        batch_size = int(batch_size)
        self.coords = np.split(coords, np.arange(batch_size, len(coords), batch_size))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        coord = torch.tensor(coord, dtype=torch.float32)
        return {'coords': coord}


class RandomSphericalCoordinateDataset(Dataset):

    def __init__(self, radius_range, batch_size, Mm_per_ds,
                 latitude_range=(0, np.pi), longitude_range=(0, 2 * np.pi),
                 radial_weighted_sampling=False, latitude_weighted_sampling=False, **kwargs):
        self.radius_range = radius_range
        self.Mm_per_ds = Mm_per_ds
        self.latitude_range = latitude_range
        self.longitude_range = longitude_range
        self.batch_size = batch_size
        self.float_tensor = torch.FloatTensor
        self.radial_weighted_sampling = radial_weighted_sampling
        self.latitude_weighted_sampling = latitude_weighted_sampling

    def __len__(self):
        return 1

    def __getitem__(self, item):
        random_coords = self.float_tensor(self.batch_size, 3).uniform_()
        # r [1, height]
        h_r = self.radius_range
        if self.radial_weighted_sampling:
            v_min, v_max = np.min(np.log(h_r)), np.max(np.log(h_r))
            random_coords[:, 0] = v_min + random_coords[:, 0] * (v_max - v_min)
            random_coords[:, 0] = torch.exp(random_coords[:, 0])
        else:
            random_coords[:, 0] = h_r[0] + random_coords[:, 0] * (h_r[1] - h_r[0])
        # theta [0, pi]
        if self.latitude_weighted_sampling:
            lat_r = self.latitude_range
            v_min, v_max = np.min(np.cos(lat_r)), np.max(np.cos(lat_r))
            random_coords[:, 1] = v_min + random_coords[:, 1] * (v_max - v_min)
            random_coords[:, 1] = torch.arccos(random_coords[:, 1])
        else:
            lat_r = self.latitude_range
            random_coords[:, 1] = lat_r[0] + random_coords[:, 1] * (lat_r[1] - lat_r[0])
        # phi [0, 2pi]
        lon_r = self.longitude_range
        random_coords[:, 2] = lon_r[0] + random_coords[:, 2] * (lon_r[1] - lon_r[0])
        # convert to cartesian
        random_coords = spherical_to_cartesian(random_coords, f=torch)
        random_coords = random_coords * (1 * u.solRad).to_value(u.Mm) / self.Mm_per_ds
        return {'coords': random_coords}


class SphereDataset(Dataset):

    def __init__(self, radius_range, Mm_per_ds, resolution=256, batch_size=1024, latitude_range=(0, np.pi), longitude_range=(0, 2 * np.pi), **kwargs):
        ratio = (latitude_range[1] - latitude_range[0]) / (longitude_range[1] - longitude_range[0])
        resolution_lat = int(resolution * ratio)
        coords = np.stack(
            np.meshgrid(np.linspace(radius_range[0], radius_range[1], resolution),
                        np.linspace(latitude_range[0], latitude_range[1], resolution_lat),
                        np.linspace(longitude_range[0], longitude_range[1], resolution),
                        indexing='ij')).T
        self.coords_shape = coords.shape[:-1]

        coords = spherical_to_cartesian(coords)
        coords = torch.tensor(coords, dtype=torch.float32)
        coords = coords.reshape((-1, 3))
        coords = coords * (1 * u.solRad).to_value(u.Mm) / Mm_per_ds
        self.coords = np.split(coords, np.arange(batch_size, len(coords), batch_size))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        return {'coords': coord}

class SphereSlicesDataset(Dataset):

    def __init__(self, radius_range, Mm_per_ds, latitude_range=(0, np.pi), longitude_range=(0, 2 * np.pi), longitude_resolution=256, batch_size=1024, n_slices=5, **kwargs):
        ratio = (latitude_range[1] - latitude_range[0]) / (longitude_range[1] - longitude_range[0])
        resolution_lat = int(longitude_resolution * ratio)
        coords = np.stack(
            np.meshgrid(np.linspace(radius_range[0], radius_range[1], n_slices),
                        np.linspace(latitude_range[0], latitude_range[1], resolution_lat),
                        np.linspace(longitude_range[0], longitude_range[1], longitude_resolution),
                        indexing='ij'), -1)
        self.cube_shape = coords.shape[:-1]

        coords = spherical_to_cartesian(coords)
        coords = torch.tensor(coords, dtype=torch.float32)
        coords = coords.reshape((-1, 3))
        coords = coords * (1 * u.solRad).to_value(u.Mm) / Mm_per_ds
        self.coords = np.split(coords, np.arange(batch_size, len(coords), batch_size))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        return {'coords': coord}

class SlicesDataset(Dataset):

    def __init__(self, coord_range, ds_per_pixel, n_slices=10, batch_size=4096, **kwargs):
        x_resolution = int((coord_range[0, 1] - coord_range[0, 0]) / ds_per_pixel)
        y_resolution = int((coord_range[1, 1] - coord_range[1, 0]) / ds_per_pixel)
        if coord_range[2, 0] == 0:
            z_range = np.linspace(coord_range[2, 0], coord_range[2, 1], n_slices, dtype=np.float32)
        else:
            z_range = np.linspace(0, coord_range[2, 1], n_slices - 1, dtype=np.float32)
            z_range = np.concatenate([np.array([coord_range[2, 0]]), z_range])
        coords = np.stack(np.meshgrid(np.linspace(coord_range[0, 0], coord_range[0, 1], x_resolution, dtype=np.float32),
                                        np.linspace(coord_range[1, 0], coord_range[1, 1], y_resolution, dtype=np.float32),
                                        z_range,
                                        indexing='ij'), -1)
        self.cube_shape = coords.shape[:-1]
        #
        coords = coords.reshape((-1, 3))
        self.coords = np.split(coords, np.arange(batch_size, len(coords), batch_size))

        super().__init__()

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        coord = torch.tensor(coord, dtype=torch.float32)
        return {'coords': coord}

class RandomCoordinateDataset(Dataset):

    def __init__(self, coord_range, batch_size=2 ** 14, buffer=None, z_sampling_exponent=1, current_density_map=None, current_biased_sampling=False, current_resample_interval=1000, bias_fraction=1.0, **kwargs):
        # Set defaults if not provided
        if current_biased_sampling is None:
            current_biased_sampling = False
        if current_resample_interval is None:
            current_resample_interval = 1000
        if bias_fraction is None:
            bias_fraction = 1.0
        super().__init__()
        if buffer:
            buffer_x = (coord_range[0, 1] - coord_range[0, 0]) * buffer
            buffer_y = (coord_range[1, 1] - coord_range[1, 0]) * buffer
            coord_range[0, 0] -= buffer_x
            coord_range[0, 1] += buffer_x
            coord_range[1, 0] -= buffer_y
            coord_range[1, 1] += buffer_y
        self.coord_range = coord_range
        self.batch_size = int(batch_size)
        self.float_tensor = torch.FloatTensor
        self.z_sampling_exponent = torch.tensor(z_sampling_exponent, dtype=torch.float32)
        self.current_density_map = current_density_map
        self.current_biased_sampling = current_biased_sampling
        self.current_resample_interval = current_resample_interval
        self.bias_fraction = bias_fraction
        self._step = 0
        if self.current_biased_sampling and self.current_density_map is not None:
            self._update_prob_map()

    def _update_prob_map(self):
        # Flatten and normalize the probability map
        prob = self.current_density_map.flatten()
        prob = prob / np.sum(prob)
        self._prob = prob
        self._shape = self.current_density_map.shape

    def __len__(self):
        return 1

    def __getitem__(self, item):
        if self.current_biased_sampling and self.current_density_map is not None and self.bias_fraction > 0:
            # Resample probability map at interval
            if self._step % self.current_resample_interval == 0:
                self._update_prob_map()
            self._step += 1
            n_biased = int(self.batch_size * self.bias_fraction)
            n_uniform = self.batch_size - n_biased
            coords_list = []
            # Biased sampling
            if n_biased > 0:
                idxs = np.random.choice(np.arange(self._prob.size), size=n_biased, p=self._prob)
                x_idx, y_idx = np.unravel_index(idxs, self._shape)
                z = np.random.uniform(self.coord_range[2, 0], self.coord_range[2, 1], n_biased)
                x = x_idx / (self._shape[0] - 1) * (self.coord_range[0, 1] - self.coord_range[0, 0]) + self.coord_range[0, 0]
                y = y_idx / (self._shape[1] - 1) * (self.coord_range[1, 1] - self.coord_range[1, 0]) + self.coord_range[1, 0]
                coords_biased = np.stack([x, y, z], axis=1)
                coords_list.append(coords_biased)
            # Uniform sampling
            if n_uniform > 0:
                random_coords = self.float_tensor(n_uniform, 3).uniform_().numpy()
                random_coords[:, 0] = (
                    random_coords[:, 0] * (self.coord_range[0, 1] - self.coord_range[0, 0]) + self.coord_range[0, 0])
                random_coords[:, 1] = (
                    random_coords[:, 1] * (self.coord_range[1, 1] - self.coord_range[1, 0]) + self.coord_range[1, 0])
                random_coords[:, 2] = random_coords[:, 2] ** self.z_sampling_exponent.item()
                random_coords[:, 2] = (
                    random_coords[:, 2] * (self.coord_range[2, 1] - self.coord_range[2, 0]) + self.coord_range[2, 0])
                coords_list.append(random_coords)
            coords = np.concatenate(coords_list, axis=0)
            np.random.shuffle(coords)
            coords = torch.tensor(coords, dtype=torch.float32)
            return {'coords': coords}
        else:
            random_coords = self.float_tensor(self.batch_size, 3).uniform_()
            random_coords[:, 0] = (
                        random_coords[:, 0] * (self.coord_range[0, 1] - self.coord_range[0, 0]) + self.coord_range[0, 0])
            random_coords[:, 1] = (
                        random_coords[:, 1] * (self.coord_range[1, 1] - self.coord_range[1, 0]) + self.coord_range[1, 0])
            random_coords[:, 2] = random_coords[:, 2] ** self.z_sampling_exponent
            random_coords[:, 2] = (
                        random_coords[:, 2] * (self.coord_range[2, 1] - self.coord_range[2, 0]) + self.coord_range[2, 0])
            return {'coords': random_coords}
