import pandas as pd
import torch as tr
import xarray as xr
import rasterio
import rioxarray as rxr
from torch.utils.data import Dataset
from dataclasses import dataclass
from .constants import *
from typing import Optional, Tuple
from pathlib import Path


@dataclass
class ERA5InferenceConfig:
    targets: Tuple[str] = AMF_TARGETS
    predictors: Tuple[str] = AMF_PREDICTORS
    aux_data: Tuple[str] = GEO_TIME_PREDICTORS + ("igbp",)
    window_size: int = 32


@dataclass
class ERA5InferenceBatch:
    sites: Tuple[str] # Unuse in inference
    igbp: Tuple[str] 
    timestamps: Tuple
    predictor_columns: Tuple[str]
    predictor_values: tr.Tensor 
    aux_columns: Tuple[str]
    aux_values: tr.Tensor
    modis: Tuple # Unuse in inference
    phenocam_ir: Tuple # Unuse in inference
    phenocam_rgb: Tuple # Unuse in inference
    target_columns: Optional[Tuple[str]] = None
    target_values: Optional[tr.Tensor] = None


class ERA5Dataset(Dataset):
    def __init__(self, nc_file: Path, igbp_file: Path, config: ERA5InferenceConfig):
        self.data = xr.open_dataset(nc_file, engine="netcdf4")
        raw_igbp = rxr.open_rasterio(rasterio.open(igbp_file), masked=True)
        self.config = config
        self.timestamps = self.data.coords[self.config.aux_data[0]]
        self.time_dim = self.data.dims[self.config.aux_data[0]]
        self.lat_dim = self.data.dims[self.config.aux_data[1]]
        self.lon_dim = self.data.dims[self.config.aux_data[2]]

        self.data["DOY"] = xr.DataArray(
            data=pd.to_datetime(self.timestamps.values).day_of_year.astype(tr.float32),
            dims=self.timestamps.dims
        )
        self.data["TOD"] = xr.DataArray(
            data=pd.to_datetime(self.timestamps.values).hour.astype(tr.float32),
            dims=self.timestamps.dims
        )

        # Downscaling
        ds_igbp = raw_igbp.to_dataset(name="igbp")
        self.igbp = ds_igbp.interp(
            y=self.data.coords[self.config.aux_data[1]],
            x=self.data.coords[self.config.aux_data[2]],
            method="nearest"
        )

    def __len__(self):
        """
                 time
           _________________
          /               / |
         /______________ /  | latitude
        |               |   |
        |               |   /
        |               |  / longitude
        |               | /
        |_______________|/

        """
        return (self.time_dim - self.config.window_size + 1) * self.lat_dim * self.lon_dim
    
    def __getitem__(self, idx):
        top_idx = idx
        bot_idx = idx + self.config.window_size

        data_in = self.data.isel(time=slice(top_idx, bot_idx))
        data = tr.tensor(data_in.to_array().values).to(tr.float32)

        igbp_in = self.igbp.isel(y=slice(top_idx, bot_idx))
        igbp = tr.tensor(igbp_in.to_array().values).to(tr.float32)

        aux_data_in = [
            tr.tensor(
                self.data.coords[dim].isel(dim=slice(top_idx, bot_idx)).values
                for dim in self.config.aux_data[:-1]
            ).to(tr.float32)
        ]
        aux_data = tr.stack(aux_data_in, axis=0)

        return igbp, self.timestamps, \
               self.config.predictors, \
               data, self.config.aux_data, \
               aux_data, self.config.targets

    def collate_fn(self, batch):
        igbp, ts, preds, pred_data, aux, aux_data, targs = zip(*batch)

        return ERA5InferenceBatch(
            sites="",
            igbp=igbp,
            timestamps=ts,
            predictor_columns=preds,
            predictor_values=pred_data,
            aux_columns=aux,
            aux_values=aux_data,
            modis=(),
            phenocam_ir=(),
            phenocam_rgb=(),
            target_columns=targs,
            target_values=None
        )