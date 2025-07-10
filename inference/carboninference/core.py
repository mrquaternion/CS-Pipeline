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
        self.config = config
        self.data = xr.open_dataset(
            nc_file,
            engine="netcdf4",
            chunks={"time": 1, "latitude": 128, "longitude": 128}
        ) \
        .swap_dims(({"valid_time": self.config.aux_data[0], 
                     "latitude": self.config.aux_data[1], 
                     "longitude": self.config.aux_data[2]})) \
        .rename_vars(({"valid_time": self.config.aux_data[0], 
                       "latitude": self.config.aux_data[1], 
                       "longitude": self.config.aux_data[2]}))


        # Precompute DOY and TOD columns
        times = pd.to_datetime(self.data[self.config.aux_data[0]].values)
        self.data = self.data.assign(
            DOY = ("timestamps", times.day_of_year.astype(float)),
            TOD = ("timestamps", times.hour.astype(float)),
        )

        # Load static IGBP and downscale
        raw_igbp = rxr.open_rasterio(rasterio.open(igbp_file), masked=True)
        self.igbp = raw_igbp.to_dataset(name="igbp") \
                            .interp(
                                y=self.data.lat,
                                x=self.data.lon,
                                method="nearest"
                            ) \
                            .drop_vars(["band", "spatial_ref", "y", "x"]) \
                            .rename_vars({"lat": "y", "lon": "x"}) \
                            .swap_dims({"lat": "y", "lon": "x"}) \
                                 

        # Build integer-index tuples (t, y, x)
        nt = self.data.dims["timestamps"]
        nlat = self.data.dims["lat"]
        nlon = self.data.dims["lon"]
        self.indices = []
        for t in range(config.window_size-1, nt):
            for y in range(nlat):
                for x in range(nlon):
                    self.indices.append((t, y, x))

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
        return len(self.indices)
    
    def __getitem__(self, idx):
        t, y, x = self.indices[idx]
        ws = self.config.window_size
        t0 = t - ws + 1

        # Slice main variables into a (window_size, n_vars) tensor
        sel = dict(timestamps=slice(t0, t+1), lat=y, lon=x)
        da = self.data.isel(**sel).to_array()      # shape: (n_vars, ws)
        pred_tensor = tr.tensor(da.values).float()

        # Slice IGBP label, then repeat to match time window if needed
        igbp_val = self.igbp.isel(y=y, x=x)["igbp"].values
        igbp_tensor = tr.full((ws,), float(igbp_val))

        # Build auxiliary time features
        aux_tensors = []
        for dim in self.config.aux_data[1:-1]:       
            vals = self.data.coords[dim].isel(timestamps=slice(t0, t+1)).values
            aux_tensors.append(tr.tensor(vals).float())
        aux_tensor = tr.stack(aux_tensors, dim=0)      # shape: (n_aux, ws)

        return (
            igbp_tensor,
            self.data.coords[self.config.aux_data[0]].values[t0:t+1],
            self.config.predictors,
            pred_tensor,
            self.config.aux_data,
            aux_tensor,
            self.config.targets
        )

    def collate_fn(self, batch):
        igbps, tss, preds, datas, auxs, aux_datas, targs = zip(*batch)
        return ERA5InferenceBatch(
            sites=("",), 
            igbp=igbps,
            timestamps=tss,
            predictor_columns=preds[0],
            predictor_values=tr.stack(datas, 0),
            aux_columns=auxs[0],
            aux_values=tr.stack(aux_datas, 0),
            modis=(), phenocam_ir=(), phenocam_rgb=(),
            target_columns=targs[0], target_values=None
        )