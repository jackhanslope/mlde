# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return data loaders with pipelines used to transform the data."""
from datetime import timedelta
import logging
import os
import pickle
import yaml

from flufl.lock import Lock
from torch.utils.data import DataLoader
import xarray as xr

from ml_downscaling_emulator.training.dataset import build_input_transform, build_target_transform, XRDataset

from models import utils as mutils

def get_variables(dataset_name):
  # ideally this would be defined here
  # but code layout means can't import datasets model from the models package!
  # so proxying to mutils here
  return mutils.get_variables(dataset_name)

def create_transform(variables, active_dataset_name, model_src_dataset_name, transform_key, builder, store_path):
  logging.info(f"Fitting transform")
  model_src_dataset_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', model_src_dataset_name)
  model_src_training_split = xr.load_dataset(os.path.join(model_src_dataset_dirpath, 'train.nc'))

  active_dataset_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', active_dataset_name)
  active_dataset_training_split = xr.load_dataset(os.path.join(active_dataset_dirpath, 'train.nc'))

  xfm = builder(variables, key=transform_key)

  xfm.fit(active_dataset_training_split, model_src_training_split)

  save_transform(xfm, store_path)

  return xfm

def save_transform(xfm, path):
  with open(path, 'wb') as f:
        logging.info(f"Storing transform: {path}")
        pickle.dump(xfm, f, pickle.HIGHEST_PROTOCOL)

def load_transform(path):
  with open(path, 'rb') as f:
    logging.info(f"Using stored transform: {path}")
    xfm = pickle.load(f)

  return xfm

def find_or_create_transforms(active_dataset_name, model_src_dataset_name, transform_dir, input_transform_key, target_transform_key, evaluation):
  dataset_transform_dir = os.path.join(transform_dir, active_dataset_name)
  os.makedirs(dataset_transform_dir, exist_ok=True)
  input_transform_path = os.path.join(dataset_transform_dir, 'input.pickle')
  target_transform_path = os.path.join(transform_dir, 'target.pickle')

  variables, target_variables = get_variables(model_src_dataset_name)

  lock_path = os.path.join(transform_dir, '.lock')
  lock = Lock(lock_path, lifetime=timedelta(hours=1))
  with lock:
    if os.path.exists(input_transform_path):
      input_transform = load_transform(input_transform_path)
    else:

      input_transform = create_transform(variables, active_dataset_name, model_src_dataset_name, input_transform_key, build_input_transform, input_transform_path)

    if os.path.exists(target_transform_path):
      target_transform = load_transform(target_transform_path)
    else:
      if evaluation:
        raise RuntimeError("Target transform should only be fitted during training")
      target_transform = create_transform(target_variables, active_dataset_name, model_src_dataset_name, target_transform_key, build_target_transform, target_transform_path)

  return input_transform, target_transform


def get_dataset(config, active_dataset_name, model_src_dataset_name, transform_dir, batch_size, split, evaluation=False):
  """Create data loaders for given split.

  Args:
    active_dataset_name: Name of dataset from which to load data splits
    model_src_dataset_name: Name of dataset used to train the diffusion model (may be the same)
    transform_dir: Path to where transforms should be stored
    batch_size: Size of batch to use for DataLoaders
    evaluation: If `True`, fix number of epochs to 1.
    split: Split of the active dataset to load

  Returns:
    data_loader, transform, target_transform.
  """
  input_transform_key = config.data.input_transform_key
  target_transform_key = config.data.target_transform_key

  transform, target_transform = find_or_create_transforms(active_dataset_name, model_src_dataset_name, transform_dir, input_transform_key, target_transform_key, evaluation)

  data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets',active_dataset_name)
  xr_data = xr.load_dataset(os.path.join(data_dirpath, f'{split}.nc'))

  variables, target_variables = get_variables(model_src_dataset_name)

  xr_data = transform.transform(xr_data)
  xr_data = target_transform.transform(xr_data)
  xr_dataset = XRDataset(xr_data, variables, target_variables)
  data_loader = DataLoader(xr_dataset, batch_size=batch_size)

  return data_loader, transform, target_transform
