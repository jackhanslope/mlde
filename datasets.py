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
"""Return training and evaluation/test datasets from config files."""
from datetime import timedelta
import logging
import os
import pickle
import yaml

from flufl.lock import Lock
from torch.utils.data import DataLoader
import xarray as xr

from ml_downscaling_emulator.training.dataset import build_input_transform, build_target_transform, XRDataset

def get_variables(config):
  data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', config.data.dataset_name)
  with open(os.path.join(data_dirpath, 'ds-config.yml'), 'r') as f:
      ds_config = yaml.safe_load(f)

  variables = [ pred_meta["variable"] for pred_meta in ds_config["predictors"] ]
  target_variables = ["target_pr"]

  return variables, target_variables

def get_transform(config, transform_dir, evaluation=False):
  dataset_transform_dir = os.path.join(transform_dir, config.data.dataset_name)
  os.makedirs(dataset_transform_dir, exist_ok=True)

  variables, target_variables = get_variables(config)

  if config.data.input_transform == "shared":
    input_transform_path = os.path.join(transform_dir, 'input.pickle')
  elif config.data.input_transform == "per-ds":
    input_transform_path = os.path.join(dataset_transform_dir, 'input.pickle')
  else:
    raise RuntimeError(f"Unknown tranform sharing {config.data.input_transform}")

  if config.data.target_transform == "shared":
    target_transform_path = os.path.join(transform_dir, 'target.pickle')
  elif config.data.target_transform == "per-ds":
    target_transform_path = os.path.join(dataset_transform_dir, 'target.pickle')
  else:
    raise RuntimeError(f"Unknown tranform sharing {config.data.target_transform}")


  lock_path = os.path.join(transform_dir, '.lock')
  lock = Lock(lock_path, lifetime=timedelta(hours=1))
  with lock:
    if os.path.exists(input_transform_path):
      with open(input_transform_path, 'rb') as f:
        logging.info(f"Using stored input transform: {input_transform_path}")
        input_transform = pickle.load(f)
    else:
      if evaluation and config.data.input_transform == "shared":
        raise RuntimeError("Shared input transform should only be fitted during training")
      logging.info("Fitting input transform")
      data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', config.data.dataset_name)
      xr_data_train = xr.load_dataset(os.path.join(data_dirpath, 'train.nc'))
      input_transform = build_input_transform(variables, config.data.image_size, key=config.data.input_transform_key)
      input_transform.fit_transform(xr_data_train)
      with open(input_transform_path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        logging.info(f"Storing input transform: {input_transform_path}")
        pickle.dump(input_transform, f, pickle.HIGHEST_PROTOCOL)
    if os.path.exists(target_transform_path):
      with open(target_transform_path, 'rb') as f:
        logging.info(f"Using stored target transform: {target_transform_path}")
        target_transform = pickle.load(f)
    else:
      if evaluation and config.data.target_transform == "shared":
        raise RuntimeError("Shared target transform should only be fitted during training")
      logging.info("Fitting target transform")
      data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', config.data.dataset_name)
      xr_data_train = xr.load_dataset(os.path.join(data_dirpath, 'train.nc'))
      target_transform = build_target_transform(target_variables, key=config.data.target_transform_key)
      target_transform.fit_transform(xr_data_train)
      with open(target_transform_path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        logging.info(f"Storing target transform: {target_transform_path}")
        pickle.dump(target_transform, f, pickle.HIGHEST_PROTOCOL)

  return input_transform, target_transform


def get_dataset(config, transform_dir, uniform_dequantization=False, evaluation=False, split='val'):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size

  # Create dataset builders for each dataset.
  if config.data.dataset == "XR":
    variables, target_variables = get_variables(config)

    transform, target_transform = get_transform(config, transform_dir, evaluation=evaluation)

    data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', config.data.dataset_name)
    xr_data_train = xr.load_dataset(os.path.join(data_dirpath, 'train.nc'))
    xr_data_eval = xr.load_dataset(os.path.join(data_dirpath, f'{split}.nc'))

    xr_data_train = transform.transform(xr_data_train)
    xr_data_train = target_transform.transform(xr_data_train)
    train_dataset = XRDataset(xr_data_train, variables)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)

    xr_data_eval = transform.transform(xr_data_eval)
    xr_data_eval = target_transform.transform(xr_data_eval)
    eval_dataset = XRDataset(xr_data_eval, variables)
    eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size)

    return train_data_loader, eval_data_loader, transform, target_transform

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')
