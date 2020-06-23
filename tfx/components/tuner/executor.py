# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Generic TFX tuner executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from typing import Any, Callable, Dict, List, Text, Optional
import absl
from kerastuner.engine import base_tuner

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.components.trainer import fn_args_utils
from tfx.components.util import udf_utils
from tfx.proto import tuner_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils

# Key for best hyperparameters in executor output_dict.
_BEST_HYPERPARAMETERS_KEY = 'best_hyperparameters'
# Key for tune args in executor exec_properties.
_TUNE_ARGS_KEY = 'tune_args'
# Default file name for generated best hyperparameters file.
_DEFAULT_FILE_NAME = 'best_hyperparameters.txt'


def get_tuner_fn(exec_properties: Dict[Text, Any]) -> Callable[..., Any]:
  """Returns tuner_fn from execution properties."""
  return udf_utils.get_fn(exec_properties, 'tuner_fn')


def get_tune_args(
    exec_properties: Dict[Text, Any]) -> Optional[tuner_pb2.TuneArgs]:
  """Returns TuneArgs protos from execution properties, if present."""
  tune_args = exec_properties.get(_TUNE_ARGS_KEY)
  if not tune_args:
    return None

  tune_args_proto = tuner_pb2.TuneArgs()
  json_format.Parse(tune_args, tune_args_proto)

  return tune_args_proto


def write_best_hyperparameters(
    tuner: base_tuner.BaseTuner,
    output_dict: Dict[Text, List[types.Artifact]]) -> None:
  """Writer out best hyperpeameters known to the given Tuner instance."""
  best_hparams_config = tuner.get_best_hyperparameters()[0].get_config()
  absl.logging.info('Best hyperParameters: %s' % best_hparams_config)
  best_hparams_path = os.path.join(
      artifact_utils.get_single_uri(output_dict[_BEST_HYPERPARAMETERS_KEY]),
      _DEFAULT_FILE_NAME)
  io_utils.write_string_file(best_hparams_path, json.dumps(best_hparams_config))
  absl.logging.info('Best Hyperparameters are written to %s.' %
                    best_hparams_path)


def search(input_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any],
           working_dir: Text) -> base_tuner.BaseTuner:
  """Conduct a single hyperparameter search loop."""
  tuner_fn = get_tuner_fn(exec_properties)

  fn_args = fn_args_utils.get_common_fn_args(input_dict, exec_properties,
                                             working_dir)

  tuner_fn_result = tuner_fn(fn_args)
  tuner = tuner_fn_result.tuner
  fit_kwargs = tuner_fn_result.fit_kwargs

  # TODO(b/156966497): set logger for printing.
  tuner.search_space_summary()
  absl.logging.info('Start tuning... Tuner ID: {}'.format(tuner.tuner_id))
  tuner.search(**fit_kwargs)
  absl.logging.info('Finished tuning... Tuner ID: {}'.format(tuner.tuner_id))
  tuner.results_summary()

  return tuner


class Executor(base_executor.BaseExecutor):
  """TFX Tuner component executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    if get_tune_args(exec_properties):
      raise ValueError(
          "TuneArgs is not supported for default Tuner's Executor.")

    tuner = search(input_dict, exec_properties, self._get_tmp_dir())

    write_best_hyperparameters(tuner, output_dict)
