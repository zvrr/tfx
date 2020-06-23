# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Helper class to start TFX Tuner as a Job on Google Cloud AI Platform."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import multiprocessing
import os
from typing import Any, Dict, List, Text

import absl

from tfx import types
from tfx.components.base import base_executor
from tfx.components.trainer import constants
from tfx.components.tuner import executor as tuner_executor
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.utils import json_utils

# Directory to store intemediate hyperparamter search progress.
_WORKING_DIRECTORY = '/tmp'


class Executor(base_executor.BaseExecutor):
  """Tuner executor that launches parallel tuning flock on Cloud AI Platform."""

  # TODO(b/160013376): Refactor common parts with Trainer Executor.
  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Starts a Tuner component as a job on Google Cloud AI Platform."""
    self._log_startup(input_dict, output_dict, exec_properties)

    custom_config = json_utils.loads(
        exec_properties.get(constants.CUSTOM_CONFIG_KEY, 'null'))
    if custom_config is None:
      raise ValueError('custom_config is not provided')

    if custom_config is not None and not isinstance(custom_config, Dict):
      raise ValueError('custom_config in execution properties must be a dict.')

    training_inputs = custom_config.get(
        ai_platform_trainer_executor.TRAINING_ARGS_KEY)
    if training_inputs is None:
      err_msg = ('\'%s\' not found in custom_config.' %
                 ai_platform_trainer_executor.TRAINING_ARGS_KEY)
      absl.logging.error(err_msg)
      raise ValueError(err_msg)

    tune_args = tuner_executor.get_tune_args(exec_properties)

    num_parallel_trials = (1
                           if not tune_args else tune_args.num_parallel_trials)
    if num_parallel_trials > 1:
      # Chief node is also responsible for conducting tuning loop.
      worker_count = num_parallel_trials - 1

      absl.logging.warning(
          'workerCount is overridden with {}.'.format(worker_count))
      training_inputs['workerCount'] = worker_count

      training_inputs['scaleTier'] = 'CUSTOM'
      training_inputs['masterType'] = (
          training_inputs.get('masterType') or 'standard')
      training_inputs['workerType'] = (
          training_inputs.get('workerType') or 'standard')

    # 'tfx_tuner_YYYYmmddHHMMSS' is the default job ID if not specified.
    job_id = (
        custom_config.get(ai_platform_trainer_executor.JOB_ID_KEY) or
        'tfx_tuner_{}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))

    executor_class = _Executor
    executor_class_path = '%s.%s' % (executor_class.__module__,
                                     executor_class.__name__)

    # Note: exec_properties['custom_config'] here is a dict.
    return runner.start_aip_training(input_dict, output_dict, exec_properties,
                                     executor_class_path, training_inputs,
                                     job_id)


def _need_chief_oracle(exec_properties: Dict[Text, Any]) -> bool:
  """Returns True if the Tuner instance requires chief oracle process."""
  # TODO(b/143900133): Add support to CloudTuner that does not require chief
  #                    oracle process for distributed tuning.
  del exec_properties
  return True


class _Executor(base_executor.BaseExecutor):
  """TFX Tuner component executor as a job on Google Cloud AI Platform."""

  def _initialize_cluster_spec(self):
    """Load cluster specification from environment variable."""

    absl.logging.info('Initializing cluster spec...')

    cluster_spec = json.loads(os.environ.get('CLUSTER_SPEC', '{}'))

    self.is_distributed = False

    # If CLUSTER_SPEC is not present, assume single-machine tuning.
    if not cluster_spec:
      return

    self.master_addr, self.master_port = (
        cluster_spec['cluster']['master'][0].split(':'))

    # Master is the chief.
    self.tuner_id = (
        'chief' if cluster_spec['task']['type'] == 'master' else 'tuner-%s-%d' %
        (cluster_spec['task']['type'], cluster_spec['task']['index']))

    absl.logging.info('Tuner ID is: %s', self.tuner_id)

    self.is_chief = cluster_spec['task']['type'] == 'master'
    self.is_distributed = True

    # Will be populated when chief oracle is started.
    self._chief_process = None

    absl.logging.info('Cluster spec initalized with: %s', cluster_spec)

  def _start_chief_oracle_in_subprocess(
      self, input_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, List[types.Artifact]]):
    """Start a chief oracle in a subprocess."""

    def run_chief_oracle():
      """Invoke chief orachle."""
      absl.logging.info('chief_oracle() starting...')

      os.environ['KERASTUNER_ORACLE_IP'] = '0.0.0.0'
      os.environ['KERASTUNER_ORACLE_PORT'] = self.master_port
      os.environ['KERASTUNER_TUNER_ID'] = self.tuner_id

      absl.logging.info('Binding oracle chief server at: %s:%s',
                        os.environ['KERASTUNER_ORACLE_IP'],
                        os.environ['KERASTUNER_ORACLE_PORT'])

      # By design of KerasTuner, chief oracle blocks forever.
      tuner_executor.search(input_dict, exec_properties, _WORKING_DIRECTORY)

    p = multiprocessing.Process(target=run_chief_oracle)
    p.start()

    absl.logging.info('Chief oracle started at PID: %s', p.pid)

    return p

  def _search(
      self, input_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, List[types.Artifact]]):
    """Conduct a single search loop, setting up chief oracle if necessary."""

    if not self.is_distributed:
      return tuner_executor.search(input_dict, exec_properties,
                                   _WORKING_DIRECTORY)

    # If distributed search, and this node is chief, start a chief oracle
    # process before conducting search by itself.
    if self.is_chief:

      if _need_chief_oracle(exec_properties):
        # If chief, Tuner will block forever. As such, start it in a subprocess.
        self._chief_process = self._start_chief_oracle_in_subprocess(
            input_dict, exec_properties)

        os.environ['KERASTUNER_ORACLE_IP'] = self.master_addr
        os.environ['KERASTUNER_ORACLE_PORT'] = self.master_port

      # Conduct the search loop as a worker on the master node as well,
      # in order to signal the termination of search.
      os.environ['KERASTUNER_TUNER_ID'] = 'tuner-master-0'
      absl.logging.info('Chief worker is running Tuner ID : %s', 'tuner')

      tuner = tuner_executor.search(input_dict, exec_properties,
                                    _WORKING_DIRECTORY)

      return tuner

    # If not chief, do tuning loop.

    if _need_chief_oracle(exec_properties):
      os.environ['KERASTUNER_ORACLE_IP'] = self.master_addr
      os.environ['KERASTUNER_ORACLE_PORT'] = self.master_port

      absl.logging.info('Oracle chief is known to be at: %s:%s',
                        os.environ['KERASTUNER_ORACLE_IP'],
                        os.environ['KERASTUNER_ORACLE_PORT'])

    os.environ['KERASTUNER_TUNER_ID'] = self.tuner_id

    return tuner_executor.search(input_dict, exec_properties,
                                 _WORKING_DIRECTORY)

  def __init__(self, context):
    super(_Executor, self).__init__(context)
    self._initialize_cluster_spec()

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:

    tuner = self._search(input_dict, exec_properties)

    if not self.is_chief:
      absl.logging.info('Returning since this is not chief worker.')

    tuner_executor.write_best_hyperparameters(tuner, output_dict)

    if self._chief_process and self._chief_process.is_alive():
      absl.logging.info('Terminating chief oracle at PID: %s',
                        self._chief_process.pid)
      self._chief_process.terminate()
