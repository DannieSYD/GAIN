import re
import os
import sys
import csv
import pty
import time
import torch
import queue
import GPUtil
import select
import timeit
import random
import logging
import enlighten
import threading
import subprocess
import numpy as np

from datetime import datetime
from itertools import repeat, count
from concurrent.futures import ThreadPoolExecutor, as_completed

allocated_memory = {}


def estimate_memory_usage(model, input_shape, batch_size=1, optimizer=None, float_dtype=torch.float32):
    float_size = np.dtype(float_dtype.numpy_dtype).itemsize

    total_params = sum([torch.prod(torch.tensor(param.shape)).item() for param in model.parameters()])
    params_memory = total_params * float_size

    if optimizer is not None:
        optimizer_memory = 0
        for state in optimizer.state.values():
            for item in state.values():
                if torch.is_tensor(item):
                    optimizer_memory += torch.prod(torch.tensor(item.shape)).item() * float_size
    else:
        optimizer_memory = 0

    input_memory = np.prod(input_shape) * batch_size * float_size

    total_memory = params_memory + optimizer_memory + input_memory

    return {
        'total_memory': total_memory,
        'params_memory': params_memory,
        'optimizer_memory': optimizer_memory,
        'input_memory': input_memory
    }


class TqdmHandler:
    def __init__(self, total, num_bars, manager):
        self.bars = queue.Queue()
        self.manager = manager
        self.total = total
        self.num_bars = num_bars
        self.BAR_FORMAT =\
            '{task}{desc_pad}{percentage:3.0f}%|{bar}| ' \
            '{count:{len_total}d}/{total:d} ' \
            '[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s, ' \
            'loss={loss:.4f}, valid_error={valid_error:.4f}, test_error={test_error:.4f}]'
        for i in range(1, num_bars + 1):
            bar = manager.counter(
                total=total, position=i, leave=False, unit='epochs',
                bar_format=self.BAR_FORMAT, task='', loss=0, valid_error=0, test_error=0)
            self.bars.put((bar, i))

    def acquire_bar(self):
        return self.bars.get()

    def release_bar(self, bar, position):
        bar.close()
        bar = manager.counter(
            total=self.total, position=position, leave=False, unit='epochs',
            bar_format=self.BAR_FORMAT, task='', loss=0, valid_error=0, test_error=0)
        self.bars.put((bar, position))

    def update(self, bar, progress):
        epoch, num_epochs, loss, valid_error, test_error = progress
        incr = epoch - bar.count
        bar.update(incr=incr, loss=loss, valid_error=valid_error, test_error=test_error)

    def close(self):
        while not self.bars.empty():
            bar, _ = self.bars.get()
            bar.close()


class Task:
    def __init__(self, model_name, memory_required, python_file, args=None):
        self.model_name = model_name
        self.memory_required = memory_required
        self.python_file = python_file
        self.args = args if args is not None else {}
        self.dataset = args['dataset'] if 'dataset' in args else None
        self.target = args['target'] if 'target' in args else None

    @property
    def name(self):
        return f'{self.model_name}_{self.dataset}_{self.target}'

    @staticmethod
    def initialize_csv_file(csv_file):
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = [
                'model_name', 'dataset', 'target',
                'best_validation_error', 'test_error', 'runtime_error', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def execute(self, handler, bar):
        logging.info(f'Executing task: {self.name}')
        start = timeit.default_timer()
        cmd = ['python', self.python_file]

        for key, value in self.args.items():
            cmd.append(f'--{key}')
            cmd.append(str(value))
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered mode

        master_fd, slave_fd = pty.openpty()

        process = subprocess.Popen(cmd, stdout=slave_fd, stderr=subprocess.PIPE, text=True, env=env)
        os.close(slave_fd)

        best_validation_error = None
        test_error = None

        progress_pattern = re.compile(r'Progress: (\d+)/(\d+)/([\d.]+)/([\d.]+)/([\d.]+)')
        best_validation_error_pattern = re.compile(r'Best validation error: ([\d.]+)')
        test_error_pattern = re.compile(r'Test error: ([\d.]+)')

        output_buffer = ''
        try:
            while process.poll() is None:
                readable, _, _ = select.select([master_fd, process.stderr], [], [], 1)

                for fd in readable:
                    output = os.read(fd, 1024).decode() if fd == master_fd else process.stderr.readline()
                    output_buffer += output
                    lines = output_buffer.split('\n')
                    output_buffer = lines.pop()

                    for line in lines:
                        if fd == master_fd:
                            flag = False

                            match = progress_pattern.search(line)
                            if match:
                                flag = True
                                epoch, num_epochs, loss, valid_error, test_error = match.groups()
                                epoch = int(epoch)
                                num_epochs = int(num_epochs)
                                loss = float(loss)
                                valid_error = float(valid_error)
                                test_error = float(test_error)
                                handler.update(bar, (epoch, num_epochs, loss, valid_error, test_error))

                            match = best_validation_error_pattern.search(line)
                            if match:
                                flag = True
                                best_validation_error = float(match.group(1))
                                logging.info(f'Task {self.name} best validation error: {best_validation_error:.5f}')

                            match = test_error_pattern.search(line)
                            if match:
                                flag = True
                                test_error = float(match.group(1))
                                logging.info(f'Task {self.name} test error: {test_error:.5f}')

                            if not flag:
                                # Log output from process.stdout
                                logging.info(f'Task {self.name}: {line.strip()}')
                        else:
                            # Log error messages from process.stderr
                            logging.warning(f'Task {self.name}: {line.strip()}')
        # except Exception as e:
        #     logging.exception(f'Error while executing task {self.name}: {e}')
        finally:
            os.close(master_fd)
            process.wait()
            end = timeit.default_timer()
            runtime = end - start

            if process.returncode == 0:
                return best_validation_error, test_error, process.returncode, runtime
            else:
                logging.warning(f'Task {self.name} failed with return code {process.returncode}')
                logging.warning(f'Error message: {process.stderr.read()}')
                return None, None, process.returncode, runtime


def gpu_memory_available():
    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        raise Exception('No GPUs available')

    available_gpus = [{'id': gpu.id, 'memoryFree': gpu.memoryFree} for gpu in gpus]
    return available_gpus


def select_gpu(task, available_gpus):
    global allocated_memory

    for gpu in available_gpus:
        if gpu['id'] in allocated_memory:
            gpu['memoryFree'] -= allocated_memory[gpu['id']]

    filtered_gpus = [gpu for gpu in available_gpus if task.memory_required <= gpu['memoryFree']]

    if not filtered_gpus:
        return None

    most_free_gpu = max(filtered_gpus, key=lambda gpu: gpu['memoryFree'])
    allocated_memory[most_free_gpu['id']] = allocated_memory.get(most_free_gpu['id'], 0) + task.memory_required

    return most_free_gpu['id']


def worker(task, csv_file, handler, max_tasks_sem):
    time.sleep(random.uniform(5, 10))
    available_gpus = gpu_memory_available()
    selected_gpu = select_gpu(task, available_gpus)

    while selected_gpu is None:
        logging.warning(f'Waiting for a GPU with sufficient memory for task: {task.name}')
        time.sleep(random.uniform(5, 10))
        available_gpus = gpu_memory_available()
        selected_gpu = select_gpu(task, available_gpus)

    logging.info(f'Selected GPU {selected_gpu} for task: {task.name}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)
    task.args['device'] = 'cuda:0'
    task.args['gpus'] = str(selected_gpu)
    # task.args['device'] = f'cuda:{selected_gpu}'

    with max_tasks_sem:
        bar, position = handler.acquire_bar()
        bar.update(task=f'[{selected_gpu}] {task.name}')
        errors = task.execute(handler, bar)
        handler.release_bar(bar, position)

    best_validation_error, test_error, runtime_error, runtime = errors
    logging.info(f'Task {task.name} finished in {runtime:.2f} seconds.')

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = [
            'model_name', 'dataset', 'target',
            'best_validation_error', 'test_error', 'runtime_error', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({
            'model_name': task.model_name,
            'dataset': task.dataset,
            'target': task.target,
            'best_validation_error': best_validation_error if best_validation_error is not None else '',
            'test_error': test_error if test_error is not None else '',
            'runtime_error': runtime_error,
            'timestamp': current_time
        })

    global allocated_memory
    allocated_memory[selected_gpu] -= task.memory_required

    if runtime_error != 0:
        raise RuntimeError(f'Failed with return code {runtime_error}')


if __name__ == '__main__':
    manager = enlighten.get_manager()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f'driver_log_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ])

    num_worker_threads = 3
    num_repeat_exps = 3
    handler = TqdmHandler(1000, num_worker_threads, manager)
    max_tasks_sem = threading.Semaphore(num_worker_threads)

    datasets = {
        # 'Kraken': [
        #     # 'dipolemoment', 'qpoletens_xx', 'qpoletens_yy', 'qpoletens_zz', 'qpole_amp',
        #     'sterimol_B5', 'sterimol_burB5', 'sterimol_L', 'sterimol_burL'],
        # 'BDE': ['BindingEnergy'],
        # 'EE': ['de'],
        # 'Drugs': ['ip', 'ea', 'chi']

        'Drugs': ['ip', 'ea', 'chi'],
        # 'EE': ['de'],
        # 'Kraken': ['sterimol_B5', 'sterimol_burB5'],
        # 'BDE': ['BindingEnergy'],
    }
    # num_conformers_params = [5, 10]
    # parameters = list(itertools.product(models, targets))
    csv_file = sys.argv[1]
    Task.initialize_csv_file(csv_file)

    tasks = []
    # for dataset, targets in datasets.items():
    #     for target in targets:
    #         for num_conformers in num_conformers_params:
    #             tasks.append(Task(
    #                 f'SchNet_DeepSets_{num_conformers}', 10000, 'train_4d.py',
    #                 args={
    #                     'model4d:graph_encoder': 'SchNet', 'model4d:set_encoder': 'DeepSets',
    #                     'dataset': dataset, 'target': target, 'max_num_conformers': num_conformers}))
    #             tasks.append(Task(
    #                 f'DimeNet_DeepSets_{num_conformers}', 10000, 'train_4d.py',
    #                 args={
    #                     'model4d:graph_encoder': 'DimeNet++', 'model4d:set_encoder': 'DeepSets',
    #                     'max_num_conformers': num_conformers,
    #                     'learning_rate': 1e-4, 'batch_size': 64, 'dataset': dataset, 'target': target}))
    #             tasks.append(Task(
    #                 f'LEFTNet_DeepSets_{num_conformers}', 10000, 'train_4d.py',
    #                 args={
    #                     'model4d:graph_encoder': 'LEFTNet', 'model4d:set_encoder': 'DeepSets',
    #                     'batch_size': 64, 'learning_rate': 1e-4,
    #                     'dataset': dataset, 'target': target, 'max_num_conformers': num_conformers}))

    # for dataset, targets in datasets.items():
    #     for target in targets:
            # tasks.extend(repeat(Task(
            #     'SchNet_eval_random', 10000, 'train_3d.py',
            #     args={'model3d:model': 'SchNet', 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'DimeNet++_eval_random', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'DimeNet++', 'batch_size': 128, 'dataset': dataset, 'target': target,
            #         'learning_rate': 1e-4}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'LEFTNet_eval_random', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'LEFTNet', 'batch_size': 64, 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'GemNet_eval_random', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'GemNet', 'batch_size': 64, 'dataset': dataset, 'target': target,
            #         'learning_rate': 1e-4}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'ClofNet_eval_random', 10000, 'train_3d.py',
            #     args={'model3d:model': 'ClofNet', 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'PaiNN_eval_random', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'PaiNN', 'batch_size': 128, 'scheduler': 'ReduceLROnPlateau',
            #         'learning_rate': 0.0001, 'dataset': dataset, 'target': target}), num_repeat_exps))

            # tasks.extend(repeat(Task(
            #     f'SchNet_eval_all', 10000, 'train_3d_aug.py',
            #     args={
            #         'model4d:graph_encoder': 'SchNet', 'model4d:set_encoder': 'Mean',
            #         'learning_rate': 1e-4, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     f'DimeNet_eval_all', 10000, 'train_3d_aug.py',
            #     args={
            #         'model4d:graph_encoder': 'DimeNet++', 'model4d:set_encoder': 'Mean',
            #         'learning_rate': 1e-4, 'batch_size': 64, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     f'LEFTNet_eval_all', 10000, 'train_3d_aug.py',
            #     args={
            #         'model4d:graph_encoder': 'LEFTNet', 'model4d:set_encoder': 'Mean',
            #         'batch_size': 64, 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'ClofNet_eval_all', 10000, 'train_3d_aug.py',
            #     args={
            #         'model4d:graph_encoder': 'ClofNet', 'model4d:set_encoder': 'Mean',
            #         'dataset': dataset, 'target': target, 'learning_rate': 5e-4}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'GemNet_eval_all', 10000, 'train_3d_aug.py',
            #     args={
            #         'model4d:graph_encoder': 'GemNet', 'model4d:set_encoder': 'Mean',
            #         'batch_size': 64, 'dataset': dataset, 'target': target,
            #         'learning_rate': 1e-4}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'PaiNN_eval_all', 10000, 'train_3d_aug.py',
            #     args={
            #         'model4d:graph_encoder': 'PaiNN', 'model4d:set_encoder': 'Mean',
            #         'batch_size': 128, 'scheduler': 'ReduceLROnPlateau',
            #         'learning_rate': 0.0001, 'dataset': dataset, 'target': target}), num_repeat_exps))

    # for target in ['sterimol_B5', 'sterimol_L', 'sterimol_burB5', 'sterimol_burL']:
    #     tasks.extend(repeat(Task(
    #         'GemNet_Attention_20', 10000, 'train_4d.py',
    #         args={
    #             'model4d:graph_encoder': 'GemNet', 'model4d:set_encoder': 'Attention',
    #             'learning_rate': 0.0001, 'batch_size': 32, 'dataset': 'Kraken', 'target': target}), 2))
    # for target in ['sterimol_L', 'sterimol_burL']:
    #     tasks.extend(repeat(Task(
    #         'PaiNN_Attention_20', 10000, 'train_4d.py',
    #         args={
    #             'model4d:graph_encoder': 'PaiNN', 'model4d:set_encoder': 'Attention',
    #             'batch_size': 128, 'scheduler': 'ReduceLROnPlateau',
    #             'learning_rate': 0.0001, 'dataset': 'Kraken', 'target': target}), 2))

    # random_offset = random.randint(0, 10000)
    # port_counter = count(start=10000 + random_offset, step=1)

    wandb_project = 'Auto4D-DimeNet++-3d-Scai3'
    additional_notes = 'lr of 5e-4 w/ 1000 epochs'
    for dataset, targets in datasets.items():
        for target in targets:
            # 1D models
            # tasks.append(Task(
            #     'LSTM', 8000, 'train_1d.py',
            #     args={'model1d:model': 'LSTM', 'dataset': dataset, 'target': target}))
            # tasks.append(Task(
            #     'Transformer', 10000, 'train_1d.py',
            #     args={
            #         'model1d:model': 'Transformer', 'batch_size': 128,
            #         'dataset': dataset, 'target': target}))
            # tasks.append(Task(
            #     'RandomForest_2D', 0, 'train_fp_rf.py',
            #     args={'dataset': dataset, 'target': target, 'modelfprf:modality': '2D'}))
            # tasks.append(Task(
            #     'RandomForest_3D', 0, 'train_fp_rf.py',
            #     args={'dataset': dataset, 'target': target, 'modelfprf:modality': '3D'}))

            # 2D models
            # tasks.append(Task(
            #     'GIN', 5000, 'train_2d.py',
            #     args={
            #         'model2d:model': 'GIN', 'model2d:gin:virtual_node': False,
            #         'dataset': dataset, 'target': target}))
            # tasks.append(Task(
            #     'GIN-VN', 10000, 'train_2d.py',
            #     args={
            #         'model2d:model': 'GIN', 'model2d:gin:virtual_node': True,
            #         'model2d:gin:num_layers': 6,
            #         'dataset': dataset, 'target': target}))
            # tasks.append(Task(
            #     'GPS', 10000, 'train_2d.py',
            #     args={'model2d:model': 'GPS', 'dataset': dataset, 'target': target}))
            # tasks.append(Task(
            #     'ChemProp', 10000, 'train_2d.py',
            #     args={
            #         'model2d:model': 'ChemProp', 'scheduler': 'OneCycleLR', 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}))

            # 3D models with random sampling
            # tasks.extend(repeat(Task(
            #     'SchNet', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'SchNet', 'dataset': dataset, 'target': target,
            #         'learning_rate': 1e-3}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'DimeNet++', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'DimeNet++', 'batch_size': 128, 'dataset': dataset, 'target': target,
            #         'num_epochs': 1000,'port': random.randint(10000, 20000), 'learning_rate': 5e-4,
            #         'wandb_project': wandb_project,
            #         'additional_notes': additional_notes}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'GemNet', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'GemNet', 'learning_rate': 1e-4, 'batch_size': 40,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'ClofNet', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'ClofNet', 'dataset': dataset, 'target': target,
            #         'learning_rate': 5e-4}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'LEFTNet', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'LEFTNet', 'batch_size': 64, 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'Equiformer', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'Equiformer',
            #         'batch_size': 64, 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'PaiNN', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'PaiNN', 'batch_size': 128, 'scheduler': 'ReduceLROnPlateau',
            #         'learning_rate': 1e-4, 'dataset': dataset, 'target': target}), num_repeat_exps))

            # 3D models without random sampling
            # tasks.extend(repeat(Task(
            #     'SchNet_w/o_aug', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'SchNet', 'model3d:augmentation': False,
            #         'learning_rate': 1e-3, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'DimeNet++_w/o_aug', 5000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'DimeNet++', 'model3d:augmentation': False,
            #         'batch_size': 256, 'dataset': dataset, 'target': target,
            #         'learning_rate': 2e-4, 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': wandb_project}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'GemNet_w/o_aug', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'GemNet', 'model3d:augmentation': False, 'learning_rate': 1e-4,
            #         'batch_size': 40, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'ClofNet_w/o_aug', 5000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'ClofNet', 'model3d:augmentation': False,
            #         'batch_size': 256, 'dataset': dataset, 'target': target,
            #         'learning_rate': 2e-4, 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': wandb_project}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'LEFTNet_w/o_aug', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'LEFTNet',  'model3d:augmentation': False,
            #         'batch_size': 64, 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'Equiformer_w/o_aug', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'Equiformer',  'model3d:augmentation': False,
            #         'batch_size': 64, 'learning_rate': 2e-4,
            #         'dataset': dataset, 'target': target, 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001,}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'PaiNN_w/o_aug', 5000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'PaiNN', 'model3d:augmentation': False,
            #         'batch_size': 256, 'dataset': dataset, 'target': target,
            #         'learning_rate': 2e-4, 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': wandb_project}), num_repeat_exps))
            # tasks.extend([Task(
            #     'EquiformerV2_w/o_aug', 10000, 'train_3d.py',
            #     args={
            #         'model3d:model': 'EquiformerV2', 'model3d:augmentation': False,
            #         'learning_rate': 2e-4, 'batch_size': 16, 'dataset': dataset, 'target': target,
            #         'optimizer': 'AdamW', 'scheduler': 'LambdaLR', 'weight_decay': 0.001,
            #         'wandb_project': wandb_project})
            #         'model3d:equiformer_v2:num_layers': 6, 'model3d:equiformer_v2:num_heads': 4,
            #         'model3d:equiformer_v2:lmax_list': 2, 'model3d:equiformer_v2:mmax_list': 2,
            #         'model3d:equiformer_v2:use_gate_act': True,
            #         'model3d:equiformer_v2:use_grid_mlp': False, 'model3d:equiformer_v2:use_sep_s2_act': False})
            #         'model3d:equiformer_v2:attn_hidden_channels': 64, })
            #     for _ in range(num_repeat_exps)])

            # 4D models
            # tasks.extend(repeat(Task(
            #     'SchNet_Mean_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'SchNet', 'model4d:set_encoder': 'Mean',
            #         'learning_rate': 1e-3, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'SchNet_DeepSets_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'SchNet', 'model4d:set_encoder': 'DeepSets',
            #         'learning_rate': 1e-3, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'SchNet_Attention_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'SchNet', 'model4d:set_encoder': 'Attention',
            #         'learning_rate': 1e-3, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'DimeNet_Mean_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'DimeNet++', 'model4d:set_encoder': 'Mean',
            #         'learning_rate': 5e-4, 'batch_size': 64, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'DimeNet_DeepSets_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'DimeNet++', 'model4d:set_encoder': 'DeepSets',
            #         'learning_rate': 5e-4, 'batch_size': 64, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'DimeNet_Attention_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'DimeNet++', 'model4d:set_encoder': 'Attention',
            #         'learning_rate': 5e-4, 'batch_size': 64, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'GemNet_Mean_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'GemNet', 'model4d:set_encoder': 'Mean',
            #         'learning_rate': 1e-4, 'batch_size': 40, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'GemNet_DeepSets_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'GemNet', 'model4d:set_encoder': 'DeepSets',
            #         'learning_rate': 1e-4, 'batch_size': 40, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'GemNet_Attention_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'GemNet', 'model4d:set_encoder': 'Attention',
            #         'learning_rate': 1e-4, 'batch_size': 40, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'ClofNet_Mean_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'ClofNet', 'model4d:set_encoder': 'Mean',
            #         'dataset': dataset, 'target': target, 'learning_rate': 5e-4}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'ClofNet_DeepSets_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'ClofNet', 'model4d:set_encoder': 'DeepSets',
            #         'dataset': dataset, 'target': target, 'learning_rate': 5e-4}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'ClofNet_Attention_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'ClofNet', 'model4d:set_encoder': 'Attention',
            #         'dataset': dataset, 'target': target, 'learning_rate': 5e-4}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'PaiNN_Mean_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'PaiNN', 'model4d:set_encoder': 'Mean',
            #         'batch_size': 128, 'scheduler': 'ReduceLROnPlateau',
            #         'learning_rate': 1e-4, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'PaiNN_DeepSets_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'PaiNN', 'model4d:set_encoder': 'DeepSets',
            #         'batch_size': 128, 'scheduler': 'ReduceLROnPlateau',
            #         'learning_rate': 1e-4, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'PaiNN_Attention_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'PaiNN', 'model4d:set_encoder': 'Attention',
            #         'batch_size': 128, 'scheduler': 'ReduceLROnPlateau',
            #         'learning_rate': 1e-4, 'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'LEFTNet_Mean_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'LEFTNet', 'model4d:set_encoder': 'Mean',
            #         'batch_size': 64, 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'LEFTNet_DeepSets_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'LEFTNet', 'model4d:set_encoder': 'DeepSets',
            #         'batch_size': 64, 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'LEFTNet_Attention_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'LEFTNet', 'model4d:set_encoder': 'Attention',
            #         'batch_size': 64, 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'Equiformer_Mean_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'Equiformer', 'model4d:set_encoder': 'Mean',
            #         'batch_size': 48, 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'Equiformer_DeepSets_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'Equiformer', 'model4d:set_encoder': 'DeepSets',
            #         'batch_size': 48, 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))
            # tasks.extend(repeat(Task(
            #     'Equiformer_Attention_20', 10000, 'train_4d.py',
            #     args={
            #         'model4d:graph_encoder': 'Equiformer', 'model4d:set_encoder': 'Attention',
            #         'batch_size': 48, 'learning_rate': 1e-4,
            #         'dataset': dataset, 'target': target}), num_repeat_exps))

            # 4D models with distributed training
            # tasks.extend([Task(
            #     'DimeNet_Mean_20', 10000, 'train_4d_dist.py',
            #     args={
            #         'model4d:graph_encoder': 'DimeNet++', 'model4d:set_encoder': 'Mean',
            #         'learning_rate': 5e-4, 'batch_size': 4, 'dataset': dataset, 'target': target,
            #         'seed': random.randint(0, 20000), 'num_epochs': 1000,
            #         'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': wandb_project, 'additional_notes': additional_notes})
            #     for _ in range(num_repeat_exps)])
            # tasks.extend([Task(
            #     'DimeNet_DeepSets_20', 10000, 'train_4d_dist.py',
            #     args={
            #         'model4d:graph_encoder': 'DimeNet++', 'model4d:set_encoder': 'DeepSets',
            #         'learning_rate': 5e-4, 'batch_size': 4, 'dataset': dataset, 'target': target,
            #         'seed': random.randint(0, 20000), 'num_epochs': 1000,
            #         'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': wandb_project, 'additional_notes': additional_notes})
            #     for _ in range(num_repeat_exps)])
            # tasks.extend([Task(
            #     'ClofNet_Mean_20', 10000, 'train_4d_dist.py',
            #     args={
            #         'model4d:graph_encoder': 'ClofNet', 'model4d:set_encoder': 'Mean',
            #         'learning_rate': 5e-4, 'batch_size': 10, 'dataset': dataset, 'target': target,
            #         'seed': random.randint(0, 20000), 'num_epochs': 1000,
            #         'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': wandb_project, 'additional_notes': additional_notes})
            #     for _ in range(num_repeat_exps)])
            # tasks.extend([Task(
            #     'ClofNet_DeepSets_20', 10000, 'train_4d_dist.py',
            #     args={
            #         'model4d:graph_encoder': 'ClofNet', 'model4d:set_encoder': 'DeepSets',
            #         'learning_rate': 5e-4, 'batch_size': 10, 'dataset': dataset, 'target': target,
            #         'seed': random.randint(0, 20000), 'num_epochs': 1000,
            #         'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': wandb_project, 'additional_notes': additional_notes})
            #     for _ in range(num_repeat_exps)])
            # tasks.extend([Task(
            #     'PaiNN_Mean_20', 10000, 'train_4d_dist.py',
            #     args={
            #         'model4d:graph_encoder': 'PaiNN', 'model4d:set_encoder': 'Mean',
            #         'learning_rate': 5e-4, 'batch_size': 6, 'dataset': dataset, 'target': target,
            #         'seed': random.randint(0, 20000), 'num_epochs': 1000,
            #         'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': wandb_project, 'additional_notes': additional_notes})
            #     for _ in range(num_repeat_exps)])
            # tasks.extend([Task(
            #     'PaiNN_DeepSets_20', 10000, 'train_4d_dist.py',
            #     args={
            #         'model4d:graph_encoder': 'PaiNN', 'model4d:set_encoder': 'DeepSets',
            #         'learning_rate': 5e-4, 'batch_size': 6, 'dataset': dataset, 'target': target,
            #         'seed': random.randint(0, 20000), 'num_epochs': 1000,
            #         'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': wandb_project, 'additional_notes': additional_notes})
            #     for _ in range(num_repeat_exps)])
            # tasks.extend([Task(
            #     'Equiformer_Mean_20', 10000, 'train_4d_dist.py',
            #     args={
            #         'model4d:graph_encoder': 'Equiformer', 'model4d:set_encoder': 'Mean',
            #         'seed': random.randint(0, 20000), 'num_epochs': 1000,
            #         'learning_rate': 5e-4, 'batch_size': 1, 'dataset': dataset, 'target': target,
            #         'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': wandb_project, 'additional_notes': additional_notes})
            #     for _ in range(num_repeat_exps)])
            # tasks.extend([Task(
            #     'Equiformer_DeepSets_20', 10000, 'train_4d_dist.py',
            #     args={
            #         'model4d:graph_encoder': 'Equiformer', 'model4d:set_encoder': 'DeepSets',
            #         'seed': random.randint(0, 20000), 'num_epochs': 1000,
            #         'learning_rate': 5e-4, 'batch_size': 1, 'dataset': dataset, 'target': target,
            #         'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': wandb_project, 'additional_notes': additional_notes})
            #     for _ in range(num_repeat_exps)])
            # tasks.extend([Task(
            #     'EquiformerV2_Mean_20', 10000, 'train_4d_dist.py',
            #     args={
            #         'model4d:graph_encoder': 'EquiformerV2', 'model4d:set_encoder': 'Mean',
            #         'learning_rate': 2e-4, 'batch_size': 1, 'dataset': dataset, 'target': target,
            #         'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': 'Auto3D-EquiformerV2',
            #         'model4d:equiformer_v2:num_layers': 6, 'model4d:equiformer_v2:num_heads': 4,
            #         'model4d:equiformer_v2:lmax_list': 3, 'model4d:equiformer_v2:num_distance_basis': 128,
            #         'additional_notes': additional_notes})
            #     for _ in range(num_repeat_exps)])
            # tasks.extend([Task(
            #     'EquiformerV2_DeepSets_20', 10000, 'train_4d_dist.py',
            #     args={
            #         'model4d:graph_encoder': 'EquiformerV2', 'model4d:set_encoder': 'DeepSets',
            #         'learning_rate': 2e-4, 'batch_size': 1, 'dataset': dataset, 'target': target,
            #         'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
            #         'weight_decay': 0.001, 'wandb_project': 'Auto3D-EquiformerV2',
            #         'model4d:equiformer_v2:num_layers': 6, 'model4d:equiformer_v2:num_heads': 4,
            #         'model4d:equiformer_v2:lmax_list': 3, 'model4d:equiformer_v2:num_distance_basis': 128,
            #         'additional_notes': additional_notes})
            #     for _ in range(num_repeat_exps)])
            tasks.extend([Task(
                'ViSNet_Mean_20', 10000, 'train_4d_dist.py',
                args={
                    'model4d:graph_encoder': 'ViSNet', 'model4d:set_encoder': 'Mean',
                    'learning_rate': 5e-4, 'batch_size': 6, 'dataset': dataset, 'target': target,
                    'seed': random.randint(0, 20000), 'num_epochs': 1000,
                    'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
                    'weight_decay': 0.001, 'wandb_project': 'Auto3D-ViSNet', 'additional_notes': additional_notes})
                for _ in range(num_repeat_exps)])
            tasks.extend([Task(
                'ViSNet_DeepSets_20', 10000, 'train_4d_dist.py',
                args={
                    'model4d:graph_encoder': 'ViSNet', 'model4d:set_encoder': 'DeepSets',
                    'learning_rate': 5e-4, 'batch_size': 6, 'dataset': dataset, 'target': target,
                    'seed': random.randint(0, 20000), 'num_epochs': 1000,
                    'port': random.randint(10000, 20000), 'optimizer': 'AdamW', 'scheduler': 'LambdaLR',
                    'weight_decay': 0.001, 'wandb_project': 'Auto3D-ViSNet', 'additional_notes': additional_notes})
                for _ in range(num_repeat_exps)])

    random.shuffle(tasks)

    # for target in datasets['Drugs']:
    #     tasks.append(Task(
    #         'PaiNN', 32000, 'train_3d.py',
    #         args={
    #             'model3d:model': 'PaiNN', 'batch_size': 128, 'scheduler': 'CosineAnnealingLR',
    #             'model3d:painn:num_interactions': 3, 'model3d:painn:num_rbf': 32, 'model3d:painn:cutoff': 5.0,
    #             'learning_rate': 0.0001, 'dataset': 'Drugs', 'target': target}))

    # performance_values_lock = threading.Lock()

    total_progress_bar = manager.counter(total=len(tasks), desc='Total progress', unit='tasks')
    status = manager.status_bar('', bar_format='{fill}{desc}', desc='\n' * 3)

    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        futures = {executor.submit(worker, task, csv_file, handler, max_tasks_sem): task for task in tasks}

        while futures:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                    total_progress_bar.update(1)
                    logging.info(f'Task {task.name} completed successfully')
                except Exception as e:
                    logging.warning(f'Task {task.name} failed: {e}. Retrying...')
                    new_future = executor.submit(worker, task, csv_file, handler, max_tasks_sem)
                    futures[new_future] = task
                del futures[future]
                logging.info(f'Remaining {len(futures)} tasks: {[task.name for task in futures.values()]}')

    logging.info('All tasks completed')
    total_progress_bar.close()
    handler.close()
    status.close()
