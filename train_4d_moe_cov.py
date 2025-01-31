import os
import time
import uuid
import copy
import wandb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from happy_config import ConfigLoader
from config import Config

import torch
import torch.distributed as dist
from torch.nn import BCELoss
import torch.nn.functional as F
import torch.multiprocessing as mp

from dataclasses import asdict
from torch_geometric import seed_everything
from torch.utils.data import random_split, Subset
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader
from data.ee import EEV2 as EE
from data.bde import BDEV2 as BDE
from data.drugs import DrugsV2 as Drugs
from data.kraken import KrakenV2 as Kraken
from sklearn.metrics import roc_auc_score, precision_score
from data.cov2 import COV23CLV2 as COV23CL, COV2V2 as COV2

from loaders.multipart import MultiPartLoaderV2
from loaders.samplers import DistributedEnsembleSampler, DistributedImbalancedEnsembleSampler
from utils.checkpoint import load_checkpoint
from utils.early_stopping import EarlyStopping

from models.dss.painn_dss import PaiNN
from models.dss.dimenet_dss import DimeNetPlusPlus
from models.dss.clofnet_v2_dss import ClofNet
from models.dss.equiformer_dss import Equiformer
from models.dss.topology import GIN
from models.dss.visnet_dss import ViSNet_DSS
from models.dss.gvp_dss import GVP_GNN
from models.dss.spherenet_dss import SphereNet
from models.dss.segnn_dss import SEGNNModel
from models.models_2d.graphgps.gps_model_2 import GPSModel
from models.model_4d_moe_cov import DSSNetV2

from utils.optim import get_optimizer, get_scheduler


def train(model, loader, optimizer, rank, epoch, batch_size, z_beta):
    model.train()

    binary_loss = BCELoss()
    total_loss = torch.zeros(2).to(rank)
    for i, data in enumerate(loader):
        # print(data[0].y)

        optimizer.zero_grad()
        num_molecules = data[0].y.size(0)
        out, z_loss, feature = model(data, epoch, batch_size, loss_expected=None)

        loss = binary_loss(out, data[0].y)
        loss.backward()
        optimizer.step()

        total_loss[0] += float(loss) * num_molecules
        total_loss[1] += num_molecules

    dist.barrier()
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    loss = float(total_loss[0] / total_loss[1])
    return loss


def evaluate(model, loader, epoch, batch_size, threshold=0.5):
    model.eval()
    all_targets = []
    all_probabilities = []
    binary_loss = BCELoss()
    total_loss = torch.zeros(2)

    with torch.no_grad():
        for data in loader:
            out, z_loss, feature = model.module(data, epoch, loss_expected=None, batch_size=batch_size)
            loss = binary_loss(out, data[0].y)
            num_molecules = data[0].y.size(0)
            total_loss[0] += float(loss) * num_molecules
            total_loss[1] += num_molecules

            all_targets.append(data[0].y.cpu())
            all_probabilities.append(out.cpu())

    loss = float(total_loss[0] / total_loss[1])

    all_targets = torch.cat(all_targets)
    all_probabilities = torch.cat(all_probabilities)
    auc = roc_auc_score(all_targets.numpy(), all_probabilities.numpy())

    predictions = (all_probabilities > threshold).numpy().astype(int)
    prc = precision_score(all_targets.numpy(), predictions)
    return loss, auc, prc


def run(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.port
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    if config.dataset == 'Drugs':
        dataset = Drugs('datasets/Drugs', max_num_conformers=config.max_num_conformers)
    elif config.dataset == 'Kraken':
        dataset = Kraken('datasets/Kraken', max_num_conformers=config.max_num_conformers)
    elif config.dataset == 'BDE':
        dataset = BDE('datasets/BDE', max_num_conformers=config.max_num_conformers)
    elif config.dataset == 'EE':
        dataset = EE('datasets/EE', max_num_conformers=config.max_num_conformers)
    elif config.dataset == 'CoV-3CL':
        dataset = COV23CL('datasets/COV2_3CL', max_num_conformers=20)
        threshold = 0.5
    elif config.dataset == 'CoV2':
        dataset = COV2('datasets/COV2', max_num_conformers=20)
        threshold = 0.1

    max_atomic_num = 100
    if config.modeldss.conf_encoder == 'DimeNet++':
        conf_model_factory = lambda: DimeNetPlusPlus(
            max_atomic_num=max_atomic_num, **asdict(config.modeldss.dimenetplusplus))
    elif config.modeldss.conf_encoder == 'PaiNN':
        conf_model_factory = lambda: PaiNN(max_atomic_num=max_atomic_num, **asdict(config.modeldss.painn))
    elif config.modeldss.conf_encoder == 'ClofNet':
        conf_model_factory = lambda: ClofNet(max_atomic_num=max_atomic_num, **asdict(config.modeldss.clofnet))
    elif config.modeldss.conf_encoder == 'Equiformer':
        conf_model_factory = lambda: Equiformer(
            max_atomic_num=max_atomic_num, **asdict(config.modeldss.equiformer))
    elif config.modeldss.conf_encoder == 'ViSNet':
        conf_model_factory = lambda: ViSNet_DSS(**asdict(config.modeldss.visnet))
    elif config.modeldss.conf_encoder == 'GVP':
        conf_model_factory = lambda: GVP_GNN(**asdict(config.modeldss.gvp))
    elif config.modeldss.conf_encoder == 'SphereNet':
        conf_model_factory = lambda: SphereNet(**asdict(config.modeldss.spherenet))
    elif config.modeldss.conf_encoder == 'SEGNN':
        conf_model_factory = lambda: SEGNNModel(**asdict(config.modeldss.segnn))

    if config.modeldss.topo_encoder == 'GIN':
        topo_model_factory = lambda: GIN(hidden_dim=config.hidden_dim, output_dim=128, num_layers=6)
    elif config.modeldss.topo_encoder == 'GPS':
        topo_model_factory = lambda: GPSModel(hidden_dim=config.hidden_dim, pe_dim=28, num_layers=6)

    seed_everything(config.seed)
    model = DSSNetV2(
        hidden_dim=config.hidden_dim, out_dim=1,
        conf_model_factory=conf_model_factory, topo_model_factory=topo_model_factory,
        num_experts=config.num_experts, num_activated=config.num_activated,
        num_parts=dataset.num_parts, device=f'cuda:{rank}',
        gig=config.gig, ad=config.ad, sag=config.sag, upc=config.upc, upcycling_epochs=config.upcycling_epochs,
        gumbel_tau=config.gumbel_tau).to(rank)
    if config.model4d.graph_encoder == 'ClofNet':
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    target_id = dataset.descriptors.index(config.target)
    dataset.y = dataset.y[:, target_id]
    print('Dataset length:', len(dataset.y))

    if config.dataset in ['CoV-3CL', 'CoV2']:
        splits = dataset.get_split_idx()
        train_dataset = Subset(dataset, splits['train'])
        valid_dataset = Subset(dataset, splits['val'])
        test_dataset = Subset(dataset, splits['test'])
    else:
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, [config.train_ratio, config.valid_ratio, 1 - config.train_ratio - config.valid_ratio])

    if config.dataset == "CoV2" or config.dataset == "CoV2_3CL":
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=DistributedImbalancedEnsembleSampler(
                dataset=train_dataset,
                num_replicas=world_size,
                rank=rank,
                batch_size=config.batch_size,
                batch_graph_size=config.batch_graph_size,
                major_to_minor_ratio=1,
            ),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=DistributedEnsembleSampler(
                dataset=train_dataset,
                num_replicas=world_size,
                rank=rank,
                batch_size=config.batch_size,
                batch_graph_size=config.batch_graph_size,
            ),
        )
    if rank == 0:
        valid_loader = DataLoader(valid_dataset, batch_size=8)
        test_loader = DataLoader(test_dataset, batch_size=8)

    optimizer = get_optimizer(model.parameters(), config)
    scheduler = get_scheduler(optimizer, config, train_dataset, world_size)

    start_epoch = 0
    if config.checkpoint is None:
        checkpoint_path = (
            f'checkpoints/'
            f'{config.dataset}_{config.target}_'
            f'{config.modeldss.conf_encoder}_{config.modeldss.topo_encoder}_'
            f'{uuid.uuid4()}.pt')
        if rank == 0:
            print(f'Saving checkpoint to: {checkpoint_path}')
    else:
        checkpoint_path = config.checkpoint
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Loaded checkpoint: {checkpoint_path} at epoch {start_epoch} on rank {rank}')
        dist.barrier()
    if rank == 0:
        early_stopping = EarlyStopping(patience=config.patience, path=checkpoint_path)
        wandb.init(project=config.wandb_project, config=asdict(config), group='train_4d.py')
        wandb.define_metric('epoch')
        wandb.define_metric('train_error', step_metric='epoch')
        wandb.define_metric('valid_error', step_metric='epoch')
        wandb.define_metric('test_error', step_metric='epoch')
        wandb.define_metric('valid_auc', step_metric='epoch')
        wandb.define_metric('test_auc', step_metric='epoch')
        wandb.define_metric('valid_prc', step_metric='epoch')
        wandb.define_metric('test_prc', step_metric='epoch')
    else:
        early_stopping = None
    dist.barrier()

    for epoch in range(start_epoch, config.num_epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        loss = train(model, train_loader, optimizer, rank, epoch, config.batch_size, config.z_beta)
        if scheduler is not None:
            scheduler.step()
        print(f'Rank: {rank}, Epoch: {epoch}/{config.num_epochs}, Loss: {loss:.5f}')
        if rank == 0:
            valid_loss, valid_auc, valid_prc = evaluate(model, valid_loader, epoch, config.batch_size, threshold)
            early_stopping(valid_loss, model, optimizer, epoch)
            if early_stopping.counter == 0:
                test_loss, test_auc, test_prc = evaluate(model, test_loader, epoch, config.batch_size, threshold)
            if early_stopping.early_stop:
                print('Early stopping...')
                break
            print(f'Progress: {epoch}/{config.num_epochs}/{loss:.5f}/{valid_loss:.5f}/{test_loss:.5f}')
            print(
                f'Metric: {epoch}: val/test auc: {valid_auc:.5f}/{test_auc:.5f}; val/test prc: {valid_prc:.5f}/{test_prc:.5f}')
            wandb.log({
                'epoch': epoch,
                'train_error': loss,
                'valid_error': valid_loss,
                'test_error': test_loss,
                'test_auc': test_auc,
                'valid_auc': valid_auc,
                'test_prc': test_prc,
                'valid_prc': valid_prc,
            })

            if (epoch + 1) % config.alert_epochs == 0:
                wandb.alert(
                    title=f'{epoch + 1} epochs reached',
                    text=f'{epoch + 1} epochs reached on '
                         f'{config.dataset} ({config.target}) using '
                         f'{config.model4d.graph_encoder} {config.model4d.set_encoder}')
        dist.barrier()

        if early_stopping is not None:
            early_stop = torch.tensor([early_stopping.early_stop], device=rank)
        else:
            early_stop = torch.tensor([False], device=rank)
        dist.broadcast(early_stop, src=0)
        if early_stop.item():
            break
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    loader = ConfigLoader(model=Config, config='/params/params_dss.json')
    config = loader()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    print(f"Visible GPU index: {config.gpus}")
    print("Cuda support:", torch.cuda.is_available(), ":", torch.cuda.device_count(), "devices")

    world_size = torch.cuda.device_count()
    print(f"Let's use {world_size} GPUs!")
    time_start = time.time()
    args = (world_size, config)
    mp.spawn(run, args=args, nprocs=world_size, join=True)
    time_end = time.time()
    print(f'Total time: {time_end - time_start:.2f} seconds')