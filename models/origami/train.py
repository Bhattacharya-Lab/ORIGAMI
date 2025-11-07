import os
import shutil
import argparse
import easydict
from easydict import EasyDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.utils.tensorboard
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader, Data
from tqdm.auto import tqdm
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_curve, auc

from utils.misc import BlackHole, get_logger, get_new_log_dir, load_config, seed_all, Counter
from utils.train import get_optimizer, get_scheduler, log_losses
from models.psr.datasets_24 import ComplexInterfaceDataset, ComplexPairBatchSampler, min_size_pair_collate, _dihedrals, _sidechains, _orientations_interface_aware
from models.psr.models import PSRNetwork
from models.psr.utils import report_correlations

def setup_ddp():
    """Set up distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        world_size = 1
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    return rank, world_size

def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False, help='Run only testing phase')
    parser.add_argument('--test_dataset', type=str, default='all', 
                       choices=['all', 'test1', 'test2', 'test3', 'casp16_5qa'], 
                       help='Which test dataset(s) to evaluate on')
    parser.add_argument('--custom_test_file', type=str, default=None,
                       help='Path to a custom .pt file for testing')
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size = setup_ddp()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed + rank)  # Different seed per rank

    # Define test dataset configurations
    test_configs = {
        'test1': {
            'pdb_dir': getattr(config.data, 'test1_pdb_dir', 'pdb_test'),
            'ilddt_dir': getattr(config.data, 'test1_ilddt_dir', 'ilddt_test'),
            'ss_file': getattr(config.data, 'test1_ss_file', 'secondary_structure/SS_test.result'),
            'rsa_dir': getattr(config.data, 'test1_rsa_dir', 'rsasa_test'),
            'name': 'Test Dataset 1 VoroIFGNN_af3 test dataset'
        },
        'test2': {
            'pdb_dir': getattr(config.data, 'test2_pdb_dir', 'pdb_casp16'),
            'ilddt_dir': getattr(config.data, 'test2_ilddt_dir', 'ilddt_casp16'),
            'ss_file': getattr(config.data, 'test2_ss_file', 'secondary_structure/SS_casp16.result'),
            'rsa_dir': getattr(config.data, 'test2_rsa_dir', 'rsasa_casp16'),
            'name': 'Test Dataset 2 Casp16 '
        },
        'test3': {
            'pdb_dir': getattr(config.data, 'test3_pdb_dir', 'pdb_casp15'),
            'ilddt_dir': getattr(config.data, 'test3_ilddt_dir', 'ilddt_casp15'),
            'ss_file': getattr(config.data, 'test3_ss_file', 'secondary_structure/SS_casp15.result'),
            'rsa_dir': getattr(config.data, 'test3_rsa_dir', 'rsasa_casp15'),
            'name': 'Test Dataset 3 Casp15'
        },
        'casp16_5qa': {
            'pdb_dir': getattr(config.data, 'casp16_5qa_pdb_dir', 'pdb_casp16_5QA_final'),
            'ilddt_dir': getattr(config.data, 'casp16_5qa_ilddt_dir', 'ilddt_casp16_5QA_final'),
            'ss_file': getattr(config.data, 'casp16_5qa_ss_file', 'secondary_structure/SS_casp16_5QA_final.result'),
            'rsa_dir': getattr(config.data, 'casp16_5qa_rsa_dir', 'rsasa_casp16_5QA_final'),
            'name': 'CASP16 5QA Test Dataset'
        }
    }

    # Logging (only rank 0 logs)
    if args.debug:
        logger = get_logger(config_name, None) if rank == 0 else BlackHole()
        writer = BlackHole()
        log_dir = None
        ckpt_dir = None
    else:
        if rank == 0:
            if args.resume is not None and args.overwrite:
                log_dir = os.path.dirname(os.path.dirname(args.resume))
            else:
                log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
                
            ckpt_dir = os.path.join(log_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            logger = get_logger('train', log_dir)
            writer = torch.utils.tensorboard.SummaryWriter(log_dir)
            logger.info(args)
            logger.info(config)
            logger.info(f"World size: {world_size}, Rank: {rank}")
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        else:
            logger = BlackHole()
            writer = BlackHole()
            log_dir = None
            ckpt_dir = None

    # Dataloaders
    if rank == 0:
        logger.info('Loading datasets...')

    # Only load training/validation datasets if we're not in test-only mode
    if not args.test:
        if rank == 0:
            logger.info('Loading training dataset...')
        
        train_set = ComplexInterfaceDataset(
            pdb_dir=os.path.join(config.data.root, config.data.train_pdb_dir),
            ilddt_dir=os.path.join(config.data.root, config.data.train_ilddt_dir),
            ss_file=os.path.join(config.data.root, "secondary_structure/SS_train.result"),
            rsa_dir=os.path.join(config.data.root, "rsasa_train")
        )
        
        if rank == 0:
            logger.info('Loading validation dataset...')
        
        val_set = ComplexInterfaceDataset(
            pdb_dir=os.path.join(config.data.root, config.data.val_pdb_dir),
            ilddt_dir=os.path.join(config.data.root, config.data.val_ilddt_dir),
            ss_file=os.path.join(config.data.root, "secondary_structure/SS_val.result"),
            rsa_dir=os.path.join(config.data.root, "rsasa_val")
        )
        
        # Create distributed samplers - use regular DistributedSampler instead of custom one
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
        
        # Use regular DataLoader with DistributedSampler
        train_loader = DataLoader(
            train_set,
            batch_size=config.data.train_batch_size,
            sampler=train_sampler,
            collate_fn=min_size_pair_collate,
            num_workers=0,  # Set to 0 for DDP to avoid multiprocessing issues
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_set,
            batch_size=config.data.val_batch_size,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=True
        )
        
        if rank == 0:
            logger.info('Train: %d | Validation: %d' % (len(train_set), len(val_set)))

    # Load test datasets based on argument
    test_loaders = {}
    test_sets = {}

    # Determine which test datasets to load
    if args.test_dataset == 'all':
        datasets_to_load = ['test1', 'test2', 'test3', 'casp16_5qa']
    else:
        datasets_to_load = [args.test_dataset]

    for test_name in datasets_to_load:
        test_config = test_configs[test_name]
        try:
            test_set = ComplexInterfaceDataset(
                pdb_dir=os.path.join(config.data.root, test_config['pdb_dir']),
                ilddt_dir=os.path.join(config.data.root, test_config['ilddt_dir']),
                ss_file=os.path.join(config.data.root, test_config['ss_file']),
                rsa_dir=os.path.join(config.data.root, test_config['rsa_dir'])
            )
            
            # Use distributed sampler for test sets
            test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
            test_loader = DataLoader(
                test_set,
                batch_size=config.data.val_batch_size,
                sampler=test_sampler,
                num_workers=0,
                pin_memory=True
            )
            
            test_sets[test_name] = test_set
            test_loaders[test_name] = test_loader
            
            if rank == 0:
                logger.info(f'{test_config["name"]}: {len(test_set)} samples')
                
        except Exception as e:
            if rank == 0:
                logger.warning(f'Failed to load {test_config["name"]}: {str(e)}')
                logger.warning(f'Skipping {test_name}...')

    if not test_loaders:
        if rank == 0:
            logger.error('No test datasets could be loaded!')
        cleanup_ddp()
        exit(1)

    # Model
    if rank == 0:
        logger.info('Building model...')
    
    # MORE ROBUST FIX: Only rank 0 creates model, then broadcast to all ranks
    if rank == 0:
        # Only rank 0 creates the model with fixed seed
        torch.manual_seed(2024)
        np.random.seed(2024)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(2024)
        
        model = PSRNetwork(**config.model)
        model_state_dict = model.state_dict()
        
        logger.info('Model created successfully on rank 0')
        logger.info(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
        
        # Log parameter shapes for debugging
        for name, param in model.named_parameters():
            logger.info(f'Parameter {name}: shape={param.shape}')
        
    else:
        # Other ranks create a dummy model structure (same architecture)
        torch.manual_seed(2024)
        np.random.seed(2024)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(2024)
        
        model = PSRNetwork(**config.model)
        model_state_dict = None
    
    # Synchronize all ranks
    if world_size > 1:
        dist.barrier()
        
        # Broadcast model state from rank 0 to all other ranks
        if rank == 0:
            logger.info('Broadcasting model state to all ranks...')
        
        # Prepare state dict for broadcasting
        state_dict_list = [model_state_dict]
        dist.broadcast_object_list(state_dict_list, src=0)
        
        # Load the broadcasted state dict on all ranks
        model.load_state_dict(state_dict_list[0])
        
        if rank == 0:
            logger.info('Model state broadcasted successfully')
    
    # Move model to device
    model = model.to(rank)
    
    # Apply rank-specific seeds for training diversity
    seed_all(config.train.seed + rank)
    
    # Wrap model with DDP
        # Wrap model with DDP or DataParallel
    if world_size > 1:
        try:
            # Ensure model is on correct device first
            model = model.to(rank)
            
            # List zero-sized parameters for debugging
            zero_params = []
            for name, param in model.named_parameters():
                if param.numel() == 0:
                    zero_params.append(f"{name}: {param.shape}")
            
            if zero_params:
                logger.warning(f"Found {len(zero_params)} zero-sized parameters:")
                for param_info in zero_params:
                    logger.warning(f"  {param_info}")
                logger.warning("These may cause DDP issues, will use DataParallel instead")
                
                # Use DataParallel for models with zero-sized parameters
                # DataParallel requires model on cuda:0
                model = model.to('cuda:0')
                model = nn.DataParallel(model, device_ids=list(range(world_size)))
                logger.info("Using DataParallel due to zero-sized parameters")
            else:
                # Use DDP for normal models
                dist.barrier()
                model = DDP(model, device_ids=[rank], output_device=rank,
                           find_unused_parameters=True)
                logger.info(f'Model successfully wrapped with DDP across {world_size} GPUs')
                
        except Exception as e:
            logger.error(f'DDP setup failed: {e}')
            logger.info('Falling back to DataParallel...')
            
            # Ensure model is on cuda:0 for DataParallel
            model = model.to('cuda:0')
            model = nn.DataParallel(model, device_ids=list(range(world_size)))
            logger.info('Using DataParallel as fallback')
    else:
        logger.info('Single GPU mode - no DDP wrapping')
    
    global_step = Counter()

    # Optimizer (only needed for training)
    if not args.test:
        optimizer = get_optimizer(config.train.optimizer, model)
        # Parse scheduler string configuration
        if isinstance(config.train.scheduler, str):
            # Parse 'type:exp gamma:0.99' format
            scheduler_params = dict(param.split(':') for param in config.train.scheduler.split())
            scheduler_config = EasyDict({
                'type': scheduler_params.get('type'),
                'gamma': float(scheduler_params.get('gamma', 0.99))
            })
        else:
            scheduler_config = config.train.scheduler
        scheduler = get_scheduler(scheduler_config, optimizer)

    # Resume
    it_first = 1
    if args.resume is not None:
        if rank == 0:
            logger.info('Resuming from checkpoint: %s' % args.resume)
        
        ckpt = torch.load(args.resume, map_location=f'cuda:{rank}')
        it_first = ckpt['iteration']
        
        # Load model state
        if world_size > 1:
            model.module.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt['model'])
            
        if not args.test:
            if rank == 0:
                logger.info('Resuming optimizer and scheduler states...')
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])

    def train(it):
        model.train()
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(it)
        
        total_loss_single = 0.0
        total_loss_pair = 0.0
        total_loss_overall = 0.0
        num_batches = 0
        
        # Only show progress bar on main process
        pbar = tqdm(train_loader, desc='Train', position=0, leave=True) if rank == 0 else train_loader
        
        for i, batch in enumerate(pbar):
            batch = batch.to(rank)
            optimizer.zero_grad()
            output = model(batch)   # (Pair*2, )
            target = batch.ilddt  # Use ilddt instead of gdt_ts 
            loss_single = F.huber_loss(output, target)

            output_pair = output.reshape(-1, 2) # (Pair, 2)
            target_pair = target.reshape(-1, 2) # (Pair, 2)
            
            # Standard ranking loss
            diff_pred = output_pair[:,0] - output_pair[:,1]
            diff_target = target_pair[:,0] - target_pair[:,1]
            loss_pair = F.huber_loss(diff_pred, diff_target)
            
            # Simple loss calculation: just loss_single + loss_pair
            loss = loss_single + loss_pair
            
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            
            # Accumulate losses
            total_loss_single += loss_single.item()
            total_loss_pair += loss_pair.item()
            total_loss_overall += loss.item()
            num_batches += 1
            
            # Log on main process only
            if rank == 0:
                log_others = {
                    'grad': orig_grad_norm,
                    'lr': optimizer.param_groups[0]['lr'],
                }
                
                log_losses(EasyDict({'overall': loss, 'single': loss_single, 'pair': loss_pair}), 
                          global_step.step(), 'train', logger=BlackHole(), writer=writer, others=log_others)
        
        # Calculate average losses
        avg_loss_single = total_loss_single / num_batches if num_batches > 0 else 0
        avg_loss_pair = total_loss_pair / num_batches if num_batches > 0 else 0
        avg_loss_overall = total_loss_overall / num_batches if num_batches > 0 else 0
        
        # Gather losses from all processes for accurate averaging
        if world_size > 1:
            avg_loss_single_tensor = torch.tensor(avg_loss_single, dtype=torch.float32, device=rank)
            avg_loss_pair_tensor = torch.tensor(avg_loss_pair, dtype=torch.float32, device=rank)
            avg_loss_overall_tensor = torch.tensor(avg_loss_overall, dtype=torch.float32, device=rank)
            
            dist.all_reduce(avg_loss_single_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_loss_pair_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_loss_overall_tensor, op=dist.ReduceOp.SUM)
            
            avg_loss_single = avg_loss_single_tensor.item() / world_size
            avg_loss_pair = avg_loss_pair_tensor.item() / world_size
            avg_loss_overall = avg_loss_overall_tensor.item() / world_size
        
        if rank == 0:
            logger.info(f"Epoch {it} - Avg Loss: {avg_loss_overall:.4f}, "
                       f"Single: {avg_loss_single:.4f}, Pair: {avg_loss_pair:.4f}")

    def validate(it):
        model.eval()
        y_true, y_pred, targets, decoys = [], [], [], []

        # Set epoch for distributed sampler
        if hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(it)

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validate') if rank == 0 else val_loader
            
            for i, batch in enumerate(pbar):
                batch = batch.to(rank)
                output = model(batch)   # (G, )

                y_pred.extend([v.item() for v in output])
                y_true.extend([v.item() for v in batch.ilddt])
                targets.extend(batch.target_id)
                decoys.extend(batch.decoy_id)

        # Gather results from all processes
        if world_size > 1:
            # Use all_gather_object for easier gathering
            all_y_true = [None for _ in range(world_size)]
            all_y_pred = [None for _ in range(world_size)]
            all_targets = [None for _ in range(world_size)]
            all_decoys = [None for _ in range(world_size)]
            
            dist.all_gather_object(all_y_true, y_true)
            dist.all_gather_object(all_y_pred, y_pred)
            dist.all_gather_object(all_targets, targets)
            dist.all_gather_object(all_decoys, decoys)
            
            if rank == 0:
                # Flatten lists
                y_true = [item for sublist in all_y_true for item in sublist]
                y_pred = [item for sublist in all_y_pred for item in sublist]
                targets = [item for sublist in all_targets for item in sublist]
                decoys = [item for sublist in all_decoys for item in sublist]

        # Only process results on main process
        if rank == 0:
            test_df = pd.DataFrame(
                np.array([targets, decoys, y_true, y_pred]).T,
                columns=['target', 'decoy', 'true', 'pred'],
            )
            
            # Get correlation results
            corrs = report_correlations(test_df, logger, writer, it, prefix='val')
            
            # Calculate global correlations (across all samples)
            y_true_float = test_df['true'].astype(float)
            y_pred_float = test_df['pred'].astype(float)
            
            # Global Pearson correlation
            global_pearson = np.corrcoef(y_true_float, y_pred_float)[0, 1]
            
            # Global Spearman correlation
            global_spearman, _ = spearmanr(y_true_float, y_pred_float)
            
            # Global Kendall correlation
            global_kendall, _ = kendalltau(y_true_float, y_pred_float)
            
            # Calculate MAE (Mean Absolute Error)
            mae = np.mean(np.abs(y_true_float - y_pred_float))
            
            # Calculate top-1 selection error for validation
            per_target_top1_error = test_df.groupby('target').apply(
                lambda x: float(x.loc[x['true'].astype(float).idxmax(), 'true']) - 
                         float(x.loc[x['pred'].astype(float).idxmax(), 'true'])
                if len(x) > 1 else np.nan
            )
            
            per_target_top1_abs_error = per_target_top1_error.abs()
            mean_top1_abs_error = per_target_top1_abs_error.mean()
            
            # Log individual components
            logger.info(f"Validation Global Pearson: {global_pearson:.4f}")
            logger.info(f"Validation Global Spearman: {global_spearman:.4f}")
            logger.info(f"Validation Global Kendall: {global_kendall:.4f}")
            logger.info(f"Validation MAE: {mae:.4f}")
            logger.info(f"Validation top-1 selection error (absolute mean): {mean_top1_abs_error:.4f}")
            
            # Calculate the new selection score
            normalized_mae = min(mae, 1.0)
            normalized_top1_error = min(mean_top1_abs_error, 1.0)
            
            # Calculate components of the selection score
            correlation_component = (global_pearson + global_kendall + global_spearman) / 3.0
            selection_error_component = 1.0 - normalized_top1_error
            mae_component = 1.0 - normalized_mae
            
            # Final selection score
            selection_score = (correlation_component + selection_error_component + mae_component) / 3.0
            
            logger.info(f"Selection Score Components:")
            logger.info(f"  Correlation component (1/3*(P+K+S)): {correlation_component:.4f}")
            logger.info(f"  Selection error component (1-top1): {selection_error_component:.4f}")
            logger.info(f"  MAE component (1-MAE): {mae_component:.4f}")
            logger.info(f"  Final Selection Score: {selection_score:.4f}")
            
            # Write components to tensorboard
            if writer is not BlackHole():
                writer.add_scalar('val/global_pearson', global_pearson, it)
                writer.add_scalar('val/global_spearman', global_spearman, it)
                writer.add_scalar('val/global_kendall', global_kendall, it)
                writer.add_scalar('val/mae', mae, it)
                writer.add_scalar('val/top1_abs_error', mean_top1_abs_error, it)
                writer.add_scalar('val/correlation_component', correlation_component, it)
                writer.add_scalar('val/selection_error_component', selection_error_component, it)
                writer.add_scalar('val/mae_component', mae_component, it)
                writer.add_scalar('val/selection_score', selection_score, it)
            
            return selection_score
        else:
            return 0.0

    def test_single_dataset(test_loader, dataset_name, model_path=None):
        """Evaluate model on a single test dataset"""
        if rank == 0:
            logger.info(f"Evaluating on {dataset_name}...")
        
        # Load best model if specified
        if model_path is not None and rank == 0:
            logger.info(f"Loading model from {model_path}")
            ckpt = torch.load(model_path, map_location=f'cuda:{rank}')
            if world_size > 1:
                model.module.load_state_dict(ckpt['model'])
            else:
                model.load_state_dict(ckpt['model'])
        
        # Set model to evaluation mode
        model.eval()
        
        # Initialize metrics containers
        y_true, y_pred, targets, decoys = [], [], [], []
        total_loss_single = 0.0
        total_samples = 0
        
        # Set epoch for distributed sampler
        if hasattr(test_loader.sampler, 'set_epoch'):
            test_loader.sampler.set_epoch(0)
        
        # Evaluate without gradient tracking
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Testing {dataset_name}') if rank == 0 else test_loader
            
            for batch in pbar:
                batch = batch.to(rank)
                output = model(batch)
                
                y_pred.extend([v.item() for v in output])
                y_true.extend([v.item() for v in batch.ilddt])
                targets.extend(batch.target_id)
                decoys.extend(batch.decoy_id)
                
                # Calculate losses
                target = batch.ilddt
                loss_single = F.huber_loss(output, target)
                total_loss_single += loss_single.item() * len(output)
                total_samples += len(output)
        
        # Gather results from all processes
        if world_size > 1:
            # Gather loss statistics
            total_loss_single_tensor = torch.tensor(total_loss_single, dtype=torch.float32, device=rank)
            total_samples_tensor = torch.tensor(total_samples, dtype=torch.int32, device=rank)
            
            dist.all_reduce(total_loss_single_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            
            total_loss_single = total_loss_single_tensor.item()
            total_samples = total_samples_tensor.item()
            
            # Gather predictions and targets
            all_y_true = [None for _ in range(world_size)]
            all_y_pred = [None for _ in range(world_size)]
            all_targets = [None for _ in range(world_size)]
            all_decoys = [None for _ in range(world_size)]
            
            dist.all_gather_object(all_y_true, y_true)
            dist.all_gather_object(all_y_pred, y_pred)
            dist.all_gather_object(all_targets, targets)
            dist.all_gather_object(all_decoys, decoys)
            
            if rank == 0:
                # Flatten lists
                y_true = [item for sublist in all_y_true for item in sublist]
                y_pred = [item for sublist in all_y_pred for item in sublist]
                targets = [item for sublist in all_targets for item in sublist]
                decoys = [item for sublist in all_decoys for item in sublist]
        
        # Only process results on main process
        if rank == 0:
            # Calculate average losses
            avg_loss_single = total_loss_single / total_samples if total_samples > 0 else 0
            
            # Log losses
            logger.info(f"{dataset_name} ILDDT Loss: {avg_loss_single:.4f}")
            
            # Create results dataframe
            test_df = pd.DataFrame(
                np.array([targets, decoys, y_true, y_pred]).T,
                columns=['target', 'decoy', 'true', 'pred'],
            )
            
            # Calculate per-target correlation
            per_target_corr = test_df.groupby('target').apply(
                lambda x: np.corrcoef(x['true'].astype(float), x['pred'].astype(float))[0,1]
                if len(x) > 1 else np.nan
            )
            logger.info(f"{dataset_name} per-target correlations: {per_target_corr.to_dict()}")
            
            # Calculate per-target top-1 selection error
            per_target_top1_error = test_df.groupby('target').apply(
                lambda x: float(x.loc[x['true'].astype(float).idxmax(), 'true']) - 
                         float(x.loc[x['pred'].astype(float).idxmax(), 'true'])
                if len(x) > 1 else np.nan
            )
            
            per_target_top1_abs_error = per_target_top1_error.abs()
            
            # Log the per-target top-1 error
            logger.info(f"{dataset_name} per-target top-1 selection error (mean): {per_target_top1_error.mean():.4f}")
            logger.info(f"{dataset_name} per-target top-1 selection error (absolute mean): {per_target_top1_abs_error.mean():.4f}")
            
            # Save test predictions to CSV
            if not args.debug and log_dir:
                dataset_prefix = dataset_name.lower().replace(' ', '_')
                csv_filename = f'{dataset_prefix}_predictions.csv'
                csv_path = os.path.join(log_dir, csv_filename)
                test_df.to_csv(csv_path, index=False)
                logger.info(f"{dataset_name} predictions saved to {csv_path}")
            
            # Calculate and report correlations
            logger.info(f"{dataset_name} results:")
            dataset_prefix = dataset_name.lower().replace(' ', '_')
            corrs = report_correlations(test_df, logger, writer, global_step.now, prefix=dataset_prefix)
            
            # Return correlation metrics and losses
            corrs['ilddt_loss'] = avg_loss_single
            corrs['top1_selection_error'] = per_target_top1_error.mean()
            corrs['top1_selection_abs_error'] = per_target_top1_abs_error.mean()
            corrs['dataset_name'] = dataset_name
            
            return corrs
        else:
            return {}

    def test_all_datasets(model_path=None):
        """Evaluate model on all available test datasets"""
        all_results = {}
        
        for test_name, test_loader in test_loaders.items():
            dataset_name = test_configs[test_name]['name']
            if rank == 0:
                logger.info(f"\n{'='*50}")
                logger.info(f"Testing on {dataset_name}")
                logger.info(f"{'='*50}")
            
            try:
                results = test_single_dataset(test_loader, dataset_name, model_path)
                all_results[test_name] = results
                
                # Log summary for this dataset (only on main process)
                if rank == 0 and results:
                    logger.info(f"\n{dataset_name} Summary:")
                    logger.info(f"All Pearson: {np.mean(results['all_pearson']):.4f}")
                    logger.info(f"Per-target Pearson: {np.mean(results['per_target_pearson']):.4f}")
                    logger.info(f"All Spearman: {np.mean(results['all_spearman']):.4f}")
                    logger.info(f"Per-target Spearman: {np.mean(results['per_target_spearman']):.4f}")
                    logger.info(f"Top-1 Selection Error: {results['top1_selection_error']:.4f}")
                    logger.info(f"Top-1 Selection Absolute Error: {results['top1_selection_abs_error']:.4f}")
                
            except Exception as e:
                if rank == 0:
                    logger.error(f"Failed to evaluate on {dataset_name}: {str(e)}")
                continue
        
        # Print overall summary (only on main process)
        if all_results and rank == 0:
            logger.info(f"\n{'='*50}")
            logger.info("OVERALL SUMMARY")
            logger.info(f"{'='*50}")
            
            # Create summary table
            summary_data = []
            for test_name, results in all_results.items():
                if results:  # Check if results is not empty
                    dataset_name = test_configs[test_name]['name']
                    summary_data.append({
                        'Dataset': dataset_name,
                        'All_Pearson': f"{np.mean(results['all_pearson']):.4f}",
                        'PT_Pearson': f"{np.mean(results['per_target_pearson']):.4f}",
                        'All_Spearman': f"{np.mean(results['all_spearman']):.4f}",
                        'PT_Spearman': f"{np.mean(results['per_target_spearman']):.4f}",
                        'Top1_Error': f"{results['top1_selection_abs_error']:.4f}",
                        'ILDDT_Loss': f"{results['ilddt_loss']:.4f}"
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                logger.info("\nSummary Table:")
                logger.info(f"\n{summary_df.to_string(index=False)}")
                
                # Save summary to CSV
                if not args.debug and log_dir:
                    summary_path = os.path.join(log_dir, 'test_summary.csv')
                    summary_df.to_csv(summary_path, index=False)
                    logger.info(f"\nSummary saved to {summary_path}")
        
        return all_results

    try:
        # Skip training if --test flag is provided
        if args.test:
            if rank == 0:
                logger.info("Skipping training, running only test phase...")
            if args.resume:
                if rank == 0:
                    logger.info(f"Testing with checkpoint: {args.resume}")
                test_results = test_all_datasets(args.resume)
            else:
                if rank == 0:
                    logger.warning("No checkpoint provided, testing with initialized model")
                test_results = test_all_datasets()
        
        else:
            # Regular training loop
            if rank == 0:
                logger.info("Starting training for %d epochs..." % config.train.max_epochs)
            
            best_val_loss = float('-inf')
            best_model_path = None
            
            for it in range(it_first, config.train.max_epochs+1):
                if rank == 0:
                    logger.info(f"Epoch {it}/{config.train.max_epochs}")
                
                train(it)
                
                if it % config.train.val_freq == 0:
                    avg_val_metric = validate(it)
                    
                    if rank == 0:
                        logger.info(f"Validation at epoch {it}: avg_val_metric = {avg_val_metric:.4f}")
                        
                        # For plateau-based schedulers
                        scheduler.step(-avg_val_metric)
                        
                        if not args.debug and ckpt_dir:
                            ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                            save_dict = {
                                'config': config,
                                'model': model.module.state_dict() if world_size > 1 else model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'iteration': it,
                                'avg_val_metric': avg_val_metric,
                            }
                            torch.save(save_dict, ckpt_path)
                            
                            # Track best model (higher metric is better)
                            if avg_val_metric > best_val_loss:
                                best_val_loss = avg_val_metric
                                best_model_path = ckpt_path
                                logger.info(f"New best model saved at epoch {it} with validation metric {best_val_loss:.4f}")
            
            # After training is complete, evaluate on all test sets
            if rank == 0:
                logger.info("Training complete. Evaluating on test sets...")
            
            # Use the best model for testing
            if best_model_path and os.path.exists(best_model_path):
                if rank == 0:
                    logger.info(f"Using best model from validation: {os.path.basename(best_model_path)}")
                test_results = test_all_datasets(best_model_path)
            else:
                # Fall back to finding best checkpoint if tracking didn't work
                if not args.debug and ckpt_dir and os.path.exists(ckpt_dir) and rank == 0:
                    ckpt_files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
                    if ckpt_files:
                        # Load each checkpoint to find the one with the best validation metric
                        best_ckpt = None
                        best_val_metric = float('-inf')
                        
                        for ckpt_file in ckpt_files:
                            ckpt = torch.load(ckpt_file, map_location='cpu')
                            if 'avg_val_metric' in ckpt and ckpt['avg_val_metric'] > best_val_metric:
                                best_val_metric = ckpt['avg_val_metric']
                                best_ckpt = ckpt_file
                        
                        if best_ckpt:
                            logger.info(f"Best checkpoint: {os.path.basename(best_ckpt)} (val_metric: {best_val_metric:.4f})")
                            test_results = test_all_datasets(best_ckpt)
                        else:
                            logger.info("No checkpoint with validation metric found, using current model")
                            test_results = test_all_datasets()
                    else:
                        logger.info("No checkpoints found, using current model")
                        test_results = test_all_datasets()
                else:
                    # Debug mode or no checkpoints, just use the final model
                    if rank == 0:
                        logger.info("Using final model for testing")
                    test_results = test_all_datasets()
        
    except KeyboardInterrupt:
        if rank == 0:
            logger.info('Terminating...')
    
    finally:
        # Clean up distributed training
        cleanup_ddp()

if __name__ == '__main__':
    main()