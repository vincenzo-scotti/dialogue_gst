import os
import sys
from shutil import copy2, move
import logging
from datetime import datetime
from argparse import ArgumentParser, Namespace
import yaml
from typing import Optional, Union, Tuple, List, Dict, Callable

import random
from itertools import product
import numpy as np
from gsttransformer.misc import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from gsttransformer.data import GSTTCorpus, DataSetSplit
from transformers import GPT2Tokenizer, GPT2Model
from mellotron_api import load_tts

from gsttransformer.model import GSTTransformer
# TODO add fine tuning script
# TODO add LR plot
# TODO add warm restart


# Variables to control gpt2 and optimization parameters
# Environment
# Global
random_seed: Optional[int]
device: torch.device
mixed_precision: bool = True
writer: SummaryWriter
training_configs: List[str]
config_id: Optional[str] = None
global_step_idx: int = 0
losses_list: List[Tuple[int, Dict]] = list()
# Step
best_validation_score: float = float('inf')
# Models
# Steps
model_config_map: Dict[str, Dict] = dict()
# Current
model_configs: Dict = dict()
gpt2: GPT2Model = None
tokenizer: GPT2Tokenizer = None
mellotron: Tuple = None
gstt: GSTTransformer = None
# Data
# Steps
corpus_config_map: Dict[str, Dict] = dict()
# Current
corpus_configs: Dict = dict()
corpora: Dict[DataSetSplit, GSTTCorpus] = dict()
corpus_loaders: Dict[DataSetSplit, DataLoader] = dict()
# Optimisation
# Steps
loss_configs_map: Dict[str, Dict] = dict()
optimizer_configs_map: Dict[str, Dict] = dict()
lr_scheduler_configs_map: Dict[str, Optional[Dict]] = dict()
evaluation_configs_map: Dict[str, Dict] = dict()
# Current
loss_configs: Optional[Dict]
optimizer_configs: Dict
optimizer: Optimizer
scaler: Optional[GradScaler] = None
lr_scheduler_configs: Optional[Dict] = None
lr_scheduler: Optional[LinearLR] = None
evaluation_configs: Dict = dict()
# Experiment dir path
current_experiment_dir_path: str
# Checkpoint path
# Steps
model_checkpoint_path_map: Dict[str, str] = dict()
best_model_checkpoint_path_map: Dict[str, str] = dict()
# Current
model_checkpoint_path: str
best_model_checkpoint_path: str


def init_environment(config_file_path: str):
    # Declare global variables
    global random_seed, device, mixed_precision, writer, training_configs
    global model_config_map, corpus_config_map, loss_configs_map, \
        optimizer_configs_map, lr_scheduler_configs_map, evaluation_configs_map
    global current_experiment_dir_path, model_checkpoint_path_map, best_model_checkpoint_path_map

    # Define helper function to create directory if not exits
    def mkd(path: str) -> str:
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    # Get date-time
    date_time_experiment: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Read YAML file
    with open(config_file_path) as f:
        configs_dump_str: str = f.read()
        f.seek(0)
        configs: Dict = yaml.full_load(f)
    # Get list of selected steps
    training_configs = [config for config in configs['training_configs']]
    # Create directories
    # Main
    experiments_dir_path: str = mkd(configs['experiments_directory_path'])
    experiment_series_dir_path: str = mkd(os.path.join(experiments_dir_path, configs['experiment_series']))
    current_experiment_dir_path = mkd(os.path.join(
        experiment_series_dir_path, f"{configs['experiment_id']}_{date_time_experiment}"
    ))
    # Model
    model_dir_path: str = mkd(os.path.join(current_experiment_dir_path, 'model'))
    tmp_subdir = ((model_checkpoint_path_map, 'latest'), (best_model_checkpoint_path_map, 'best'))
    for config in training_configs:
        for dir_map, prefix in tmp_subdir:
            dir_map[config] = mkd(os.path.join(model_dir_path, f'{prefix}_checkpoint_{config}'))
    # Logging
    # Tensorboard
    tb_dir_path = mkd(os.path.join(current_experiment_dir_path, 'tensorboard'))
    # Create file paths
    if configs.get('log_file', False):
        log_file_path = os.path.join(
            current_experiment_dir_path, f"{configs['experiment_id']}_{date_time_experiment}.log"
        )
    else:
        log_file_path = None
    configs_dump_path = os.path.join(current_experiment_dir_path, 'configs.yaml')
    # Init logging
    logging.basicConfig(filename=log_file_path, level=configs['log_level'])
    # Start Logging info
    logging.info(f"{configs['experiment_series']} training script started")
    logging.info(f"Current experiment directories created at '{current_experiment_dir_path}'")
    if log_file_path is not None:
        logging.info(f"Current experiment log created at '{log_file_path}'")
    # Set all random seeds
    random_seed = configs.get('random_seed', None)
    logging.info("Random seeds set")
    # Tensor-Board writer
    writer = SummaryWriter(tb_dir_path)
    logging.info(f"Tensor-Board writer created at '{tb_dir_path}'")
    # Dump configs
    copy2(config_file_path, configs_dump_path)
    writer.add_text('Configs', f"<pre>{configs_dump_str}</pre>")
    logging.info(f"Current experiment configuration dumped at '{configs_dump_path}'")
    # Set device
    device = torch.device(configs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Device set to '{device}'")
    # Set mixed precision
    mixed_precision = configs.get('mixed_precision', mixed_precision)
    logging.info(f"Mixed precision set to '{mixed_precision}'")
    # Load remaining configs
    for config in training_configs:
        model_config_map[config] = configs[config]['model']
        corpus_config_map[config] = configs[config]['data']
        loss_configs_map[config] = configs[config].get('loss', dict())
        optimizer_configs_map[config] = configs[config]['optimizer']
        lr_scheduler_configs_map[config] = configs[config].get('lr_scheduler')
        evaluation_configs_map[config] = configs[config].get('evaluation', dict())
    logging.info("Initialisation completed")


def clean_environment():
    # Declare global variables
    global current_experiment_dir_path
    # TODO search for closest in time
    # List files
    file_list = sorted(f for f in os.listdir() if f.startswith("experiment_"))
    if len(file_list) > 0:
        output_file = file_list.pop(-1)
        move(output_file, os.path.join(current_experiment_dir_path, output_file))
        logging.info("Cleaning completed")


def init_model():
    # Declare global variables
    global gpt2, tokenizer, mellotron, gstt
    # Create GPT2 and Tokenizer instance
    gpt2 = GPT2Model.from_pretrained(model_configs['lm']['gpt2']).eval().to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_configs['lm']['tokenizer'], pad_token='<|endoftext|>')
    logging.info(f"GPT2 and Tokenizer instantiated and moved to device: {device}")
    # Create mellotron instance (if not already available)
    mellotron = load_tts(model_configs['tts'], device=torch.device('cpu')) if mellotron is None else mellotron
    logging.info("Mellotron instantiated")
    # Create GSTT instance
    gstt = GSTTransformer(
        gpt2.config,
        mellotron[0].gst.stl.attention.num_units,
        (mellotron[0].gst.stl.attention.num_heads, mellotron[0].gst.stl.embed.size(0))
    ).to(device)
    logging.info(f"GST-Transformer instantiated and moved to device: {device}")


def init_data_loaders():
    # Declare global variables
    global corpora, corpus_loaders, gpt2
    # Init corpora and loaders dict
    corpora = dict()
    corpus_loaders = dict()
    # Iterate over splits
    for split in corpus_configs['splits']:
        # Create data set instance
        data_set: GSTTCorpus = GSTTCorpus(
            corpus_configs['corpora_dir_path'],
            gpt2,
            tokenizer,
            mellotron,
            split,
            corpus_configs['cache_dir_path'],
            corpus_configs['encoding_mode'],
            device=device,
            mixed_precision=mixed_precision,
            **corpus_configs.get('kwargs', dict())
        )
        logging.info(f"{split.capitalize()} data set instantiated")
        # Add created data set and data loader to dict
        corpora[DataSetSplit(split)] = data_set
        logging.info(f"{split.capitalize()} data set added to dictionary")
        # Create data loader instance
        data_loader: DataLoader = DataLoader(
            data_set,
            batch_size=corpus_configs['splits'][split]['mini_batch_size'],
            num_workers=corpus_configs['splits'][split]['n_workers'],
            shuffle=DataSetSplit(split) == DataSetSplit.TRAIN,
            collate_fn=data_set.collate
        )
        logging.info(f"{split.capitalize()} data loader instantiated")
        # Add created data loader to dict
        corpus_loaders[DataSetSplit(split)] = data_loader
        logging.info(f"{split.capitalize()} data loader added to dictionary")
    logging.info("All data loaders instantiated")
    # Delete langauge model
    del gpt2
    for split in corpora:
        del corpora[split].gpt2
    logging.info("Language model removed")


def init_optimisation_tools():
    # Declare global variables
    global lr_scheduler_configs, optimizer, lr_scheduler, scaler
    # Create optimiser instance
    optimizer = torch.optim.AdamW(
        params=gstt.parameters(), **optimizer_configs['kwargs']
    )
    logging.info(f"Optimiser instantiated (ID: {config_id})")
    # Create learning rate scheduler instance if required
    if lr_scheduler_configs is not None:
        # Get total number of misc steps
        steps = len(corpus_loaders[DataSetSplit('train')]) * optimizer_configs['n_epochs'] + 1
        # Update learning rate scheduler configs with missing info
        lr_scheduler_configs['lr_steps'] = steps
        lr_scheduler = LinearLR(optimizer, **lr_scheduler_configs)
        logging.info(f"Learning rate scheduler instantiated (ID: {config_id})")
    else:
        lr_scheduler = None

    # Create scaler if using mixed precision
    if mixed_precision:
        scaler = GradScaler()
        logging.info("Gradient scaler for mixed precision instantiated")


# Define helper function set environment of current training step
def set_env():
    global global_step_idx, best_validation_score, losses_list
    global model_configs, corpus_configs, optimizer_configs, loss_configs, lr_scheduler_configs, evaluation_configs
    global model_checkpoint_path, best_model_checkpoint_path

    logging.info(f"Setting environment for training (ID: {config_id})")
    # Random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    logging.info("Random seeds set")

    # Reset accumulators
    global_step_idx = 0
    best_validation_score = float('inf')
    losses_list = list()
    # Set configs
    model_configs = model_config_map[config_id]
    corpus_configs = corpus_config_map[config_id]
    loss_configs = loss_configs_map[config_id]
    optimizer_configs = optimizer_configs_map[config_id]
    lr_scheduler_configs = lr_scheduler_configs_map[config_id]
    evaluation_configs = evaluation_configs_map[config_id]
    # Checkpoint paths
    model_checkpoint_path = model_checkpoint_path_map[config_id]
    best_model_checkpoint_path = best_model_checkpoint_path_map[config_id]
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_mini_batch(
        split: str, input_embeds, attention_mask, gst_embeddings, gst_scores
):
    # Declare global variables
    # Compute helper params
    in_mem: int = corpus_configs['splits'][split]['in_mem']
    # Loss accumulators
    mse_loss = torch.tensor(0., device=device) if gstt.training else torch.empty(0, device=device)
    kl_div_loss = torch.tensor(0., device=device) if gstt.training else torch.empty(0, device=device)
    # Number elements in batch
    n_samples = input_embeds.size(0)
    # Losses weights
    w_mse = loss_configs.get('mse_weight', 1.0)
    w_kl = loss_configs.get('kl_weight', 1.0)

    # Move tensors to devices
    input_embeds = input_embeds.to(device)
    attention_mask = attention_mask.to(device)
    gst_embeddings = gst_embeddings.to(device)
    gst_scores = gst_scores.to(device)

    logging.debug('Processing mini-batch')
    # Loop over sub_batches to fit in memory
    idxs = ((idx, min(n_samples, idx + in_mem)) for idx in range(0, n_samples, in_mem))
    for s_idx, e_idx in idxs:
        with torch.autocast(device.type, enabled=mixed_precision):
            # Process current elements
            outputs = gstt(input_embeds[s_idx:e_idx], attention_mask[s_idx:e_idx])
            # Compute losses
            if gstt.training:
                tmp_mse_loss = F.mse_loss(outputs['gst_embeds'], gst_embeddings[s_idx:e_idx])
                tmp_kl_div_loss = F.kl_div(
                    F.log_softmax(outputs['gst_scores'], -1).view(-1, gst_scores.size(-1)),
                    gst_scores[s_idx:e_idx].log().view(-1, gst_scores.size(-1)),
                    reduction='batchmean',
                    log_target=True
                )
            else:
                tmp_mse_loss = F.mse_loss(outputs['gst_embeds'], gst_embeddings[s_idx:e_idx], reduction='none').mean(-1)
                tmp_kl_div_loss = F.kl_div(
                    F.log_softmax(outputs['gst_scores'], -1),
                    gst_scores[s_idx:e_idx].log(),
                    reduction='none',
                    log_target=True
                ).sum(-1).mean(1)
            # Scale loss if using gradient accumulation (only in training)
            if gstt.training:
                tmp_mse_loss = tmp_mse_loss * (e_idx - s_idx) / n_samples
                tmp_kl_div_loss = tmp_kl_div_loss * (e_idx - s_idx) / n_samples

                tmp_loss = w_mse * tmp_mse_loss + w_kl * tmp_kl_div_loss
        # Compute gradients if gpt2 is training
        if gstt.training:
            # Compute gradients
            if scaler is not None:
                scaler.scale(tmp_loss).backward()
            else:
                tmp_loss.backward()
        # Accumulate losses
        if gstt.training:
            mse_loss += tmp_mse_loss.detach()
            kl_div_loss += tmp_kl_div_loss.detach()
        else:
            mse_loss = torch.hstack([mse_loss, tmp_mse_loss])
            kl_div_loss = torch.hstack([kl_div_loss, tmp_kl_div_loss])
    # Compute total loss
    loss = (w_mse * mse_loss) + (w_kl * kl_div_loss)
    logging.debug('Processing complete')

    if gstt.training:
        # Clip gradient norm
        if optimizer_configs['max_gradient_norm'] > 0.0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(gstt.parameters(), max_norm=optimizer_configs['max_gradient_norm'])
        # Update weights
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        # Reset optimiser and gpt2 gradients
        for param in gstt.parameters():
            param.grad = None

    # Generate losses dict
    losses = {
        'MSE loss': mse_loss, 'KL-Divergence loss': kl_div_loss, 'Loss': loss
    }

    return losses


@torch.no_grad()
def process_evaluation(
        split: str,
        tag: str,
        best_validation_score: Optional[float] = None
):

    # Initialize validation accumulator
    validation_loss = {
        'MSE loss': torch.empty(0, device=device),
        'KL-Divergence loss': torch.empty(0, device=device),
        'Loss': torch.empty(0, device=device)
    }
    # Iterate over validation mini batches
    for b_idx, mini_batch in enumerate(corpus_loaders[DataSetSplit(split)]):
        # Process current mini-batch
        tmp_losses_dict = process_mini_batch(split, *mini_batch)
        # Update accumulator
        for loss in validation_loss:
            validation_loss[loss] = torch.cat([validation_loss[loss], tmp_losses_dict[loss]])
    # Accumulate values
    validation_loss = {key: values.mean() for key, values in validation_loss.items()}
    # Log values
    for key, value in validation_loss.items():
        writer.add_scalar(f'{config_id.upper()}/{key}/{tag}', value.cpu().item(), global_step=global_step_idx)

    # If this is the standard validation process check for best gpt2
    if best_validation_score is not None:
        if validation_loss['Loss'] <= best_validation_score:
            # Save GSTT state dictionary
            torch.save(gstt.state_dict(), os.path.join(best_model_checkpoint_path, 'gstt.pt'))
            # Update best score
            best_validation_score = validation_loss['Loss']
            # Log update
            logging.info("Validation objective improved, GSTT model checkpoint triggered")

        return best_validation_score, validation_loss
    # Else do the final report
    else:
        output_report = f"Evaluation (split: {split})\n" \
                        f"\tLoss: {validation_loss['Loss']:.4f}\n" \
                        f"\tMSE loss: {validation_loss['MSE loss']:.4f}\n" \
                        f"\tKL-Divergence loss: {validation_loss['KL-Divergence loss']:.4f}\n"
        writer.add_text(f'{config_id.upper()}/Final report/{split}', f"<pre>{output_report}</pre>")

        return output_report


def fit_model():
    # Declare global variables
    global global_step_idx

    # Define helper function to lo accumulated into
    def log_training():
        # Log info (mini-batch level) on Tensorboard
        for idx, tmp_losses_dict in losses_list:
            for key, value in tmp_losses_dict.items():
                writer.add_scalar(f'{config_id.upper()}/{key}/Training', value.cpu().item(), global_step=idx)

    # Define helper function to run training step
    def training_loop():
        global global_step_idx

        # Initialise variables
        n_epochs = optimizer_configs.get('n_epochs', 0)
        validation_period = evaluation_configs.get('validation_period', len(corpus_loaders[DataSetSplit('train')]))
        logging_period = evaluation_configs.get('logging_period', validation_period)
        # Run initial validation step
        evaluation_step()
        # Update validation counter
        # Loop over epochs
        for epoch in range(n_epochs):
            logging.info(f"Epoch {epoch + 1}/{n_epochs} started")
            # Iterate over mini-batches
            for b_idx, mini_batch in enumerate(corpus_loaders[DataSetSplit('train')]):
                # Process current mini-batch
                mini_batch_losses_dict = process_mini_batch('train', *mini_batch)
                losses_list.append((global_step_idx + 1, mini_batch_losses_dict))
                logging.info(f"Training mini-batch {b_idx + 1}/{len(corpus_loaders[DataSetSplit('train')])} done")
                # Update global step counter and step counter
                global_step_idx += 1
                # Check if training is completed
                training_completed = (epoch == n_epochs - 1) and (b_idx == len(corpus_loaders[DataSetSplit('train')]) - 1)
                # Log loss if required
                if global_step_idx % logging_period == 0 or training_completed:
                    # Call logging step
                    log_training()
                    # Clear accumulator
                    losses_list.clear()
                # Do validation step if required
                if global_step_idx % validation_period == 0 or training_completed:
                    # Run validation step
                    evaluation_step()
            # Save end of epoch checkpoint
            torch.save(gstt.state_dict(), os.path.join(model_checkpoint_path, 'gstt.pt'))
            # Log end of epoch
            logging.info(f"Epoch {epoch + 1}/{n_epochs} finished")

    # Define helper function to call validation
    def evaluation_step():
        global best_validation_score

        # Log start of validation
        logging.info(f"Validation started")
        # Set gpt2 in evaluation mode
        gstt.eval()
        logging.info("GSTT model set in evaluation mode")
        # Do validation step
        best_validation_score, validation_loss = process_evaluation(
            'validation',
            'Validation',
            best_validation_score=best_validation_score
        )
        # Log end of validation
        logging.info(f"Validation completed - Loss {validation_loss['Loss']:.4f}")
        # Set GSTT back in training mode
        gstt.train()
        logging.info("GSTT model set in training mode")

    # Train and validation process
    # Get current date and time
    start_time: datetime = datetime.now()
    # Log start of misc
    logging.info(f"Training started - Current date and time {start_time} - (ID {config_id})")
    # Set GSTT in training mode
    gstt.train()
    # Log start of specific training step
    logging.info(f"Training started")
    # Run training step  #NOTE unused training steps will be simply skipped
    training_loop()
    # Log end of specific training step
    logging.info(f"Training finished - (ID {config_id})")
    # Restore best validation gpt2 weights
    gstt.load_state_dict(torch.load(os.path.join(best_model_checkpoint_path, 'gstt.pt')))
    gstt.to(device)
    logging.info("Best validation GSTT weights restored")
    # Close training
    # Get current date and time
    end_time: datetime = datetime.now()
    # Log end of training
    logging.info(f"Training finished - Current date and time {end_time} - (ID {config_id})")
    logging.info(f"Elapsed time {end_time - start_time}")


def evaluate_model():
    # Declare global variables
    global writer, gstt
    # Start evaluation
    # Get current date and time
    start_time: datetime = datetime.now()
    # Log start of evaluation
    logging.info(f"Evaluation started - Current date and time {start_time} - (ID {config_id})")
    # Set gpt2 in evaluation mode
    gstt.eval()
    logging.info(f"GSTT model set in evaluation mode")
    # Log start on validation set
    logging.info(f"Validation set evaluation started")
    # Compute summary report on validation set
    validation_report: str = process_evaluation('validation', 'Final evaluation (validation set)')
    # Log end on validation set
    logging.info(f"Validation set evaluation finished")
    logging.info('\n' + validation_report)
    # Print test results
    print(validation_report)
    # Log test results in TensorBoard
    writer.add_text(f'{config_id.upper()}/Validation set evaluation results', validation_report)
    # Log start on test set
    logging.info(f"Test set evaluation started")
    # Compute summary report on test set
    test_report: str = process_evaluation('test', 'Final evaluation (test set)')
    # Log end on test set
    logging.info(f"Test set evaluation finished")
    logging.info('\n' + test_report)
    # Print test results
    print(test_report)
    # Log test results in TensorBoard
    writer.add_text(f'{config_id.upper()}/Test set evaluation results', test_report)
    # Close evaluation
    # Get current date and time
    end_time: datetime = datetime.now()
    # Log end of evaluation
    logging.info(f"Evaluation finished - Current date and time {end_time} - (ID {config_id})")
    logging.info(f"Elapsed time {end_time - start_time}")
    # Remove trained model
    del gstt


def main(args: Namespace):
    global config_id
    # Perform preparation steps
    # Prepare the environment
    init_environment(args.config_file_path)
    # Iterate over training configs
    for config_id in training_configs:
        # Set environmental configs
        set_env()
        # Create gpt2 and tokeniser
        init_model()
        # Create data sets and data loaders
        init_data_loaders()
        # Create optimiser, scaler and scheduler
        init_optimisation_tools()
        # Training and validation process
        fit_model()
        # Testing process
        evaluate_model()
    # Clean misc output if any
    clean_environment()

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser()
    # Add arguments to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str,
        help="Path to the YAML file containing the configuration for the experiment."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
