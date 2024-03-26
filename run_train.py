import os
import hydra
from torch import set_float32_matmul_precision
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from utils.misc import init_storage_folders
from os.path import join, exists
from pipeline import FPM_Pipeline
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler

def get_ckpt_path(args):
    '''
    Get checkpoint path from folder of previous experiment
    '''
    ckpt_file = 'epoch={:04d}.ckpt'.format(int(args.checkpoint))
    ckpt_path = join(args.exp_root, args.exp_name, 'models', ckpt_file)

    print('Resuming from checkpoint at {}'.format(ckpt_path))

    return ckpt_path

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_pipeline(args : DictConfig) -> None:

    set_float32_matmul_precision('medium')
    
    error = False
    model_path = None
    if args.resume_ckpt != 'none':
        ckpt_n = int(args.resume_ckpt)
        model_path = join('exp_data',args.exp_name,'models',f'epoch={ckpt_n:04d}.ckpt')
        config_path = join('exp_data',args.exp_name,'config.yaml')
        if not os.path.exists(config_path):
            print(f"Experiment {args.exp_name} does not exist.")
            error = True
        elif not os.path.exists(model_path):
            print(f"Checkpoint {ckpt_n} not present for experiment {args.exp_name}.")
            error = True
        if not error:
            args = OmegaConf.load(config_path)
            args.resume_ckpt = ckpt_n
            print(f"Resuming from {model_path}, be careful with config.json compatibility!")

    if args.resume_ckpt == 'none' or error:
        if error:
            model_path=None
            print("Starting new experiment.")
        checkpoint_out, logs_out, results_out = init_storage_folders(args, 2)    
        
        args.tmp.logs_out = logs_out
        args.tmp.ckpt_out = checkpoint_out
        args.tmp.results_out = results_out
    
    system = FPM_Pipeline(args, test_model=False)
    if args.profiler:
        profiler = AdvancedProfiler(args.tmp.logs_out,'profiler_log')
    else:
        profiler = None

    strategy = 'ddp_find_unused_parameters_false'  

    trainer = Trainer(
        logger = system.get_logger(),
        profiler=profiler,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        callbacks=system.get_callbacks(),
        accelerator=args.device,
        strategy=strategy,
        log_every_n_steps=10,
        devices='auto',
        num_nodes=1,
        detect_anomaly=False,
        check_val_every_n_epoch=args.training.freq_valid,
        max_epochs=args.training.n_epochs
    )

    train_data = system.get_train_dataloader()
    valid_data = system.get_valid_dataloader()
    test_data = system.get_test_dataloader()
    
    trainer.fit(
        system, 
        train_dataloaders=train_data, 
        val_dataloaders=valid_data,
        ckpt_path=model_path
    )

    trainer.test(system, test_data, 'last')

if __name__ == '__main__':
    run_pipeline()
