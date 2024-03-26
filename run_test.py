import hydra
from torch import set_float32_matmul_precision
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer

from pytorch_lightning import Trainer
from pipeline import FPM_Pipeline
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler


@hydra.main(version_base=None, config_path="configs", config_name="config")
def eval_pipeline(args: DictConfig) -> None:

    set_float32_matmul_precision('medium')
    system = FPM_Pipeline(args, test_model=True)
    
    if args.profiler:
        profiler = AdvancedProfiler(args.tmp.logs_out,'profiler_log')
    else:
        profiler = None


    trainer = Trainer(
        logger = None,
        profiler=profiler,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        callbacks=system.get_callbacks(),
        accelerator=args.device,
        log_every_n_steps=10,
        devices='auto',
        num_nodes=1,
        check_val_every_n_epoch=args.training.freq_valid,
        max_epochs=args.training.n_epochs
    )
    print(args.exp_tag)
    print("TEST CONFIGURATION:")
    for k,v in args.test.items():
        print(f'{k} : {v}')
    print("Loading checkpoint ", args.eval.ckpt)    
    test_data = system.get_test_dataloader()
    trainer.test(system, test_data, ckpt_path=args.eval.ckpt)

if __name__ == '__main__':
    
    eval_pipeline()