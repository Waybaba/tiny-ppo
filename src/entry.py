import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
import numpy as np
import hydra
import wandb
import utils


import warnings
warnings.filterwarnings('ignore')

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="train.yaml")	
def main(cfg):
	# print config
	utils.print_config_tree(cfg, resolve=True)
	# wandb (before instantiating cfg)
	wandb.init(project=cfg.task_name, tags=cfg.tags, config=utils.config_format(cfg),dir=cfg.output_dir)
	# initialize hydra cfg
	cfg = hydra.utils.instantiate(cfg)
	cfg.runner().start(cfg)

	
if __name__ == "__main__":
	main()
