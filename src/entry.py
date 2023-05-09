import os
import warnings
import random
import string
import datetime

import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
import hydra
import wandb
import utils
from time import sleep
from rich.console import Console

console = Console()

warnings.filterwarnings('ignore')

def initialize_wandb(cfg):
	# Generate a unique directory name with a timestamp and 10 random characters
	timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
	unique_dir = f"{timestamp}-{random_chars}"

	# Create unique wandb directory
	wandb_dir = os.path.join("./output", unique_dir) if cfg.wandb.buf_dir else cfg.output_dir

	os.makedirs(wandb_dir, exist_ok=True)

	wandb.init(
		project=cfg.task_name,
		tags=cfg.tags,
		config=utils.config_format(cfg),
		dir=wandb_dir,
		mode=cfg.wandb.mode
	)
	return wandb_dir

def move_output_to_wandb_dir(src_dir, dest_dir):
	print("\n\n###")
	print("Moving output to wandb dir ...")
	print(f"From: {src_dir}")
	print(f"To: {dest_dir}")
	utils.copy_all_files(src_dir, dest_dir)
	print("Moving wandb done!")


@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="train.yaml")	
def main(cfg):
	# Print and set up the config
	utils.print_config_tree(cfg, resolve=True)

	print("Initializing wandb ...")
	wandb_dir = initialize_wandb(cfg)

	print("Initializing and running Hydra config ...")
	cfg = hydra.utils.instantiate(cfg)

	print("Initializing and running runner ...")
	cfg.runner().start(cfg)

	# Move output to wandb dir if necessary
	if cfg.wandb.buf_dir:
		retry = 10
		time_to_sleep = 5
		for i in range(retry):
			try:
				move_output_to_wandb_dir(wandb_dir,cfg.output_dir)
				break
			except Exception as e:
				print(f"Failed to move output to wandb dir. Retrying ({i+1}/{retry}) ...")
				print(e)
				sleep(time_to_sleep)

	print("Done!")

if __name__ == "__main__":
	main()
