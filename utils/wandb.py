import wandb

def init_wandb(cfg):
	wandb.init(project="RL", config=cfg)
	return wandb