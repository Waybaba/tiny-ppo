import wandb
wandb.init(project="RL")

def init_wandb(cfg):
	wandb.init(project="RL", config=cfg)
	return wandb