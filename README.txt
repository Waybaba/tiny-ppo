Code logic:

python src/entry.py \
	experiment=sac

# time flow
src/entry.py -> configs/train.yaml 
-> configs/experiment/sac.yaml
-> src/entry.py
-> src/runner.py-XXXRunner().start(cfg)


# How does the code work?
The main function in entry.py is linked to configs/train.yaml.
configs/train.yaml is almost empty, which is designed as an 
template as hydra entry function needs a fixed yaml file to start.
The main configuration is in configs/experiment/***.yaml, which is
specified by the experiment argument `experiment=sac` in the command.
Then, going back to entry.py, the main function will use the cfg to
src/runner.py-XXXRunner().start(cfg) to start the training process

ps. entry is consitent for all project, while runner is for some 
algorithm, which would setup env, policy, etc.

ps. we use different runner as the init process of different algorithm
is different, e.g. SACRunner would setup actor and critic.
