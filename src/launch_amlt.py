"""

Use command line to launch a amlt sweep job.

Usage:
	run cmd line in shell (see template below)
	>>> read from configs/amlt/template.yaml
	>>> fill content based on template and command line input
	>>> replace the default file in configs/amlt/latest.yaml
	>>> output about the sweeped parameters and job numbers, file directory and other informations
	>>> enter y to launch

	


=========================== command >>>>>>>>>>>>>>>>

Example:
python src/launch_amlt.py \
	amlt.search.job_template.sku=G1 \
	python src/entry.py \
	-m \
	hydra/launcher=wsl_parallel \
	++trainer.log_upload_interval=100000 \
	n_jobs=5 \
	seed=#1,2,3,4# \
	env.delay=0,1,2,4,8 \
	tags="test_tag"


Explaination:
python launch_amlt.py \ # start
	amlt.search.job_template.sku=G1 \ # start with amlt would change the template key
	python src/entry.py -m \ all single options without [] would be copied
	hydra/launcher=wsl_parallel \
	++trainer.log_upload_interval=100000 \
	n_jobs=5 \
	seed=#1,2,3,4# \ # these in #xxx# would be kept as multiple in the final version to support parallel
	env.delay=0,1,2,4,8 \
	tags="test_tag"

	
The arguments has following types:
	1. start with amlt.: would change the value in the template
	2. python, entry file .py: should follows the type 1
	3. contains no "," and "#": would keep the same in the template
	4. contains "," but no "#": hparams, would be sweeped
	5. contains "," and "#": parallel key, # would be removed and the remaining would be same in template


=========================== command <<<<<<<<<<<<<<<<
	
would generate the following config file

=========================== configs/amlt/latest.yaml >>>>>>>>>>>>>>>>
description: AMLT
target:
  service: singularity
  name: msrresrchvc
environment: ...
code: ...
...
search:
	type: grid
	max_trials: 1000
	job_template:
		name: RL_Delayed_{experiment_name:s}_{auto:5s}
		sku: G1
		command:
			- python src/entry.py -m hydra/launcher=wsl_parallel ++trainer.log_upload_interval=100000 n_jobs=5 seed={seed} env.delay={env_delay} tags="test_tag"
	params:
	- name: env_delay
	  values: [0,1,2,4,8]
	- name: seed
	  values: [1,2,3,4]

=========================== configs/amlt/latest.yaml <<<<<<<<<<<<<<<<

"""
import yaml
import re
import sys
import os
import itertools
import random
import string
import subprocess


AMLT_CONFIG_TEMPLATE = """
description: AMLT

target:
  service: singularity
  name: msrresrchvc    # more GPUs 

environment:
  image: waybaba/rl:v3
  username: waybaba
  setup:
    - echo "setup start..."
    - export UPRJDIR=/mnt/default/
    - export UDATADIR=/mnt/storage/data
    - export UOUTDIR=/mnt/storage/output
    - mkdir -p /mnt/storage/output /mnt/storage/data
    - echo "setup finished!"


code:
  local_dir: $CONFIG_DIR/../../

storage:
  input:
    storage_account_name: resrchvc4data
    container_name: v-wangwei1
    mount_dir: /mnt/storage
    local_dir: /home/v-wangwei1/storage

search:
  job_template:
    name: RL_Delayed_{experiment_name:s}_{auto:5s}
    # sku: 24G1-P40
    sku: G1
    command:
    - python src/entry.py
      -m
      hydra/launcher=wsl_parallel
      ++trainer.log_upload_interval=100000
      trainer.progress_bar=false
      trainer.max_epoch=200
      trainer.step_per_epoch=5000
      n_jobs={n_jobs}
      seed={seed}
      env.name={env_name}
      env.delay={env_delay}
      experiment={experiment}
      global_cfg.critic_input.obs_type={critic_input_obs_type}
      global_cfg.actor_input.obs_type={actor_input_obs_type}
      global_cfg.actor_input.history_num={actor_input_history_num}
      global_cfg.actor_input.trace_direction={actor_input_trace_direction}
      buffer.size={buffer_size}
      tags=[{tag}]
  type: grid
  max_trials: 10000
  params:
    - name: env_delay
      values: [0,1,2,4,8]
      # values: [4,2,1,8]
      # values: ["0,2","4,8","16,32"]
      # values: ["0,2,4,8,16,32"]
      # values: [4,1,2,8,12,0]
    - name: env_name
      # values: [HalfCheetah-v4]
      # values: [Hopper-v4,HalfCheetah-v4]
      values: [Hopper-v4,HalfCheetah-v4,Ant-v4,Walker2d-v4]
    - name: experiment
      values: [td3_rnn]
    - name: buffer_size
      values: [1000000]
    - name: critic_input_obs_type
      values: [oracle]
    - name: actor_input_obs_type
      values: [oracle,normal]
    - name: actor_input_trace_direction
      values: [next]
    - name: actor_input_history_num
      values: [2,4,8]
    - name: seed
      values: ["0,1,2,3"]
    - name: n_jobs
      values: [5]
    - name: tag
      values: ["custom_td3_rnnOracleDebug_v2"] 


"""

def execute_command(command_str):
    command_args = command_str.split()
    try:
        process = subprocess.Popen(command_args)

        # Wait for the process to complete
        process.communicate()

        if process.returncode == 0:
            print("\nCommand executed successfully")
        else:
            print("\nCommand failed with return code:", process.returncode)
    except KeyboardInterrupt:
        print("\nExecution interrupted by the user")

def replace_with_dot_keys(source_dict, dot_keys):
	"""
	replace the keys in source dict with dot_keys
	e.g. 
		source_dict:
			{a:{b:10,d:20},c:2}
		dot_keys:
			{
				a.b: 3
				c: 1
			}
		return: 
			{a:{b:3,d:20},c:1}
	"""
	def set_nested_value(d, key, value):
		keys = key.split('.')
		for k in keys[:-1]:
			d = d.setdefault(k, {})
		d[keys[-1]] = value

	for dot_key, value in dot_keys.items():
		set_nested_value(source_dict, dot_key, value)

	return source_dict


class CmdJob:
	def __init__(self, entry_file, args, kwargs):
		self.entry_file = entry_file
		self.args = args
		self.kwargs = kwargs
	
	def __str__(self):
		return f"python {self.entry_file} " + " ".join(self.args) + " " + " ".join([f"{k}={v}" for k, v in self.kwargs.items()])

class AmltLauncher:
	"""
	TODO format check: e.g. amlt args should be a list
	"""
	def __init__(self):
		self.entry_file = None

		self.launcher_args = {}
		self.args = {}

		self.args["sweep"] = {}

		self.args["normal"] = {}
		self.args["parallel"] = {}
		self.args["others"] = []

		self.config_file_dict = None

	def load_template(self, input, type):
		if type == "file":
			with open(input, "r") as f:
				self.config_file_dict = yaml.safe_load(f)
		elif type == "content":
			self.config_file_dict = yaml.safe_load(input)
		else:
			raise ValueError("type should be file or content")
		
	def parse_cmd_args(self, args):
		type_cur = 1

		for arg in args:
			if arg.startswith("amlt."):
				assert type_cur == 1, "type should be 1"
				key, value = arg.split("=", 1)
				key = key.replace("amlt.", "")
				self.launcher_args[key] = value

			elif arg.endswith(".py") or arg == "python":
				if arg == "python": continue
				assert type_cur == 1, "type should be 1"
				self.entry_file = arg
				type_cur = 2

			else:
				assert type_cur in [2, 3], "type should be 2 or 3"
				type_cur = 3
				if "," not in arg and "#" not in arg:
					if "=" not in arg: 
						self.args["others"].append(arg)
						continue
					key, value = arg.split("=", 1)
					self.args["normal"][key] = value
				elif "," in arg and "#" not in arg:
					key, values = arg.split("=", 1)
					values = [x.strip() for x in values.split(",")]
					self.args["sweep"][key] = values
				elif "," in arg and "#" in arg:
					key, values = arg.split("=", 1)
					values = re.sub(r"[#]", "", values)
					self.args["parallel"][key] = [int(x) for x in values.split(",")]
				else:
					raise KeyError
	
	def launch(self):
		assert self._pre_launch_check(), "pre launch check failed"
		print("pre launch check passed")
		# modify config file keys
		self.config_file_dict = replace_with_dot_keys(self.config_file_dict, self.launcher_args)
		# make jobs
		cmd_str = ""
		params = []
		cmd_str += f"python {self.entry_file}"
		cmd_str += " " + " ".join(self.args["others"])
		cmd_str += " " + " ".join([f"{k}={v}" for k, v in self.args["normal"].items()])
		cmd_str += " " + " ".join([f"{k}={','.join([str(v_) for v_ in v])}" for k, v in self.args["parallel"].items()])
		cmd_str += " " + " ".join([k+"={"+k.replace(".","_")+"}" for k, v in self.args["sweep"].items()])
		for k, v in self.args["sweep"].items():
			params.append({"name": k.replace(".","_"), "values": v})
		# print(cmd_str)
		# print(params)
		# set config file
		self.config_file_dict["search"]["job_template"]["command"] = [cmd_str]
		self.config_file_dict["search"]["params"] = params
		# write config file
		with open("configs/amlt/latest.yaml", "w") as f:
			yaml.dump(self.config_file_dict, f)
		# print summary and ask to run
		print(yaml.dump(self.config_file_dict))
		# numbers
		num_jobs = 1
		for k, v in self.args["sweep"].items():
			num_jobs *= len(v)
		print(f"number of jobs: {num_jobs}")
		# sweeped args
		print("sweeped args:")
		for k, v in self.args["sweep"].items():
			print(f"    {k}: {v}")
		# ask to run
		choice = input("Submit, Run locally or Exit? (S/l/n): ")
		if choice in ["n", "N"]: return
		if choice in ["r", "R", "L", "l"]:
			cmd = f"amlt run -t local configs/amlt/latest.yaml"
			# cmd = "echo 123"
		elif choice in ["s", "S"]:
			name = self.args["normal"]["tags"]
			name += "-"+"".join(random.choices(string.ascii_uppercase + string.digits, k=4))
			cmd = f"amlt run configs/amlt/latest.yaml {name}"
			cmd = "echo 1231"
		print(cmd)
		execute_command(cmd)

	def _pre_launch_check(self):
		# TODO
		return True

	def make_jobs(self):
		assert self.jobs is None, "jobs already made"
		self.jobs = []
		args = []
		kwargs = {}
		kwargs.update(self.args["normal"])
		kwargs.update(self.args["parallel"])
		args += self.args["others"]

		# make grid for all sweep args
		sweep_keys, sweep_values = zip(*self.args["sweep"].items())
		sweep_arg_combinations = list(itertools.product(*sweep_values))

		# make jobs
		for sweep_arg_combo in sweep_arg_combinations:
			sweep_arg = dict(zip(sweep_keys, sweep_arg_combo))
			job_kwargs = {**kwargs, **sweep_arg}
			job = CmdJob(self.entry_file, args, job_kwargs)
			self.jobs.append(job)
		
	def show_info(self):
		total_jobs = len(self.jobs)
		
		print("\nJobs:")
		for index, job in enumerate(self.jobs, start=1):
			print(f"[{index}/{total_jobs}]\n{job}\n")

		print("\nSweept Parameters:")
		for key, values in self.args["sweep"].items():
			print(f"    {key}: {values}")
	
		print("Total number of jobs:", total_jobs)


def main():
	args = sys.argv[1:]
	amlt_launcher = AmltLauncher()
	amlt_launcher.load_template(AMLT_CONFIG_TEMPLATE, type="content")
	amlt_launcher.parse_cmd_args(args)
	# amlt_launcher.show_info()
	amlt_launcher.launch()

def old():
	# Read the template YAML file
	with open("configs/amlt/template.yaml", "r") as f:
		template_dict = yaml.safe_load(f)

	# Parse arguments to update the content of the template
	args = sys.argv[1:]

	type_cur = 1
	amlt_arg = {}
	normal_arg = {}
	sweep_arg = {}
	parallel_arg = {}
	others = []

	# parse_value
	for arg in args:
		# 1. Start with amlt.: change the value in the template
		if arg.startswith("amlt."):
			assert type_cur == 1, "type should be 1"
			key, value = arg.split("=", 1)
			key = key.replace("amlt.", "")
			amlt_arg[key] = value

		# 2. Python, entry file .py: follows the type 1
		elif arg.endswith(".py") or arg == "python":
			if arg == "python": continue
			assert type_cur == 1, "type should be 1"
			template_dict["code"]["command"] = arg
			type_cur = 2
		# 3. Contains no "," and "#": keep the same in the template normal_arg
		# 4. Contains "," but no "#": hparams, will be sweeped
		# 5. Contains "," and "#": parallel key, # will be removed, and the remaining will be the same in the template
		# 6. other such as "-m", just keep it
		else:
			assert type_cur in [2, 3], "type should be 2 or 3"
			type_cur = 3
			if "," not in arg and "#" not in arg:
				if "=" not in arg: 
					others.append(arg)
					continue
				key, value = arg.split("=", 1)
				normal_arg[key] = value
			elif "," in arg and "#" not in arg:
				key, values = arg.split("=", 1)
				values = [x.strip() for x in values.split(",")]
				sweep_arg[key] = values
			elif "," in arg and "#" in arg:
				key, values = arg.split("=", 1)
				values = re.sub(r"[#]", "", values)
				parallel_arg[key] = [int(x) for x in values.split(",")]
			else:
				raise KeyError
			


if __name__ == "__main__":
	main()