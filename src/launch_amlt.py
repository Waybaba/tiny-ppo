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

python launch_amlt.py \ # start
	amlt.search.job_template.sku=G1 \ # start with amlt would change the template key
	
	python src/entry.py -m \ all single options without [] would be copied
	hydra/launcher=wsl_parallel \
	++trainer.log_upload_interval=100000 \
	n_jobs=5 \
	seed=#1,2,3,4# \ # these in #xxx# would be kept as multiple in the final version to support parallel
	env.delay=0,1,2,4,8 \
	tags="test_tag" \

	
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
			- python src/entry.py -m
				hydra/launcher=wsl_parallel
				++trainer.log_upload_interval=100000
				n_jobs=5
				seed={seed}
				env.delay={env_delay}
				tags="test_tag"
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

def main():
	# Read the template YAML file
	with open("configs/amlt/template.yaml", "r") as f:
		template_dict = yaml.safe_load(f)

	# Parse arguments to update the content of the template
	args = sys.argv[1:]

	type_cur = 1
	normal_arg = {}
	sweep_arg = {}
	parallel_arg = {}
	others = []
	for arg in args:
		# 1. Start with amlt.: change the value in the template
		if arg.startswith("amlt."):
			assert type_cur == 1, "type should be 1"
			key, value = arg.split("=", 1)
			key = key.replace("amlt.", "", 1)
			subkeys = key.split(".")
			subdict = template_dict
			for subkey in subkeys[:-1]:
				subdict = subdict[subkey]
			subdict[subkeys[-1]] = value
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
			
	# Output information about the sweep
	print("Sweeped parameters and job numbers:")
	for key, values in sweep_arg.items():
		print(f"{key}: {values}")
	print("")

	# Output file directory information
	print("File directory:")
	print("configs/amlt/latest.yaml")
	print("")

	# Output other information
	print("Other information:")
	print(f"Normal arguments: {normal_arg}")
	print(f"Parallel arguments: {parallel_arg}")
	print(f"Others: {others}")
	print("")

if __name__ == "__main__":
	main()