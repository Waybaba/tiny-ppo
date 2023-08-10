"""

Use command line to launch a amlt sweep job.

Usage:
    run cmd line in shell (see template below)
    >>> read from configs/amlt/template.yaml
    >>> fill content based on template and command line input
    >>> replace the default file in {CONFIG_OUTPUT_PATH}
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

=========================== {CONFIG_OUTPUT_PATH} >>>>>>>>>>>>>>>>
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

=========================== {CONFIG_OUTPUT_PATH} <<<<<<<<<<<<<<<<

"""
import yaml
import re
import sys
import os
import itertools
import random
import string
from functools import reduce
import subprocess


AMLT_CONFIG_TEMPLATE = """
description: AMLT

target:
    service: singularity
    name: msrresrchvc

environment:
    image: waybaba/rl:v4 # ! may need to be changed
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

CONFIG_OUTPUT_PATH = "configs/amlt/latest.yaml"

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

def parse_string_list(input_string):
    # Using regex to match elements separated by commas
    elements = re.findall(r'(\[.*?\]|".*?"|\'.*?\'|[^,\s]+)', input_string)
    parsed_elements = []

    for elem in elements:
        if "[" in elem and "]" in elem:
            # Keep nested lists as strings
            parsed_elements.append(elem)
        else:
            parsed_elem = elem.strip('"\'')
            if parsed_elem.isnumeric():
                parsed_elem = int(parsed_elem)
            elif re.match(r'^-?\d+\.\d+$', parsed_elem):
                parsed_elem = float(parsed_elem)
            parsed_elements.append(parsed_elem)

    return parsed_elements

class CmdJob:
    def __init__(self, entry_file, args, kwargs):
        self.entry_file = entry_file
        self.args = args
        self.kwargs = kwargs
    
    def __str__(self):
        return f"python {self.entry_file} " + " ".join(self.args) + " " + " ".join([f"{k}={v}" for k, v in self.kwargs.items()])

class AmltLauncher:
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

            else: # args
                # ! case actor.net.mlp_hidden_sizes=[256,256] would be treated as sweep 
                if "," not in arg and "#" not in arg:
                    if "=" not in arg: 
                        self.process_others(arg)
                    else:
                        self.process_normal(arg)
                elif "," in arg and "#" not in arg:
                    self.process_sweep(arg)
                elif "," in arg and "#" in arg:
                    self.process_parallel(arg)
                else:	
                    raise KeyError("unknown arg type, {arg}")
                
    def launch(self):
        assert self._pre_launch_check(), "pre launch check failed"
        print("pre launch check passed")
        
        # modify config file keys
        self.config_file_dict = replace_with_dot_keys(self.config_file_dict, self.launcher_args)
        
        # create command string
        cmd_str = ""
        cmd_str += f"python {self.entry_file}"
        cmd_str += " " + " ".join(self.args["others"])
        cmd_str += " " + " ".join([f"{k}={v}" for k, v in self.args["normal"].items()])
        cmd_str += " " + " ".join([f"{k}={v}" for k, v in self.args["parallel"].items()])
        cmd_str += " " + " ".join([k+"={"+k.replace(".","_")+"_"+"}" for k, v in self.args["sweep"].items()])
    
        # create params list
        params = [{"name": k.replace(".", "_")+"_", "values": v} for k, v in self.args["sweep"].items()]
        
        # set config file
        self.config_file_dict["search"]["job_template"]["command"] = [cmd_str]
        self.config_file_dict["search"]["params"] = params
        
        # write config file
        with open(CONFIG_OUTPUT_PATH, "w") as f:
            yaml.dump(self.config_file_dict, f)
        
        # print summary and ask to run
        print(yaml.dump(self.config_file_dict))
        
        num_jobs = reduce(lambda x, y: x*y, map(len, self.args['sweep'].values()), 1)
        
        print("\n" + "="*20)
        print(f"\nTags: {self.args['normal']['tags']}")
        print(f"\nCommand: {cmd_str}")
        print("\nSweeped args:")
        for k, v in self.args["sweep"].items():
            print(f"    {k}: {v}")
        print(f"\nNumber of jobs: {num_jobs}")
        
        # ask to run
        print("\nSubmit(s), Run locally(l) or Exit(n)? (S/l/n): ")
        try:
            choice = input()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            exit()
        
        if choice.lower() == "n":
            return
        
        if choice.lower() == "l":
            cmd = f"amlt run -t local {CONFIG_OUTPUT_PATH}"
        elif choice.lower() in ["", "s"]:
            name = self.args["normal"]["tags"].strip("[]\"").replace(" ", "_")
            if "search.job_template.sku" in self.launcher_args: # add sku name
                sku = self.launcher_args['search.job_template.sku']
                if "P40" in sku: gpu_name = "P40"
                elif "P100" in sku: gpu_name = "P100"
                elif "V100" in sku: gpu_name = "V100"
                elif sku == "G1": gpu_name = "P100"
                name += f"___{gpu_name}___"
            name += "-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
            cmd = f"amlt run {CONFIG_OUTPUT_PATH} {name}"
        else:
            raise ValueError(f"invalid choice {choice}")
        
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

    def process_normal(self, arg):
        key, value = arg.split("=", 1)
        self.args["normal"][key] = value

    def process_sweep(self, arg):
        key, values = arg.split("=", 1)
        self.args["sweep"][key] = parse_string_list(values)

    def process_parallel(self, arg):
        key, values = arg.split("=", 1)
        values = re.sub(r"[#]", "", values)
        self.args["parallel"][key] = values

    def process_others(self, arg):
        self.args["others"].append(arg)

def main():
    args = sys.argv[1:]
    amlt_launcher = AmltLauncher()
    amlt_launcher.load_template(AMLT_CONFIG_TEMPLATE, type="content")
    amlt_launcher.parse_cmd_args(args)
    # amlt_launcher.show_info()
    amlt_launcher.launch()

if __name__ == "__main__":
    main()