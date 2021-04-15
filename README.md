# tiny-ppo
a single file implementation of ppo

## Install
```bash
git clone https://github.com/Waybaba/tiny-ppo.git
cd tiny-ppo
pip install -r requirements.txt
```

## Usage
Install and run
```bash
cd tiny-ppo
python ./agent/ppo.py
# or you can just run ppo.py in your IDE
# this will generate tensorboard logs in ./data/log/tensorboard_log/
# the reward history will be saved in ./data/res/
```
Draw result
```bash
# this script will read result in ./data/res/ and plot them
# the result figure will be saved in ./output/ as *.png
python ./utils/draw_res.py
```
There are some files end with _.png, _.npy, these are my result files.
