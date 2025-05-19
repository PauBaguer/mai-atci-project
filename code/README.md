# PPO implementation project

## How to run

```
# INSTALL DEPENDENCIES
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# RUN
python3 main_discrete.py --track --capture-video

# or
python3 main_continuous.py --track --capture-video


# I had to use xvfb-run to get the video to work (server without display).
sudo apt install xvfb
xvfb-run python3 main_discrete.py --track --capture-video
```