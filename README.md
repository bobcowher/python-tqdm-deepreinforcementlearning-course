# Set up on a remote system
mkdir /root/app
pip install -r requirements.txt
scp -P PORT -r .\*.py root@HOST:/root/app

# Monitor Tensorboard
conda activate ddpg
tensorboard --logdir=./2023-10-11

# Monitor Tensorboard Remotely
ssh -L 16006:127.0.0.1:6006 root@HOST -p 43436
tensorboard --logdir=./2023-10-11
