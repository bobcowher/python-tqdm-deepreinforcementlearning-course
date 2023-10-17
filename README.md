# Set up on a remote system
mkdir /root/app
pip install -r requirements.txt
scp -P 10267 -r .\*.py root@79.116.10.163:/root/app

# Copy from remote to local
scp -P 10267 -r root@79.116.10.163:/root/app/models/* ./models

# Monitor Tensorboard
conda activate ddpg
tensorboard --logdir=./2023-10-11

# Monitor Tensorboard Remotely
ssh -L 16006:127.0.0.1:6006 root@79.116.10.163 -p 10267
tensorboard --logdir=./2023-10-11


