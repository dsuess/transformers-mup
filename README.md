# Transformers-muP

This repo currently contains the source code for the blog post **TBD**.
It serves as a demo how to use Google Cloud TPU VMs in Ray Tune for hyperparameter tuning a Huggingface Transformers model implemented in Pytorch.
Thanks to the [TPU Research cloud](https://sites.research.google/trc/about/) for providing the compute for free.

## Steps

- Install gcloud, set the default region to where your free TPUs are located and make sure you're using the correct project. If you don't get this right, we will launch TPUs in non-free zones, which can get \$\$\$. Below, we'll use v2-8 instances, which for me are located in `us-central1-f`.
```
(local) $ gcloud config set compute/region us-central1
(local) $ gcloud config set compute/zone us-central1-f
```

- Create a VM used for launching everything. We give this VM full google-cloud permissions to simplify the instructions, so use with care!
```
(local) $ gcloud compute instances create interactive-vm \
    --image https://www.googleapis.com/compute/v1/projects/ubuntu-os-cloud/global/images/ubuntu-2004-focal-v20220308 \
    --boot-disk-size 50GB \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

- Connect to the machine
```
(local) $ gcloud compute ssh ubuntu@interactive-vm
```

- All the following commands will need to be run on the `interactive-vm`, so we'll drop the prompt indication
- Clone the repo and setup environment:

```
$ git clone https://github.com/dsuess/transformers-mup -b TODO
$ cd transformers-mup
$ sudo apt update && sudo apt install -y python3-pip
$ python3 -m pip install -r requirements.txt
$ echo "export PATH=$HOME/.local/bin:\$PATH" >> $HOME/.bashrc
$ source $HOME/.bashrc
```

- edit the `cluster.yaml` file, things you need to change: `provider.project_id`, `max_workers`
- launch the cluster (this can take a while, you need to confirm generation of head node)
```
$ ray up cluster.yaml
```

- note the address under "To connect to this Ray runtime from another node, run", use that command to connect to cluster
```
$ ray start --address='10.128.15.199:6379' --redis-password='5241590000000000'
```

- this should print a command to use for monitoring autoscaling (preferably in separate ssh session):
```
$ ray exec /home/ubuntu/transformers-mup/cluster.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
```

- should print sth like this:
```
---------------------------------------------------------------
Healthy:
 1 ray_head_default
Pending:
 10.128.15.203: ray_tpu, uninitialized
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
```

- if no pending worker nodes, keep an eye open for errors saying "no resources available".
- launch hyperparameter optimization using the address from above:

```
$ python3 run.py tune --address='10.128.15.199:6379'
```

- this will start printing progress bars from all workers, keep an eye open for the trial-overview (with 3 workers):
```
Number of trials: 4/infinite (1 PENDING, 3 RUNNING)
+------------------+----------+--------------------+-----------------+--------------------+----------------+
| Trial name       | status   | loc                |   learning_rate |   num_train_epochs |   warmup_ratio |
|------------------+----------+--------------------+-----------------+--------------------+----------------|
| DEFAULT_b7d5ba74 | RUNNING  | 10.128.15.207:7757 |     0.00160626  |            30.3291 |      0.0296713 |
| DEFAULT_b8b1bf4c | RUNNING  | 10.128.15.206:8967 |     2.98192e-06 |            25.476  |      0.0593617 |
| DEFAULT_b8b1bf4d | RUNNING  | 10.128.15.203:8860 |     0.000817843 |            42.3444 |      0.093343  |
| DEFAULT_ba2abed2 | PENDING  |                    |     9.17652e-05 |            47.821  |      0.0391263 |
+------------------+----------+--------------------+-----------------+--------------------+----------------+
```

- results will be written to csv file in wroking when done
- when finsihed, shutdown cluster and interactive-vm (not shown)

```
$ ray down cluster.yaml
```