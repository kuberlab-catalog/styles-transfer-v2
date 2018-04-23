# Styles transfer demo
This is a ditributed tensorflow implementation based on the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

## Parallel Training
Used for training [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576).

Execution command for workers:

```
python styles.py --job_name=worker --style=style/udnie.jpg --train_dir=$TRAINING_DIR/$BUILD_ID --test=content/chicago.jpg  --content-weight=1.5e1  --batch-size=10 --train-path=$DATA_DIR/train2014/*.jpg --vgg-path=$DATA_DIR/imagenet-vgg-verydeep-19.mat --epochs=1 --style-weight=200 --limit_train=2000 --task_index=$REPLICA_INDEX --ps_hosts=$PS_NODES --worker_hosts=$WORKER_NODES
```
Execution command for parameter server:

```
python styles.py --job_name=ps --task_index=$REPLICA_INDEX --ps_hosts=$PS_NODES --worker_hosts=$WORKER_NODES
```
Argumets:

* `batch_size` - Batch size for trining.
* `epochs` - number of epochs. One is enough for testing model, at least three epochs should be used in real training.
* `style` - style image to learn style content.
* `limit_train` - limit train iterations for tests.
* `task` - trianing process index. Use provided enviroment variable to set this parameter.
* `num_gpus` - number of GPU availble. Use provided enviroment variable to set this parameter.
* `ps_hosts` - adresses of parameter servers. Use provided enviroment variable to set this parameter.
* `worker_hosts` - adresses of workers. Use provided enviroment variable to set this parameter.
* `role` - role of tasks 'worker' or 'ps'.
* `train_dir` - directory to save training results and log metrics which can be used by tensorboard. Use provided enviroment variables to set this parameter.
* `vgg-path` - path to pretrained imagenet-vgg-verydeep-19.

<mark>Training this model on CPU is to slow!!!<mark>

## Serving
Example of a trained model served as microservice application.

Execution command:

```
tensorflow_model_server --port=9000 --model_name=transform --model_base_path=$checkpoint_path
```
Argumets:

* `model_name` - Name of the model. Use provided enviroment variable to set this parameter.
* `model_base_path` - Use provided enviroment variable to set this parameter.