from __future__ import print_function
import sys, os, pdb
from optimize import optimize, hororovod, single
from argparse import ArgumentParser
from utils import get_img, exists, list_files
import tensorflow as tf
import export
from mlboardclient.api import client

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoints'
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014/*.jpg'
BATCH_SIZE = 4


PS_HOSTS = 'localhost:2222'
WORKER_HOSTS = 'localhost:2222'
JOB_NAME = 'worker'
TASK_INDEX = 0

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--train_dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=False)

    parser.add_argument('--export_path', type=str,
                        dest='export_path', help='Dir to export model',
                        metavar='EXPORT_PATH', required=False)

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=False)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--ps_hosts', type=str,
                        dest='ps_hosts',
                        help='comma-separated list of hostname:port pairs',
                        metavar='PS_HOSTS', default='localhost:2222')
    parser.add_argument('--worker_hosts', type=str,
                        dest='worker_hosts',
                        help='comma-separated list of hostname:port pairs',
                        metavar='WORKER_HOSTS', default='localhost:3333')
    parser.add_argument('--job_name', type=str,
                        dest='job_name',
                        help='job name: worker or ps',
                        metavar='JOB_NAME', default='worker')
    parser.add_argument('--task_index', type=int,
                        dest='task_index',
                        help="Worker task index, should be >= 0. task_index=0 is "
                             "the master worker task the performs the variable "
                             "initialization ",
                        metavar='TASK_INDEX', default=0)
    parser.add_argument('--limit_train', type=int,
                        dest='limit_train',
                        help='Limit train set by number',
                        metavar='LIMIT_TRAIN', default=0)

    return parser

def check_opts(opts):
    if opts.task_index==0:
        if not os.path.exists(opts.checkpoint_dir):
            os.makedirs(opts.checkpoint_dir)
    exists(opts.style, "style path not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0
    assert opts.task_index >=0

def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]


def main():
    parser = build_parser()
    options = parser.parse_args()

    ps_spec = options.ps_hosts.split(",")
    worker_spec = options.worker_hosts.split(",")

    cluster = None

    if options.job_name == "mpi":
        print('Use MPI')
    else:
        cluster = tf.train.ClusterSpec({
            "ps": ps_spec,
            "worker": worker_spec})
        if options.job_name == "ps":
            print("Start parameter server %d" % (options.task_index))
            server = tf.train.Server(
            cluster, job_name=options.job_name, task_index=options.task_index)
            server.join()
            return

    check_opts(options)

    style_target = get_img(options.style)

    kwargs = {
        "epochs":options.epochs,
        "batch_size":options.batch_size,
        "save_path":options.checkpoint_dir,
        "learning_rate":options.learning_rate,
        "test_image":options.test
    }

    args = [
        cluster,
        options.task_index,
        options.limit_train,
        options.train_path,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path
    ]
    client.update_task_info({'checkpoint_path': options.checkpoint_dir})
    if options.job_name == "mpi":
        hororovod(*args, **kwargs)
    else:
        if options.job_name == "single":
            single(*args, **kwargs)
        if options.job_name == "export":
            print('Export model for serving:')
            expath = os.path.join(options.export_path,"1")
            export.export2(options.checkpoint_dir,expath)
            client.update_task_info({'model_path': expath},{'checkpoint_path': options.export_path})
            return
        else:
            optimize(*args, **kwargs)
    print('Export model for serving:')
    if options.task_index == 0:
        export.export2(options.checkpoint_dir,os.path.join(options.checkpoint_dir,"1"))
        client.update_task_info({'model_path': os.path.join(options.checkpoint_dir,"1")})

if __name__ == '__main__':
    main()
