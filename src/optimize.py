from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np
import transform
from utils import get_img,styles_data
import horovod.tensorflow as hvd
from tensorflow import logging

tf.logging.set_verbosity(tf.logging.INFO)

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

# np arr, np arr
def optimize(cluster,task_index,limit,file_pattern, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2,
             batch_size=4, save_path='saver',
             learning_rate=1e-3,test_image=""):
    logging.info("START")
    local_step = 0
    t_img = get_img(test_image,(256,256,3)).astype(np.float32)
    Test = np.zeros((batch_size,256,256,3), dtype=np.float32)
    for i in range(0, batch_size):
        Test[i] = t_img

    style_features = {}

    batch_shape = (batch_size,256,256,3)
    style_shape = (1,) + style_target.shape
    is_chief = (task_index == 0)
    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu'), tf.Session():
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    server = tf.train.Server(
        cluster, job_name="worker", task_index=task_index)
    worker_device = "/job:worker/task:%d" % (task_index)

    time_begin = time.time()
    logging.info("Training begins @ %f",time_begin)
    #sess_config = tf.ConfigProto(
    #    allow_soft_placement=True,
    #    log_device_placement=False,
    #    device_filters=["/job:ps", worker_device])
    sess_config = tf.ConfigProto()
    with tf.device(
            tf.train.replica_device_setter(
                worker_device=worker_device,
                ps_device="/job:ps",
                cluster=cluster)):
        dataset = styles_data(file_pattern,batch_size,limit,True)
        num_examples = dataset['size']
        num_samples = num_examples / batch_size
        num_global =  num_samples * epochs
        logging.info("Number of iterations %d",num_global)
        global_step =tf.train.get_or_create_global_step()
        X_content,_ = dataset['batch']
        #X_content,_ = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        preds = transform.net(X_content/255.0)
        preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
                                         )

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size


        loss = content_loss + style_loss + tv_loss

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('tv_loss', tv_loss)
        tf.summary.scalar('style_loss', style_loss)
        tf.summary.scalar('content_loss', content_loss)

        result = preds*255.0

        tf.summary.image('result', result,max_outputs=1)

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

        all_summary = tf.summary.merge_all()


        scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                     summary_op=all_summary)
        scaffold.global_step = global_step
        step = 0
        local_step = 0
        stopAt = tf.train.StopAtStepHook(num_steps=num_global)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               hooks=[stopAt],
                                               is_chief=is_chief,checkpoint_dir=save_path,
                                               config=sess_config,
                                               save_checkpoint_secs=None,
                                               save_summaries_steps=100,
                                               log_step_count_steps=10,
                                               scaffold=scaffold) as sess:
            while not sess.should_stop():
                _, step = sess.run([train_step, global_step])
                local_step += 1
                logging.info("Worker %d: training step %d done (global step: %d)" ,task_index, local_step, step)

            time_end = time.time()
            logging.info("Training ends @ %f" , time_end)
            training_time = time_end - time_begin
            logging.info("Training elapsed time: %f s" , training_time)
        return

def single(cluster,task_index,limit,file_pattern, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2,
             batch_size=4, save_path='saver',
             learning_rate=1e-3,test_image=""):
    logging.info("START")
    local_step = 0
    t_img = get_img(test_image,(256,256,3)).astype(np.float32)
    Test = np.zeros((batch_size,256,256,3), dtype=np.float32)
    for i in range(0, batch_size):
        Test[i] = t_img

    style_features = {}

    batch_shape = (batch_size,256,256,3)
    style_shape = (1,) + style_target.shape
    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu'), tf.Session():
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram


    time_begin = time.time()
    logging.info("Training begins @ %f",time_begin)
    #sess_config = tf.ConfigProto(
    #    allow_soft_placement=True,
    #    log_device_placement=False,
    #    device_filters=["/job:ps", worker_device])
    sess_config = tf.ConfigProto()
    with tf.Session() as sess:
        dataset = styles_data(file_pattern,batch_size,limit,True)
        num_examples = dataset['size']
        num_samples = num_examples / batch_size
        num_global =  num_samples * epochs
        logging.info("Number of iterations %d",num_global)
        global_step =tf.train.get_or_create_global_step()
        X_content,_ = dataset['batch']
        #X_content,_ = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        preds = transform.net(X_content/255.0)
        preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
                                         )

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size


        loss = content_loss + style_loss + tv_loss

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('tv_loss', tv_loss)
        tf.summary.scalar('style_loss', style_loss)
        tf.summary.scalar('content_loss', content_loss)


        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
        step = 0
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
        while step < num_global:

                #X_batch, _ = sess.run(dataset['batch'])
                #feed_dict = {
                #    X_content:X_batch
                #}
            start_time = time.time()
            _, step = sess.run([train_step, global_step])
            local_step += 1
            logging.info("Worker %d: training step %d done (global step: %d) %.2f", task_index, local_step, step,time.time() - start_time)

        time_end = time.time()
        logging.info("Training ends @ %f" , time_end)
        training_time = time_end - time_begin
        logging.info("Training elapsed time: %f s" , training_time)
        return

def hororovod(cluster,task_index,limit,file_pattern, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2,
             batch_size=4, save_path='saver',
             learning_rate=1e-3,test_image=""):
    logging.info("START")
    local_step = 0
    t_img = get_img(test_image,(256,256,3)).astype(np.float32)
    Test = np.zeros((batch_size,256,256,3), dtype=np.float32)
    for i in range(0, batch_size):
        Test[i] = t_img

    style_features = {}

    batch_shape = (batch_size,256,256,3)
    style_shape = (1,) + style_target.shape
    print(style_shape)
    # precompute style features
    hvd.init()
    sess_config = tf.ConfigProto()
    #sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
    with tf.Graph().as_default(), tf.device('/cpu'), tf.Session(config=sess_config):
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    time_begin = time.time()
    logging.info("Training begins @ %f", time_begin)
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    checkpoint_dir = save_path
    is_chief = True
    log_step_count_steps = 3
    save_summaries_steps = 100
    if hvd.rank() != 0:
        checkpoint_dir = None
        is_chief = False
        save_summaries_steps = 100000
        log_step_count_steps = None

    dataset = styles_data(file_pattern,batch_size,limit,True)
    num_examples = dataset['size']
    num_samples = num_examples / batch_size
    num_global =  num_samples * epochs
    logging.info("Number of iterations %d" , num_global)
    global_step =tf.train.get_or_create_global_step()
    #X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
    X_content,_ = dataset['batch']
    X_pre = vgg.preprocess(X_content)

    # precompute content features
    content_features = {}
    content_net = vgg.net(vgg_path, X_pre)
    content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

    preds = transform.net(X_content/255.0)
    preds_pre = vgg.preprocess(preds)

    net = vgg.net(vgg_path, preds_pre)

    content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
    assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
    content_loss = content_weight * (2 * tf.nn.l2_loss(
        net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
                                     )

    style_losses = []
    for style_layer in STYLE_LAYERS:
        layer = net[style_layer]
        bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
        size = height * width * filters
        feats = tf.reshape(layer, (bs, height * width, filters))
        feats_T = tf.transpose(feats, perm=[0,2,1])
        grams = tf.matmul(feats_T, feats) / size
        style_gram = style_features[style_layer]
        style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

    style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

    # total variation denoising
    tv_y_size = _tensor_size(preds[:,1:,:,:])
    tv_x_size = _tensor_size(preds[:,:,1:,:])
    y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
    x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
    tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size


    loss = content_loss + style_loss + tv_loss

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('tv_loss', tv_loss)
    tf.summary.scalar('style_loss', style_loss)
    tf.summary.scalar('content_loss', content_loss)
    result = preds*255.0
    tf.summary.image('result', result,max_outputs=1)

    # overall loss
    train_step = tf.train.AdamOptimizer(learning_rate*hvd.size())
    train_step = hvd.DistributedOptimizer(train_step)
    train_op = train_step.minimize(loss,global_step=global_step)
    all_summary = tf.summary.merge_all()


    scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                 summary_op=all_summary)
    scaffold.global_step = global_step
    step = 0
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           config=sess_config,
                                           save_summaries_steps=save_summaries_steps,
                                           hooks=hooks,
                                           save_checkpoint_secs=None,
                                           log_step_count_steps=log_step_count_steps,
                                           scaffold=scaffold) as sess:
        while step < num_global and not sess.should_stop():
            #X_batch, _ = sess.run(dataset['batch'])
            #feed_dict = {
            #    X_content:X_batch
            #}
            _, step = sess.run([train_op, global_step])
            local_step += 1
            logging.info("Worker %d: training step %d done (global step: %d)" ,task_index, local_step, step)
        time_end = time.time()
        logging.info("Training ends @ %f" , time_end)
        training_time = time_end - time_begin
        logging.info("Training elapsed time: %f s" , training_time)
        sess.request_stop()
    return


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)