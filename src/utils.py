import scipy.misc, numpy as np, os, sys

from glob import glob
import tensorflow as tf

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target

def get_img(src, img_size=False):
    img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img

def exists(p, msg):
    assert os.path.exists(p), msg

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files

def read_image(filename_queue):
    image_reader = tf.WholeFileReader()
    path, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file, 3)
    image = tf.image.resize_images(image,(256,256))
    image = tf.to_float(image, name='ToFloat')
    return [image, path]
def styles_data(file_pattern,batch_size,limit, shuffle):
    files = glob(file_pattern)
    if limit >0 :
        print("Limit train set %d" % limit)
        files = files[0:limit]
    filename_queue = tf.train.string_input_producer(
        files,
        shuffle=shuffle,
        num_epochs=None if shuffle else 1)
    image = read_image(filename_queue)

    # Mini batch
    num_preprocess_threads = 4
    min_queue_examples = 10000
    if shuffle:
        images = tf.train.shuffle_batch(
            image,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images = tf.train.batch(
            image,
            batch_size,
            allow_smaller_final_batch=True)
    return dict(batch=images, size=len(files))