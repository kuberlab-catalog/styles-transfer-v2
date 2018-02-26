import glob
import sys

import tensorflow as tf
from tensorflow import logging

args = sys.argv[1:]
if len(args) < 1:
    print("Usage: image.py <image-dir>")
    sys.exit(1)

base_path = args[0]
base_path = base_path.rstrip('/')

tf.logging.set_verbosity(tf.logging.INFO)
images = sorted(glob.glob("%s/*.jpg" % base_path))
filename_queue = tf.train.string_input_producer(images, shuffle=False)

image_reader = tf.WholeFileReader()
path, image_content = image_reader.read(filename_queue)
image = tf.image.decode_jpeg(image_content, 3)
image = tf.image.resize_images(image, (256, 256))
image = tf.to_float(image, name='ToFloat')

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for image_path in images:
        logging.info(image_path)
        _ = sess.run(image)
    coord.request_stop()
    coord.join(threads)
