import tensorflow as tf
import transform, pdb, os

def export(checkpoint_dir,batch_shape):
    with tf.Graph().as_default(),tf.Session() as sess:
        images= tf.placeholder(tf.float32, shape=batch_shape,name='images')
        preds = transform.net(images/255.0)
        tensor_info_images = tf.saved_model.utils.build_tensor_info(images)
        tensor_info_preds = tf.saved_model.utils.build_tensor_info(preds)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_images},
                outputs={'preds': tensor_info_preds},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        saver = tf.train.Saver()
        saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))
        export_path = os.path.join(checkpoint_dir,"1")
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                    prediction_signature
            })
        builder.save()

def export2(checkpoint_dir):
    with tf.Graph().as_default(),tf.Session() as sess:
        feature_spec = {'image': tf.FixedLenFeature([],dtype=tf.string)}
        features, _, inputs = tf.contrib.learn.utils.input_fn_utils.build_parsing_serving_input_fn(feature_spec)()
        image = tf.image.decode_image(features['image'][0], channels=3)
        image.set_shape([None,None,3])
        image = tf.image.resize_image_with_crop_or_pad([image],512,512)
        image = tf.to_float(image, name='ToFloat')
        preds = transform.net(image/255.0)
        result = preds*255.0
        result = tf.cast(result, tf.uint8)
        result_image = tf.image.encode_png(result[0])
        tensor_info_result = tf.saved_model.utils.build_tensor_info(result_image)
        inputs={'examples':tf.saved_model.utils.build_tensor_info(inputs['examples'])}
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs,
                outputs={'result': tensor_info_result},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        saver = tf.train.Saver()
        saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))
        export_path = os.path.join(checkpoint_dir,'5')
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'transform':
                    prediction_signature
            })
        builder.save()