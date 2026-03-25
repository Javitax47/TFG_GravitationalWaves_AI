# src/data_processing/tfrecord_parsers.py

import tensorflow as tf

TARGET_SHAPE_2D = (256, 512, 1)
TARGET_SHAPE_1D = (8192, 1)

def get_parse_fn(dimension, tasks):
    def parse_fn(example):
        if dimension == '2D':
            feature_desc = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
                'event_type': tf.io.FixedLenFeature([], tf.int64),
                'mass1': tf.io.FixedLenFeature([], tf.float32),
                'mass2': tf.io.FixedLenFeature([], tf.float32),
            }
            parsed = tf.io.parse_single_example(example, feature_desc)
            inputs = tf.io.parse_tensor(parsed['image'], out_type=tf.float32)
            inputs.set_shape(TARGET_SHAPE_2D)
            inputs = tf.image.per_image_standardization(inputs)
            
        elif dimension == '1D':
            feature_desc = {
                'signal': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
                'event_type': tf.io.FixedLenFeature([], tf.int64),
                'mass1': tf.io.FixedLenFeature([], tf.float32),
                'mass2': tf.io.FixedLenFeature([], tf.float32),
            }
            parsed = tf.io.parse_single_example(example, feature_desc)
            inputs = tf.io.parse_tensor(parsed['signal'], out_type=tf.float32)
            inputs.set_shape(TARGET_SHAPE_1D)
            mean = tf.math.reduce_mean(inputs)
            std = tf.math.reduce_std(inputs)
            inputs = (inputs - mean) / tf.maximum(std, 1e-6)

        outputs = {}
        
        if 'detection' in tasks:
            outputs['head_detection'] = tf.cast(parsed['label'], tf.float32)
            
        if 'classification' in tasks:
            outputs['head_classification'] = tf.cast(parsed['event_type'], tf.int64)
            
        if 'masses' in tasks:
            m1 = parsed['mass1']
            m2 = parsed['mass2']
            mass_array = tf.stack([tf.maximum(m1, m2), tf.minimum(m1, m2)])
            outputs['head_masses'] = tf.cast(mass_array, tf.float32)

        if len(tasks) == 1:
            return inputs, list(outputs.values())[0]
        
        return inputs, outputs

    return parse_fn

def apply_mixup_multitask(inputs, outputs, alpha=0.2):
    batch_size = tf.shape(inputs)[0]
    
    gamma_1 = tf.random.gamma([batch_size], alpha)
    gamma_2 = tf.random.gamma([batch_size], alpha)
    l = tf.cast(gamma_1 / (gamma_1 + gamma_2), inputs.dtype)
    l_img = tf.reshape(l, [-1, 1, 1, 1] if len(inputs.shape) == 4 else [-1, 1, 1])

    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_inputs = tf.gather(inputs, indices)
    mixed_inputs = l_img * inputs + (tf.cast(1.0, inputs.dtype) - l_img) * shuffled_inputs

    if isinstance(outputs, dict):
        mixed_outputs = {}
        for key, val in outputs.items():
            if key == 'head_detection':
                l_lbl = tf.cast(tf.reshape(l, [-1]), val.dtype)
                shuffled_val = tf.gather(val, indices)
                mixed_outputs[key] = l_lbl * val + (tf.cast(1.0, val.dtype) - l_lbl) * shuffled_val
            else:
                shuffled_val = tf.gather(val, indices)
                mask = tf.cast(l > 0.5, val.dtype)
                mixed_outputs[key] = mask * val + (1 - mask) * shuffled_val
        return mixed_inputs, mixed_outputs
    else:
        l_lbl = tf.cast(tf.reshape(l, [-1]), outputs.dtype) 
        shuffled_val = tf.gather(outputs, indices)
        mixed_outputs = l_lbl * outputs + (tf.cast(1.0, outputs.dtype) - l_lbl) * shuffled_val
        return mixed_inputs, mixed_outputs

def create_grid_dataset(file_paths, dimension, tasks, batch_size, shuffle=True, is_training=False, use_mixup=False, use_time_shift=False):
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    if shuffle:
        path_ds = path_ds.shuffle(len(file_paths), reshuffle_each_iteration=True)

    dataset = path_ds.interleave(lambda x: tf.data.TFRecordDataset(x), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000, reshuffle_each_iteration=True)

    parser = get_parse_fn(dimension, tasks)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)

    if is_training:
        # 1. DATA AUGMENTATION
        if use_time_shift:
            # Solo aplicamos la traslación de imagen si es un espectrograma 2D
            if dimension == '2D':
                shift_layer = tf.keras.layers.RandomTranslation(
                    height_factor=0.0, width_factor=0.15, fill_mode='constant', fill_value=0.0
                )
                dataset = dataset.map(lambda x, y: (shift_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        
        # 2. MIXUP
        # CRÍTICO: Anulamos Mixup si predecimos masas O si estamos en 1D (para evitar interferencia destructiva)
        if use_mixup and 'masses' not in tasks and dimension == '2D':
            dataset = dataset.map(apply_mixup_multitask, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(tf.data.AUTOTUNE)