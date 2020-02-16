# Copyright 2019 Damian Schori. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import random
import numpy as np
import tensorflow as tf  # TF2
import matplotlib.pyplot as plt
import cv2
import skimage

assert tf.__version__.startswith('2'), 'use tensorflow 2.x'

IMG_WIDTH = 384
IMG_HEIGHT = 384
PARALLEL_CALLS = 4

BUFFER_SIZE = 400
BATCH_SIZE = 4
EPOCHS = 50

SMOOTH = 1e-5

BACKBONE_LAYER_NAMES = {
    'vgg19': [
        'block2_conv2',
        'block3_conv4',
        'block4_conv4',
        'block5_conv4',
        'block5_pool'],
    'resnet50': [
        'conv1_relu',
        'conv2_block3_out',
        'conv3_block4_out',
        'conv4_block6_out',
        'conv5_block3_out'],
    'resnet50v2': [
        'conv1_conv',
        'conv2_block3_1_relu',
        'conv3_block4_1_relu',
        'conv4_block6_1_relu',
        'post_relu'],
    'resnet101': [
        'conv1_relu',
        'conv2_block3_out',
        'conv3_block4_out',
        'conv4_block6_out',
        'conv5_block3_out'],
    'mobilenetv2': [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project']
}


class Config():
    dates = ['20190703', '20190719', '20190822']
    fields = ['Field_A', 'Field_C']
    seed = 1
    train_size = 0.8


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return img


def process_path(image_path, mask_path):
    # load the raw data from the file as a string

    img = tf.io.read_file(image_path)
    msk = tf.io.read_file(mask_path)
    img = decode_img(img)
    msk = decode_img(msk)
    return img, msk, image_path


def random_flip(image, mask, image_path):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    return image, mask, image_path


def random_crop(image, mask, image_path):
    stacked_image = tf.stack([image, mask], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1], image_path


@tf.function
def central_crop(image, mask, image_path):
    image = image[64:-64, 64:-64]
    mask = mask[64:-64, 64:-64]
    return image, mask, image_path


def add_gaussian_noise(image, mask, image_path):
    # image must be scaled in [0, 1]
    if tf.random.uniform(()) > 0.5:
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=(10) / (255), dtype=tf.float32)
        noise_img = image + noise
        noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
    else:
        noise_img = image
    return noise_img, mask, image_path


def unindex(image, mask, image_path):
    return image, mask


def create_train_datasets(train_set_list, val_set_list, test_set_list, buffer_size, batch_size):
    """Creates a tf.data Dataset.
    Args:
    path_to_train_images: Path to train images folder.
    path_to_test_images: Path to test images folder.
    buffer_size: Shuffle buffer size.
    batch_size: Batch size
    Returns:
    train dataset, test dataset
    """
    train_set_images = tf.data.Dataset.list_files(train_set_list[0], shuffle=False)
    train_set_masks = tf.data.Dataset.list_files(train_set_list[1], shuffle=False)
    train_set = tf.data.Dataset.zip((train_set_images, train_set_masks))
    train_set = train_set.shuffle(buffer_size)

    train_set = train_set.map(process_path, num_parallel_calls=PARALLEL_CALLS)
    train_set = train_set.map(random_flip, num_parallel_calls=PARALLEL_CALLS)
    train_set = train_set.map(random_crop, num_parallel_calls=PARALLEL_CALLS)
    train_set = train_set.map(add_gaussian_noise, num_parallel_calls=PARALLEL_CALLS)
    train_set = train_set.batch(batch_size, drop_remainder=False)

    val_set_images = tf.data.Dataset.list_files(val_set_list[0], shuffle=False)
    val_set_masks = tf.data.Dataset.list_files(val_set_list[1], shuffle=False)
    val_set = tf.data.Dataset.zip((val_set_images, val_set_masks))

    val_set = val_set.map(process_path, num_parallel_calls=PARALLEL_CALLS)
    val_set = val_set.map(central_crop, num_parallel_calls=PARALLEL_CALLS)
    val_set = val_set.batch(batch_size, drop_remainder=False)

    test_set_images = tf.data.Dataset.list_files(test_set_list[0], shuffle=False)
    test_set_masks = tf.data.Dataset.list_files(test_set_list[1], shuffle=False)
    test_set = tf.data.Dataset.zip((test_set_images, test_set_masks))

    test_set = test_set.map(process_path, num_parallel_calls=PARALLEL_CALLS)
    test_set = test_set.batch(batch_size, drop_remainder=False)

    return train_set, val_set, test_set


def residual_upblock(filters, size, block_name, norm_type='batchnorm', apply_dropout=False):
    # TODO:
    result = tf.keras.Sequential(name=block_name)
    result.add(tf.keras.layers.UpSampling2D(2))

    result.add(tf.keras.layers.Conv2D(filters, size, padding='same'))
    result.add(tf.keras.layers.ReLU())

    result.add(tf.keras.layers.Add([]))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.3))


def simple_upblock_func(input_layer, filters, size, block_name, norm_type='batchnorm', apply_dropout=False):
    x = tf.keras.layers.UpSampling2D(2, name=block_name)(input_layer)

    x = tf.keras.layers.Conv2D(filters, size, padding='same')(x)
    x = tf.keras.layers.ReLU()(x)

    if norm_type.lower() == 'batchnorm':
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters, size, padding='same')(x)
    x = tf.keras.layers.ReLU()(x)

    if norm_type.lower() == 'batchnorm':
        x = tf.keras.layers.BatchNormalization()(x)

    if apply_dropout:
        x = tf.keras.layers.Dropout(0.3)(x)

    return x


def simple_upblock(filters, size, block_name, norm_type='batchnorm', apply_dropout=False):
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
    Returns:
    Upsample Sequential Model
    """

    result = tf.keras.Sequential(name=block_name)
    result.add(tf.keras.layers.UpSampling2D(2))

    result.add(tf.keras.layers.Conv2D(filters, size, padding='same'))
    result.add(tf.keras.layers.ReLU())

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.Conv2D(filters, size, padding='same'))
    result.add(tf.keras.layers.ReLU())

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.3))

    return result


def get_backbone_outputlayers(backbone, layer_names):
    layers = [backbone.get_layer(name).output for name in layer_names]
    return layers


def create_backbone(name='vgg19', set_trainable=True):
    if name == 'vgg19':
        backbone = tf.keras.applications.VGG19(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)
    elif name == 'resnet50':
        backbone = tf.keras.applications.ResNet50(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)
    elif name == 'resnet50v2':
        backbone = tf.keras.applications.ResNet50V2(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)
    elif name == 'mobilenetv2':
        backbone = tf.keras.applications.MobileNetV2(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)
    elif name == 'resnet101':
        backbone = tf.keras.applications.ResNet101(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)
    else:
        raise ValueError('No Backbone for Name "{}" defined \nPossible Names are: {}'.format(name, list(
            BACKBONE_LAYER_NAMES.keys())))
    layers = get_backbone_outputlayers(backbone, BACKBONE_LAYER_NAMES[name])
    extracted_backbone = tf.keras.Model(inputs=backbone.input, outputs=layers, name='backbone_' + name)
    extracted_backbone.trainable = set_trainable

    return extracted_backbone


def create_backbone_func(name='vgg19', set_trainable=True):
    if name == 'vgg19':
        backbone = tf.keras.applications.VGG19(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)
    elif name == 'resnet50':
        backbone = tf.keras.applications.ResNet50(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)
    elif name == 'resnet50v2':
        backbone = tf.keras.applications.ResNet50V2(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)
    elif name == 'mobilenetv2':
        backbone = tf.keras.applications.MobileNetV2(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)
    elif name == 'resnet101':
        backbone = tf.keras.applications.ResNet101(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)
    else:
        raise ValueError('No Backbone for Name "{}" defined \nPossible Names are: {}'.format(name, list(
            BACKBONE_LAYER_NAMES.keys())))
    backbone.trainable = set_trainable
    return backbone


def create_decoder():
    decoder = [
        simple_upblock(512, 3, 'up_stack512'),  # 4x4 -> 8x8
        simple_upblock(256, 3, 'up_stack256'),  # 8x8 -> 16x16
        simple_upblock(128, 3, 'up_stack128'),  # 16x16 -> 32x32
        simple_upblock(64, 3, 'up_stack64'),  # 32x32 -> 64x64
    ]
    return decoder


def segmentation_model_func(output_channels, backbone_name, backbone_trainable=True):
    down_stack = create_backbone_func(name=backbone_name, set_trainable=backbone_trainable)

    skips = [down_stack.get_layer(BACKBONE_LAYER_NAMES[backbone_name][0]).output,
             down_stack.get_layer(BACKBONE_LAYER_NAMES[backbone_name][1]).output,
             down_stack.get_layer(BACKBONE_LAYER_NAMES[backbone_name][2]).output,
             down_stack.get_layer(BACKBONE_LAYER_NAMES[backbone_name][3]).output,
             down_stack.get_layer(BACKBONE_LAYER_NAMES[backbone_name][4]).output]

    up_stack_filters = [64, 128, 256, 512]

    x = skips[-1]
    skips = reversed(skips[:-1])
    up_stack_filters = reversed(up_stack_filters)

    # Upsampling and establishing the skip connections
    for skip, filters in zip(skips, up_stack_filters):
        x = simple_upblock_func(x, filters, 3, 'up_stack' + str(filters))
        x = tf.keras.layers.Concatenate()([x, skip])

    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax', padding='same', name='final_output')(x)

    return tf.keras.Model(inputs=down_stack.layers[0].input, outputs=x)


def segmentation_model_seq(output_channels, backbone_name, backbone_trainable=True):
    down_stack = create_backbone(name=backbone_name, set_trainable=backbone_trainable)
    up_stack = create_decoder()

    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax', padding='same', name='final_output')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def display(image, mask, prediction=None):
    if prediction is None:
        _, ax = plt.subplots(1, 2, figsize=(15, 15))
    else:
        _, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(image)
    ax[0].set_title('image')
    ax[0].axis('off')
    ax[1].imshow(mask)
    ax[1].set_title('mask')
    ax[1].axis('off')
    if prediction is not None:
        ax[2].imshow(prediction)
        ax[2].set_title('prediction')
        ax[2].axis('off')
    plt.tight_layout()


def show(dataset, model=None, rows=1, threshold=0.5):
    for batch in dataset.shuffle(512).take(rows):
        if model is None:
            image, mask = batch[0][0], batch[1][0]
            tmp_mask = mask.numpy().copy()
            tmp_mask[:, :, 2] = 0
            overlay = cv2.add(image.numpy().astype(float), np.multiply(tmp_mask, 0.5).astype(float))
            overlay = np.clip(overlay, 0, 1)
            display(image, mask, overlay)
        else:
            prediction = model.predict(batch[0]) > threshold
            image, mask, prediction = batch[0][0], batch[1][0], prediction[0].astype(float)
            display(image, mask, prediction)


def get_dice_score(msk, pred, skip_background=True):
    if skip_background:
        msk = msk[..., 0:2]
        pred = pred[..., 0:2]

    batch_size = msk.shape[0]
    metric = []

    for batch in range(batch_size):
        m, p = msk[batch], pred[batch]
        intersection = np.logical_and(m, p)
        denominator = np.sum(m) + np.sum(p)
        if denominator == 0.0:
            denominator = np.finfo(float).eps
        dice_score = 2. * np.sum(intersection) / denominator
        metric.append(dice_score)

    return np.mean(metric)


def my_dice_metric_hemp(label, pred):
    return tf.py_function(get_dice_score, [label > 0.5, pred > 0.5], tf.float32)


def my_dice_metric_all(label, pred):
    return tf.py_function(get_dice_score, [label > 0.5, pred > 0.5, False], tf.float32)


def gather_channels(*xs, indexes=None, **kwargs):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes, **kwargs) for x in xs]
    return xs


def get_reduce_axes(per_image, backend=tf.keras.backend, **kwargs):
    axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes


def round_if_needed(x, threshold, backend=tf.keras.backend, **kwargs):
    if threshold is not None:
        x = backend.greater(x, threshold)
        x = backend.cast(x, backend.floatx())
    return x


def average(x, per_image=False, class_weights=None, backend=tf.keras.backend, **kwargs):
    if per_image:
        x = backend.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return backend.mean(x)


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_indexes=None, backend=tf.keras.backend, **kwargs):
    r"""Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
    """
    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    # Calculate focal loss
    loss = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))

    return backend.mean(loss)


def f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None,
            backend=tf.keras.backend, **kwargs):
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:
    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}
    The formula in terms of *Type I* and *Type II* errors:
    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}
    where:
        TP - true positive;
        FP - false positive;
        FN - false negative;
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        beta: f-score coefficient
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round
    Returns:
        F-score in range [0, 1]
    """

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # calculate score
    tp = backend.sum(gt * pr, axis=axes)
    fp = backend.sum(pr, axis=axes) - tp
    fn = backend.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


def dice_loss(gt, pr):
    return 1 - f_score(gt, pr, class_weights=np.array([0.5, 0.5, 1.]), smooth=1.0)


def focal_loss(gt, pr):
    return categorical_focal_loss(gt, pr)


def cce_loss(gt, pr):
    return tf.keras.losses.categorical_crossentropy(gt, pr, label_smoothing=0.3)


def dice_focal_loss(gt, pr, dice_weight=1., focal_weight=1.):
    return dice_weight * dice_loss(gt, pr) + focal_weight * focal_loss(gt, pr)


def dice_cce(gt, pr, dice_weight=1., cce_weight=1.):
    return dice_weight * dice_loss(gt, pr) + cce_weight * cce_loss(gt, pr)


class StepDecay():
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)

        # return the learning rate
        print(" Learning Rate: " + str(float(alpha)))
        return float(alpha)


def visualize_layers(input_img, input_msk, model, outputs, shift=0):
    fig, ax = plt.subplots(len(outputs), 5, figsize=(14, 12))

    input_msk[:, :, 2] = 0
    out_img = cv2.add(input_img.astype(float), np.multiply(input_msk, 0.3).astype(float))
    out_img = np.clip(out_img, 0, 1)

    for j, output in enumerate(outputs):
        submodel = tf.keras.models.Model([model.inputs[0]], [model.get_layer(output).output])
        pred = submodel.predict(input_img.reshape(1, 384, 384, 3))
        plt.yticks()
        channels = []
        stds = []
        for channel in range(pred.shape[-1]):
            layer = pred.squeeze()[:, :, channel]
            stds.append(np.std(layer))
            channels.append(layer)

        if shift == 0:
            stds = sorted(range(len(stds)), key=lambda x: stds[x])[-4:]
        else:
            stds = sorted(range(len(stds)), key=lambda x: stds[x])[-4 - shift:-shift]

        channels = [channels[i] for i in stds]
        channels = np.stack(channels, 0)

        for c in range(4):
            ax[j, c].imshow(skimage.filters.gaussian(channels[c], sigma=0.1), cmap='jet', aspect='auto')

        ax[j, 4].imshow(out_img, aspect='auto')

        for a in ax.flat:
            a.set(ylabel=output)
            a.set(xlabel="Layer response")

        ax[j, 4].set_xlabel("Input Image")

        for a in ax.flat:
            a.label_outer()

    plt.tight_layout(pad=2.)

    from rasterio.features import shapes