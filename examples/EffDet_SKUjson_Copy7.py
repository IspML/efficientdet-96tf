#!/usr/bin/env python
# coding: utf-8

# # Train Efficientdet with Labelme Dataset
#
# A convenient API is provided to the final user to easily train and use a model, on your dataset created with labelme or a dataset formatted as labelme output. Ref: [GitHub Repository](https://github.com/Guillem96/efficientdet-tf).

# ## Import libs & Set parameters

# In[1]:

#@markdown ## `import *`
#%%capture

import json
from pathlib import Path
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import efficientdet

#@markdown EfficientDet compound scaling (Most times with D0 or D1 you'll be OK)
D = 0 #@param {type: "slider", min: 0, max: 7}
model_config = efficientdet.config.EfficientDetCompudScaling(D=D)
im_size = model_config.input_size
print('im_size:', im_size)

train_ratio = 0.8
batch_size = 2 #@param {type: "slider", min: 2, max: 64} ?!!
valdiate_freq = 10 #@param {type: "slider", min: 1, max: 10}#@markdown Compute COCO mAP every `valdiate_freq` epochs
#
epochs_ratio = 50 #!!!
epochsAdw, step_lenAdw, weight_decayAdw = 1*epochs_ratio, 1e-4, 4e-5 #epoches: min: 2, max: 120
#---
data_path = Path('/root/bw/datasets/SKU110K_fixed/')
root_data = data_path / 'JPEGImages'
class_names_file = data_path / 'SKU110K.names'
print(f'''{datetime_str()}::\n\tD: {D},batch_size: {batch_size}, epochs_nmb: {epochsAdw}
\tclass_names_file: {class_names_file},\n\tdata_path: {root_data}''')
classes, class2idx = efficientdet.utils.io.read_class_names(class_names_file)


# ## Prepare the data & Train the model

# In[2]:

ds = efficientdet.data.labelme.build_dataset(annotations_path=root_data, images_path=root_data, class2idx=class2idx, im_input_size=im_size, shuffle=True)
ds_len = sum(1 for _ in ds)
train_len = int(ds_len * train_ratio) # xxx% of validation data
train_ds = ds.take(train_len) # Take the first instances
valid_ds = ds.skip(train_len) # Skip the first instances, so the sets do not intersect
# Data augmentation on training set
train_ds = (train_ds
            .map(efficientdet.augment.RandomHorizontalFlip())
            .map(efficientdet.augment.RandomCrop()))

padded_image_shape = (*im_size, 3)
padded_labels_shape = (None,)
boxes_padded_shape = (None, 4)

train_ds = train_ds.padded_batch(batch_size=batch_size,
                                 padded_shapes=(padded_image_shape, (padded_labels_shape, boxes_padded_shape)),
                                 padding_values=(0., (-1, -1.)))

valid_ds = valid_ds.padded_batch(batch_size=batch_size,
                                 padded_shapes=(padded_image_shape, (padded_labels_shape, boxes_padded_shape)),
                                 padding_values=(0., (-1, -1.)))

wrapped_train_ds = efficientdet.wrap_detection_dataset(train_ds, im_size=im_size, num_classes=len(class2idx))
wrapped_valid_ds = efficientdet.wrap_detection_dataset(valid_ds, im_size=im_size, num_classes=len(class2idx))

# === === === === ===

# create the losses
clf_loss = efficientdet.losses.EfficientDetFocalLoss()
reg_loss = efficientdet.losses.EfficientDetHuberLoss()

# the optimizer
# We have to calculate the steps per epoch in order to create the learning rate scheduler
steps_per_epoch = sum(1 for _ in wrapped_train_ds)
lrAdw = efficientdet.optim.WarmupCosineDecayLRScheduler(step_lenAdw, warmup_steps=steps_per_epoch, decay_steps=steps_per_epoch * (epochsAdw - 1), alpha=0.1)
optimizerAdw = tfa.optimizers.AdamW(learning_rate=lrAdw, weight_decay=weight_decayAdw)

# build the model.
model = efficientdet.EfficientDet(D=D, num_classes=len(class2idx), training_mode=True, weights='D0-VOC', custom_head_classifier=True)
model.compile(loss=[reg_loss, clf_loss], optimizer=optimizerAdw, loss_weights=[1., 1.])
model.build([None, *im_size, 3])
model.summary()

# train the model
print(f'TrainingAdw {datetime_str()}')
callbacks = [efficientdet.callbacks.COCOmAPCallback(valid_ds, class2idx, validate_every=valdiate_freq)]
model.fit(wrapped_train_ds, validation_data=wrapped_valid_ds, epochs=epochsAdw, callbacks=callbacks)
print(f'TrainedAdw! {datetime_str()}')

# ## Predict the boxes

# In[3]:

model.training_mode = False
model.filter_detections.score_threshold = 0.2

image, _ = next(iter(valid_ds.unbatch().shuffle(20).take(1)))
print(f"Predicting {datetime_str()}")
bbs, labels, scores = model(tf.expand_dims(image, 0), training=False)
print(f"Predicted! {datetime_str()}")
image_n = efficientdet.data.preprocess.unnormalize_image(image)

# Covert dataset labels to names
labels_name = [classes[l] for l in labels[0].numpy().tolist()]

# For each different name get a color
colors = efficientdet.visualizer.colors_per_labels(labels_name)

# Get a Pillow image with drew boxes, and automatically plot it
efficientdet.visualizer.draw_boxes(image_n, bbs[0], labels_name, scores=scores[0], colors=colors)

