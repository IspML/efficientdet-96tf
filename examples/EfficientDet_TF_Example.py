#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Guillem96/efficientdet-tf/blob/master/examples/EfficientDet_TF_Example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


#get_ipython().system('pip install git+https://github.com/Guillem96/efficientdet-tf')
#get_ipython().system('wget -O sample.jpg https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSRcGUTmpIASquxz8ocDSzHMTzseDg5eH_1l_UOsli1ENbkvAOX&usqp=CAU')


# In[ ]:


from efficientdet import visualizer
from efficientdet import EfficientDet
from efficientdet.data import preprocess
from efficientdet.utils.io import load_image
from efficientdet.data.voc import IDX_2_LABEL

import tensorflow as tf


# In[3]:


model = EfficientDet.from_pretrained('D0-VOC', score_threshold=.3)
image_size = model.config.input_size


# In[4]:


image = load_image('sample.jpg', image_size)
image.shape


# In[ ]:


n_image = preprocess.normalize_image(image)
n_image = tf.expand_dims(n_image, 0)


# In[ ]:


predictions = model(n_image, training=False)
boxes, labels, scores = predictions
labels = [IDX_2_LABEL[o] for o in labels[0]]


# In[7]:


colors = visualizer.colors_per_labels(labels)
visualizer.draw_boxes(image, 
                      boxes=boxes[0], 
                      labels=labels, 
                      scores=scores[0], 
                      colors=colors)

