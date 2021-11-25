#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import splitfolders
from fastai.vision.all import *
from fastai.data.all import *
from fastai.vision.widgets import *


# In[2]:


torch.cuda.set_device(6)


# # Skip section

# In[ ]:


category_names = os.listdir('/home/alao/ml/swedish_leaf_dataset')
category_names


# In[ ]:


input_folder = '/home/alao/ml/swedish_leaf_dataset'


# In[ ]:


#split the dataset 80:20
splitfolders.ratio(input_folder, output="swl_data_split", seed=42, ratio=(.8,.2,), group_prefix=None)


# # Continue Section

# In[3]:


path = '/home/alao/ml/swl_data_split'
path


# In[ ]:


fnames = get_image_files(path)
fnames


# In[ ]:


im = PILImage.create(fnames[20])
im.show()
im.size


# In[ ]:


rsz = Resize(128, method=method)
#     show_image(rsz(img, split_idx=0), ctx=ax, title=method);
rd = rsz(im)
show_image(rd)
rd.size


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'Resize')


# # Build Datablocks and Train Model

# In[4]:



plants = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=GrandparentSplitter(train_name='train', valid_name='val'),
    get_y=parent_label,
    item_tfms=Resize(256))


# In[5]:


dls = plants.dataloaders(path, bs=32)


# # ResNet 34 Architecture

# In[6]:



learn_res34 = cnn_learner(dls, resnet34, metrics=(error_rate, accuracy))
learn_res34.fine_tune(5)


# In[8]:


learn_res34.save('SWL/resnet-34')


# # Evaluate Model

# In[9]:



interp_res34 = ClassificationInterpretation.from_learner(learn_res34)
interp_res34.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[10]:


learn_res34.summary()


# In[ ]:


learn.show_results()


# In[ ]:



interp.plot_top_losses(5)


# # ResNet 18 Architecture

# In[11]:



learn_res18 = cnn_learner(dls, resnet18, metrics=(error_rate, accuracy))
learn_res18.fine_tune(5)


# In[13]:


learn_res18.save('SWL/resnet-18')


# # Evaluate Model

# In[14]:



interp_res18 = ClassificationInterpretation.from_learner(learn_res18)
interp_res18.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[15]:


learn_res18.summary()


# # ResNet 50 Architecture

# In[18]:



learn_res50 = cnn_learner(dls, resnet50, metrics=(error_rate, accuracy))
learn_res50.fine_tune(5)


# In[19]:


learn_res50.save('SWL/resnet-50')


# # Evaluate Model

# In[20]:



interp_res50 = ClassificationInterpretation.from_learner(learn_res50)
interp_res50.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[21]:


learn_res50.summary()


# # ResNet 101 Architecture

# In[22]:



learn_res101 = cnn_learner(dls, resnet101, metrics=(error_rate, accuracy))
learn_res101.fine_tune(5)


# In[23]:


learn_res101.save('SWL/resnet-101')


# # Evaluate Model

# In[24]:



interp_res101 = ClassificationInterpretation.from_learner(learn_res101)
interp_res101.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[25]:


learn_res101.summary()


# # ResNet 152 Architecture

# In[26]:



learn_res152 = cnn_learner(dls, resnet152, metrics=(error_rate, accuracy))
learn_res152.fine_tune(5)


# In[27]:


learn_res152.save('SWL/resnet-152')


# # Evaluate Model

# In[28]:



interp_res152 = ClassificationInterpretation.from_learner(learn_res152)
interp_res152.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[29]:


learn_res152.summary()


# # AlexNet Architecture

# In[30]:



learn_alex = cnn_learner(dls, alexnet, metrics=(error_rate, accuracy))
learn_alex.fine_tune(5)


# In[31]:


learn_alex.save('SWL/alexnet')


# # Evaluate Model

# In[32]:



interp_alex = ClassificationInterpretation.from_learner(learn_alex)
interp_alex.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[33]:


learn_alex.summary()


# # Make prediction

# In[ ]:



# hide_output
cleaner = ImageClassifierCleaner(learn)
cleaner


# 

# In[ ]:


#hide_output
uploader = widgets.FileUpload()
uploader


# In[ ]:



img = PILImage.create(uploader.data[0])
is_acer,_,probs = learn.predict(img)
print(f"What is this?: {is_acer}.")


# In[ ]:




