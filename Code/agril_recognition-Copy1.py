#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from fastai.vision.all import *
from fastai.data.all import *
from fastai.vision.widgets import *


# In[2]:


torch.cuda.set_device(6)


# In[3]:


path = '/home/alao/ml/AgrilPlant_Dataset_2017'
# path = os.listdir('/home/ele_group_4/ml/Olaniyi/swl_data_split')

path


# In[ ]:


fnames = get_image_files(path)
fnames


# In[ ]:


im = PILImage.create(fnames[2])
im.show()
im.size


# In[ ]:


rsz = Resize(128, method=method)
#     show_image(rsz(img, split_idx=0), ctx=ax, title=method);
rd = rsz(im)
show_image(rd)
rd.size


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'DataBlock.dataloaders')


# # Build Datablocks and Train Model

# In[4]:



plants = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=GrandparentSplitter(train_name='train', valid_name='test'),
    get_y=parent_label,
    item_tfms=Resize(256))


# In[5]:


dls = plants.dataloaders(path, bs=32)


# # ResNet 34 Model

# In[6]:



learn_res34 = cnn_learner(dls, resnet34, metrics=(error_rate, accuracy))
learn_res34.fine_tune(5)


# In[7]:


learn_res34.save('resnet-34')


# # Evaluate Model

# In[8]:



interp_res34 = ClassificationInterpretation.from_learner(learn_res34)
interp_res34.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[26]:


learn_res34.summary()


# In[9]:


learn_res34.show_results()


# In[10]:



interp_res34.plot_top_losses(5)


# # AlexNet Architecture

# In[ ]:





# In[12]:



learn_alex = cnn_learner(dls, alexnet, metrics=(error_rate, accuracy))
learn_alex.fine_tune(5)


# In[13]:


learn_alex.save('alexnet')


# # Evaluate Model

# In[14]:



interp_alex = ClassificationInterpretation.from_learner(learn_alex)
interp_alex.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[30]:


learn_alex.summary()


# In[15]:


learn_alex.show_results()


# In[16]:



interp_alex.plot_top_losses(5)


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'cnn_learner')


# # ResNet 152 Architecture

# In[17]:



learn_res152 = cnn_learner(dls, resnet152, metrics=(error_rate, accuracy))
learn_res152.fine_tune(5)


# In[18]:


learn_res152.save('resnet-152')


# # Evaluate Model

# In[19]:



interp_res152 = ClassificationInterpretation.from_learner(learn_res152)
interp_res152.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[31]:


learn_res152.summary()


# In[20]:


learn_res152.show_results()


# In[21]:



interp_res152.plot_top_losses(5)


# # ResNet 101 Architecture

# In[22]:



learn_res101 = cnn_learner(dls, resnet101, metrics=(error_rate, accuracy))
learn_res101.fine_tune(5)


# In[23]:


learn_res101.save('resnet-101')


# # Evaluate Model

# In[24]:



interp_res101 = ClassificationInterpretation.from_learner(learn_res101)
interp_res101.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[32]:


learn_res101.summary()


# # ResNet 50 Architecture

# In[25]:



learn_res50a = cnn_learner(dls, resnet50, metrics=(error_rate, accuracy))
learn_res50a.fine_tune(5)


# In[27]:


learn_res50a.save('resnet-50')


# # Evaluate Model

# In[28]:



interp_res50a = ClassificationInterpretation.from_learner(learn_res50a)
interp_res50a.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[33]:


learn_res50a.summary()


# # ResNet 18 Architecture

# In[29]:



learn_res18 = cnn_learner(dls, resnet18, metrics=(error_rate, accuracy))
learn_res18.fine_tune(5)


# In[34]:


learn_res18.save('resnet-18')


# # Evaluate Model

# In[35]:



interp_res18 = ClassificationInterpretation.from_learner(learn_res18)
interp_res18.plot_confusion_matrix(dpi = 90,figsize = (6,6))


# In[36]:


learn_res18.summary()


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
is_apple,_,probs = learn.predict(img)
print(f"What is this?: {is_apple}.With a probability of: ")


# # Fine tune Model

# In[ ]:


learn.lr_find()


# In[ ]:


learn.fine_tune(4, 6.309573450380412e-07)


# # Alex Architecture

# In[ ]:



alex_learn = cnn_learner(dls, alexnet, metrics=(error_rate, accuracy))
alex_learn.fine_tune(4)


# In[ ]:





# # Evaluate Model

# In[ ]:



alex_interp = ClassificationInterpretation.from_learner(alex_learn)
alex_interp.plot_confusion_matrix()


# In[ ]:


alex_learn.lr_find()


# In[ ]:


alex_learn.show_results()


# In[ ]:



alex_interp.plot_top_losses(5)


# In[ ]:





# In[ ]:


#hide_output
uploader = widgets.FileUpload()
uploader


# In[ ]:



img = PILImage.create(uploader.data[0])
is_apple,_,probs = alex_learn.predict(img)
print(f"Is this a banana?: {is_apple}.")


# In[ ]:




