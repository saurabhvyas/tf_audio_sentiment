
# coding: utf-8

# In[14]:

# use tf dataset api to load dataset 
# prior to this , use downloadv3.sh and post_download.sh


# self notes
# custom function must strictly pass numpy dtype as input and numpy dtype as output ( not tensors)


# In[15]:

import tensorflow  as tf
import numpy as np


# In[3]:

import sys

sys.path.insert(0, '/media/saurabh/New Volume/tf_audio_sentiment/pyAudioAnalysis')

from  pyAudioAnalysis  import audioBasicIO
from  pyAudioAnalysis import audioFeatureExtraction


# In[4]:

timesteps=1000 # max timesteps
data_base_directory='/media/saurabh/New Volume/tf_audio_sentiment/data_test/datav2/'


# In[5]:

def audio_to_features(fileurl):
    
    [Fs, x] = audioBasicIO.readAudioFile(fileurl);
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
    
    a=F[:21,:]
    b=F[33]

    b=np.expand_dims(b, axis=0)

    #print(b.shape)


    d = np.concatenate( (a, b), axis=0)
    #print(d.shape)
    d = np.swapaxes(d, 0, 1)
    return d


# In[6]:

# define a function, that pads audio so that audio frames = max frames

def pad(input):
   # print(input.shape[0])
     if input.shape[0] < timesteps:
        
        diff = timesteps - input.shape[0]
        
        # pad and return input
        return np.pad(input,((0,diff),(0,0)), mode="constant")
    
     elif input.shape[0] > timesteps:
        
        return input[:timesteps,:]


# In[7]:

def process_function(input):
  
  

  path=data_base_directory + str(input,'utf-8')
  print(str(input,'utf-8'))

  if 'A' in str(input,'utf-8'):
        y=0
  elif 'B'  in str(input,'utf-8'):
        y=1

  input=pad(audio_to_features(path))

  
    
  return  input,y


# In[ ]:




# In[8]:

logfilepath='/media/saurabh/New Volume/tf_audio_sentiment/data_test/dataset.txt'
dataset = tf.contrib.data.TextLineDataset(logfilepath).shuffle(buffer_size=1000) #.map(map_function)

dataset = dataset.map(
    lambda filename: tuple(tf.py_func(
        process_function, [filename],  [tf.double,tf.int64])))

dataset=dataset.batch(3)


# In[9]:

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


# In[13]:

#with tf.Session() as sess:
 #   sess.run(iterator.initializer)
  #  for i in range(1000):
   #   value = sess.run(next_element[1])
    #  print(value)


# In[ ]:




# In[ ]:



