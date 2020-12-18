#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
import pickle


# In[2]:


import data_utils
import glove_utils
import models
import display_utils
from goog_lm import LM


# In[3]:


import lm_data_utils
import lm_utils


# In[4]:


np.random.seed(1001)
tf.set_random_seed(1001)


# In[5]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[6]:


VOCAB_SIZE  = 50000
with open('aux_files/dataset_%d.pkl' %VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)


# In[7]:


doc_len = [len(dataset.test_seqs2[i]) for i in 
           range(len(dataset.test_seqs2))]


# In[8]:


dist_mat = np.load('aux_files/dist_counter_%d.npy' %VOCAB_SIZE)
# Prevent returning 0 as most similar word because it is not part of the dictionary
dist_mat[0,:] = 100000
dist_mat[:,0] = 100000

skip_list = np.load('aux_files/missed_embeddings_counter_%d.npy' %VOCAB_SIZE)


# ### Demonstrating how we find the most similar words

# In[9]:


for i in range(300, 305):
    src_word = i
    nearest, nearest_dist = glove_utils.pick_most_similar_words(src_word, dist_mat,20, 0.5)
        
    print('Closest to `%s` are:' %(dataset.inv_dict[src_word]))
    for w_id, w_dist in zip(nearest, nearest_dist):
          print(' -- ', dataset.inv_dict[w_id], ' ', w_dist)

    print('----')


# ### Preparing the dataset

# In[10]:


max_len = 250
train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')
train_y = np.array(dataset.train_y)
test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
test_y = np.array(dataset.test_y)


# ### Loading the sentiment analysis model

# In[13]:


tf.reset_default_graph()
if tf.get_default_session():
    sess.close()
sess = tf.Session()
batch_size = 1
lstm_size = 128
#max_len =  100

with tf.variable_scope('imdb', reuse=False):
    model = models.SentimentModel(batch_size=batch_size,
                           lstm_size = lstm_size,
                           max_len = max_len,
                           embeddings_dim=300, vocab_size=dist_mat.shape[1],is_train = False)
saver = tf.train.Saver()
saver.restore(sess, './models/imdb_model')


# ## Loading the Google Language model

# In[14]:


goog_lm = LM()


# #### demonstrating the GoogLM

# In[15]:


src_word = dataset.dict['play']
nearest, nearest_dist = glove_utils.pick_most_similar_words(src_word, dist_mat,20)
nearest_w = [dataset.inv_dict[x] for x in nearest]
print('Closest to `%s` are %s' %(dataset.inv_dict[src_word], nearest_w))


# In[16]:


prefix = 'is'
suffix = 'with'
lm_preds = goog_lm.get_words_probs(prefix, nearest_w, suffix)
print('most probable is ', nearest_w[np.argmax(lm_preds)])


# ## Try Attack

# In[17]:


from attacks import GeneticAtack


# ## Main Attack 

# In[18]:


pop_size = 60
n1 = 8

with tf.variable_scope('imdb', reuse=True):
    batch_model = models.SentimentModel(batch_size=pop_size,
                           lstm_size = lstm_size,
                           max_len = max_len,
                           embeddings_dim=300, vocab_size=dist_mat.shape[1],is_train = False)
    
with tf.variable_scope('imdb', reuse=True):
    neighbour_model = models.SentimentModel(batch_size=n1,
                           lstm_size = lstm_size,
                           max_len = max_len,
                           embeddings_dim=300, vocab_size=dist_mat.shape[1],is_train = False)
ga_atttack = GeneticAtack(sess, model, batch_model, neighbour_model, dataset, dist_mat, 
                                  skip_list,
                                  goog_lm, max_iters=30, 
                                   pop_size=pop_size,
                                  
                                  n1 = n1,
                                  n2 = 4,
                                 use_lm = True, use_suffix=False)


# In[19]:


SAMPLE_SIZE = 5000
TEST_SIZE = 200
test_idx = np.random.choice(len(dataset.test_y), SAMPLE_SIZE, replace=False)
test_len = []
for i in range(SAMPLE_SIZE):
    test_len.append(len(dataset.test_seqs2[test_idx[i]]))
print('Shortest sentence in our test set is %d words' %np.min(test_len))

test_list = []
orig_list = []
orig_label_list = []
adv_list = []
dist_list = []

for i in range(SAMPLE_SIZE):
    x_orig = test_x[test_idx[i]]
    orig_label = test_y[test_idx[i]]
    orig_preds=  model.predict(sess, x_orig[np.newaxis, :])[0]
    # print(orig_label, orig_preds, np.argmax(orig_preds))
    if np.argmax(orig_preds) != orig_label:
        #print('skipping wrong classifed ..')
        #print('--------------------------')
        continue
    x_len = np.sum(np.sign(x_orig))
    if x_len >= 100:
        #print('skipping too long input..')
        #print('--------------------------')
        continue
    # if np.max(orig_preds) < 0.90:
    #    print('skipping low confidence .. \n-----\n')
    #    continue
    print('****** ', len(test_list) + 1, ' ********')
    test_list.append(test_idx[i])
    orig_list.append(x_orig)
    target_label = 1 if orig_label == 0 else 0
    orig_label_list.append(orig_label)
    x_adv = ga_atttack.attack( x_orig, target_label)
    adv_list.append(x_adv)
    if x_adv is None:
        print('%d failed' %(i+1))
        dist_list.append(100000)
    else:
        num_changes = np.sum(x_orig != x_adv)
        print('%d - %d changed.' %(i+1, num_changes))
        dist_list.append(num_changes)
        # display_utils.visualize_attack(sess, model, dataset, x_orig, x_adv)
    print('--------------------------')
    if (len(test_list)>= TEST_SIZE):
        break


# ## Compute Attack success rate

# In[20]:


orig_len = [np.sum(np.sign(x)) for x in orig_list]
normalized_dist_list = [dist_list[i]/orig_len[i] for i in range(len(orig_list)) ]


# In[24]:


SUCCESS_THRESHOLD  = 0.25
successful_attacks = [x < SUCCESS_THRESHOLD for x in normalized_dist_list]
print('Attack success rate : {:.2f}%'.format(np.mean(successful_attacks)*100))
print('Median percentange of modifications: {:.02f}% '.format(
    np.median([x for x in normalized_dist_list if x < 1])*100))
print('Mean percentange of modifications: {:.02f}% '.format(
    np.mean([x for x in normalized_dist_list if x < 1])*100))


# In[26]:


visual_idx = np.random.choice(len(orig_list))
display_utils.visualize_attack(sess, model, dataset, orig_list[visual_idx], adv_list[visual_idx])


# In[27]:


visual_idx = np.random.choice(len(orig_list))
display_utils.visualize_attack(sess, model, dataset, orig_list[visual_idx], adv_list[visual_idx])


# In[28]:


## Save success
with open('attack_results_final.pkl', 'wb') as f:
    pickle.dump((test_list, orig_list, orig_label_list, adv_list, normalized_dist_list), f)
    

