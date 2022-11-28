# PHO-LID (Accepted to Interspeech 2022 as an oral presentation)[https://arxiv.org/abs/2203.12366]
PHO-LID: A Unified Model to Incorporate Acoustic-Phonetic and Phonotactic Information for Language Identification  

Some people have successfully reproduced the results (and achieved even better ones). There are no specific hyper-parameters or random seed, just remember to have a low number of nega_frame (see the config.json).  
  
DDP is not applicable during training, will try to update soon. Single-GPU training works well.  
An example to run it:
python train_PHOLID.py --json /home/tony/PHOLID_config_file.json  
   
The training data txt file should be look like:  
data0_path.npy(space)language_index0(space)T'_0  
data1_path.npy language_index1 T'_1  
data2_path.npy language_index2 T'_2  
data3_path.npy language_index3 T'_3  
Where the T'_0 is sequence length of the reshaped data from T x D to T' x 20 x D, 20 denotes 20 frames in each embedding. You may also prepare your own script and only use this model config.  
