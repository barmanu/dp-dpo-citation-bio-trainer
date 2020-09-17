{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/joshib/.cache/pypoetry/virtualenvs/dp-dpo-citation-bio-trainer-FaGiNfZ9-py3.7/lib/python3.7/site-packages/pandas/compat/__init__.py:120: UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.\n",
      "  warnings.warn(msg)\n"
     ]
    }
   ],
   "source": [
    "### for reproducible experiments\n",
    "from numpy.random import seed\n",
    "seed(42)\n",
    "from tensorflow.random import set_seed\n",
    "set_seed(42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/joshib/.cache/pypoetry/virtualenvs/dp-dpo-citation-bio-trainer-FaGiNfZ9-py3.7/lib/python3.7/site-packages/scipy/sparse/sparsetools.py:21: DeprecationWarning: `scipy.sparse.sparsetools` is deprecated!\n",
      "scipy.sparse.sparsetools is a private module for scipy.sparse, and should not be used.\n",
      "  _deprecated()\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import git\n",
    "import mlflow.keras\n",
    "import os, sys\n",
    "from gensim.models.wrappers import FastText\n",
    "\n",
    "#import tensorflow_hub as hub\n",
    "from tensorflow.keras.callbacks import *\n",
    "from tensorflow.keras.optimizers import *\n",
    "\n",
    "from sklearn.metrics import *\n",
    "from sklearn.model_selection import train_test_split\n",
    "import matplotlib.pyplot as plt\n",
    "#import random\n",
    "%matplotlib inline\n",
    "pd.set_option('display.max_rows', None)\n",
    "from tqdm import tqdm\n",
    "sys.path.append('../citation_bio_trainer')\n",
    "#from feature.SpacyFeaturizer import get_spacy_feats_from_text\n",
    "from feature_window.Featurizer_window import Featurizer_window\n",
    "from util.Utils import calulate_ser_jer, load_from_folder, pad_sequences, load_embedding_matrix, evaluate, log_mlflow_results\n",
    "import warnings\n",
    "from model.FTLSTM import calulate_ser_jer, get_model, plot_output\n",
    "warnings.filterwarnings(\"ignore\", category=DeprecationWarning) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## LSTM with random embedding model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 2min 34s, sys: 1.15 s, total: 2min 35s\n",
      "Wall time: 2min 46s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "train_data_path = '/nlp/cs_mixed_20k/cs_mixed_20k_train/'\n",
    "test_data_path  = '/nlp/cs_mixed_20k/cs_mixed_20k_test/'\n",
    "eval_without_intra_newline_path  = '/nlp/eval_data_spacy_tokenized_extra_space_removed/'\n",
    "\n",
    "sentences_train, sent_tags_train = load_from_folder(train_data_path)\n",
    "sentences_test, sent_tags_test = load_from_folder(test_data_path)\n",
    "sentences_eval1, sent_tags_eval1 = load_from_folder(eval_without_intra_newline_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'1 ) J. Chang et al . , “ 12.1 A 7 nm 256Mb SRAM in high - k metal - gate FinFET technology with write - assist circuitry for low - VMIN applications , ” 2017 IEEE International Solid - State Circuits Conference ( ISSCC ) , San Francisco , CA , 2017 , pp . 206 - 207 . \\n 2 ) T. Standaert et al . , “ BEOL process integration for the 7 nm technology node , ” 2016 IEEE International Interconnect Technology Conference / Advanced Metallization Conference ( IITC / AMC ) , San Jose , CA , 2016 , pp . 2 - 4 . \\n 3 ) \\n https://www-03.ibm.com/press/us/en/pressrelease/47301.wss \\n 4 ) https://newsroom.intel.com/news-releases/intel-supports-american-innovation-7-billion-investment-next-generation-semiconductor-factory-arizona/ \\n 5 ) Standard Test Methods of Bitumen and Bituminous Mixtures For Highway Engineering JTJ E20 - \\n 2011 , China Communications Press \\n , Beijing , 2011 ( in Chinese ) . \\n 6 ) Specifications for Design of Highway Asphalt Pavement JTG D50 - 2006 , China Communications Press , Beijing , \\n 2006 ( in Chinese ) . \\n 7 ) Bullough C , Gatzen C , Jakiel C , Koller M , Nowi A , Zunft S. Advanced adiabatic compressed air energy storage for the integration of wind energy . In : EWEC 2004 : Proceedings of the European wind energy conference . London , UK : November 22 - 5 2004 . \\n 8) Xu , S. , Nam , S.M. , Kim , J.H. , Das , R. , Choi , S.K. , Nguyen \\n , T.T. , Quan , X. , Choi , S.J. , Chung , C.H. , Lee , E.Y. , Lee , I.K. , Wiederkehr , A. , Wollheim , C.B. , Cha , S.K. , Park , K.S. , 2015 . Palmitate induces ER calcium depletion and apoptosis in mouse podocytes subsequent to mitochondrial oxidative stress . Cell Death Dis 6 , e1976 . \\n 9 ) Balistrocchi , M. , Grossi , G. , Bacchi , B. , 2013 . Deriving a practical analytical - probabilistic method to size flood routing reservoirs . \\n Adv . Water Resour \\n . , 62 , Part A , 37–46 . http://dx.doi.org/10.1016/j.advwatres.2013.09.018 . \\n 10 ) Bergmann , H. , Sackl , B. , 1989 . Determination of design flood hydrographs based on regional hydrological data . IAHS Publ.no . 181 , 261–269 . \\n 11 ) Butera , I. , Tanda , M.G. , 2006 . Analysing river bank seepage with a synthetic design hydrograph . P. I. Civil Eng.-Wat . M. 159 ( 2 ) , 119 - 127 \\n . http://dx.doi.org/10.1680/wama.2006.159.2.119Castellarin , A.,Kohnova , S.,Gaal , L.,Fleig , A.,Salinas , J.L.,Toumazis , A.,Kjeldsen , T.R.,Macdonald , N. , 2012.Review of applied - statistical methods for flood - frequency analysis in Europe . NERC / Centre for Ecology & Hydrology , 122pp . ( ESSEM COST Action ES0901 ) . \\n 12 ) Franchini , \\n M. , Galeati , G. , 2000 . Comparative analysis \\n of some \\n methods for deriving the expected flood reduction curve in the frequency domain . Hydrol . Earth Syst . Sci . 4 ( 1 ) 155–172 . http://dx.doi.org/10.5194/hess-4-155-2000 . \\n 13 ) Gräler , \\n B. , van den Berg , M.J. , Vandenberghe , S. , Petroselli , A. , Grimaldi , S. , De Baets , B. , Verhoest , N.E.C. , 2013 . Multivariate return periods in hydrology : a critical and practical review \\n focusing on synthetic design hydrograph estimation . Hydrol . Earth Syst . Sci . 17 , 1281–1296 . http://dx.doi.org/10.5194/hess-17-1281-2013 . \\n 14 ) Keifer , C.J. , Chu , H.H. 1957 . Synthetic storm pattern for drainage design , J. Hydr . Eng . Div- ASCE 83 ( 4),1332.1–1332.25 . \\n 15 ) NERC ( National Environmental Research \\n Council ) \\n , 1975 . Flood Studies Report , vol . 1 , London . \\n 16 ) \\n Salvadori , G. , De Michele , C. , 2004 . Frequency analysis via copulas \\n : theoretical aspects and applications to hydrological \\n events . Water Resour Res 40 ( 12 ) , WR003133 . http://dx.doi.org/10.1029/2004WR003133 . \\n 17 ) Sauquet , E. , Ramos , M.H. , Chapel , L. , Bernardara , P. , 2008 . Streamflow scaling properties : investigating characteristic scales from different statistical approaches . Hydrol . Process 22 ( 17 ) , 3462–3475 . http://dx.doi.org/10.1002/hyp.6952 . \\n 18 ) ECR , 2006 . European Commission Regulation No . 1907/2006 Concerning the Registration Evaluation Authorisation and Restriction of Chemicals ( REACH ) establishing a European Chemicals Agency Amending Directive 1999/45 / EC and Repealing Council Regulation ( EEC ) No . 793/93 and Commission Regulation ( EC ) No . 1488/94 as well as Council Directive 76/769 / EEC and Commission Directives 91/155 / EEC 93/67 / EEC 93/105 / EC and 2000/21 / EC . December 18 , 2006 . \\n 19 ) ECR , 2008 . European Commission Regulation No . 629/2008 Amending Regulation ( EC ) No . 1881/2006 Setting Maximum Levels for Certain Contaminants in Foodstuffs . \\n 20 ) ECR , 2011a . European Commission Regulation No . 835/2011 Amending Regulation ( EC ) No . 1881/2006 as Regards Maximum Levels for Polycyclic Aromatic Hydrocarbons in Foodstuffs . \\n 21 ) ECR , 2011b . European Commission Regulation No . 1259/2011 Amending Regulation ( EC ) No . 1881/2006 as Regards Maximum Levels for Dioxins , Dioxin - like PCBs and Non - dioxin - like PCBs in Foodstuffs . \\n 22 ) ECR , 2014 . European Commission Regulation No . 488/2014 Amending Regulation ( EC ) No . 1881/2006 as Regards Maximum Levels of Cadmium in Foodstuffs . \\n 23 ) D.R. Tobergte , S. Curtis , Environmentally Benign Photocatalysts Applications of Titanium Oxide - based Materials , n.d . doi:10.1017 / CBO9781107415324.004 . \\n 24 ) Zhu , Y. , Wang , H. , Jin , Y. , Wu , D. , & Zhou , L. ( 2006 ) . Texture synthesis for repairing damaged images . US Patent 7,012,624 . \\n'"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sentences_train[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "xtrain, xvalid, ytrain, yvalid = train_test_split(sentences_train, sent_tags_train, test_size=0.1, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "feat_config = {'max_vocab':100000,\n",
    "               'lstm_feats':True, \n",
    "               'spacy_feats':True, \n",
    "               'google_feats': False, \n",
    "               'parscit_feats': False,\n",
    "              'custom_feats': True, \n",
    "              'spacy_mode': 'production',\n",
    "              'window':100,\n",
    "              'step':10}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "loading train ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "15817it [00:16, 936.88it/s] \n",
      "100%|██████████| 15817/15817 [00:35<00:00, 442.72it/s] \n",
      "100%|██████████| 15817/15817 [00:08<00:00, 1890.95it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "total partitions = 1321\n",
      "CPU times: user 15min 52s, sys: 30.8 s, total: 16min 22s\n",
      "Wall time: 26min 10s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "featurizer = Featurizer_window(feat_config)\n",
    "print(\"loading train ...\")\n",
    "train_dict, tokenizer, maxlen = featurizer.fit_transform(xtrain, ytrain)\n",
    "# print(\"loading valid ...\")\n",
    "# valid_dict = featurizer.transform(xvalid, yvalid)\n",
    "# print(\"loading test ...\")\n",
    "# test_dict  = featurizer.transform(sentences_test, sent_tags_test)\n",
    "# print(\"loading evals ...\")\n",
    "# eval1_dict  = featurizer.transform(sentences_eval1, sent_tags_eval1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['featurizer.pkl']"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "joblib.dump(featurizer, 'featurizer.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%time\n",
    "# ### Load from file\n",
    "# maxlen = 3861\n",
    "# train_dict={}\n",
    "# train_dict['labels'] = np.load('/nlp/temp/train_dict_labels.npy')\n",
    "# train_dict['lstm_feats']=np.load('/nlp/temp/train_dict_lstm_feats.npy')\n",
    "# train_dict['spacy_num_feats'] = np.load('/nlp/temp/train_dict_spacy_feats.npy')\n",
    "\n",
    "# valid_dict={}\n",
    "# valid_dict['labels']=np.load('/nlp/temp/valid_dict_labels.npy')\n",
    "# valid_dict['lstm_feats']=np.load('/nlp/temp/valid_dict_lstm_feats.npy')\n",
    "# valid_dict['spacy_num_feats']=np.load('/nlp/temp/valid_dict_spacy_feats.npy')\n",
    "\n",
    "# test_dict={}\n",
    "# test_dict['labels'] = np.load('/nlp/temp/test_dict_labels.npy')\n",
    "# test_dict['lstm_feats']=np.load('/nlp/temp/test_dict_lstm_feats.npy')\n",
    "# test_dict['spacy_num_feats']=np.load('/nlp/temp/test_dict_spacy_feats.npy')\n",
    "\n",
    "# eval1_dict={}\n",
    "# eval1_dict['labels'] = np.load('/nlp/temp/eval1_dict_labels.npy')\n",
    "# eval1_dict['lstm_feats']=np.load('/nlp/temp/eval1_dict_lstm_feats.npy')\n",
    "# eval1_dict['spacy_num_feats']=np.load('/nlp/temp/eval1_dict_spacy_feats.npy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# def get_custom_feats(data_ls, maxlen):\n",
    "#     feat_arr = np.zeros((len(data_ls), maxlen, 5), dtype='int8') #[[[0]* 5]* maxlen]* len(data_ls)\n",
    "#     for ind in tqdm(range(len(data_ls))):\n",
    "#         sent_ls = data_ls[ind].split(\" \") \n",
    "#         for i in range(len(sent_ls) - 2):\n",
    "#             if i>= maxlen:\n",
    "#                 break\n",
    "#             if i == 0:\n",
    "#                 feat_arr[ind][i][0] = 1\n",
    "#             elif sent_ls[i-1] == '\\n':\n",
    "#                 if sent_ls[i].isdigit() and len(sent_ls[i]) <= 2 and sent_ls[i+1] in ('.', ')'):\n",
    "#                     feat_arr[ind][i][1] = 1\n",
    "#                 elif sent_ls[i].isalpha() and len(sent_ls[i]) == 1 and sent_ls[i+1] in ('.', ')'):\n",
    "#                     feat_arr[ind][i][2] = 1\n",
    "#                 elif sent_ls[i] in ('[', '(') and (sent_ls[i+1].isdigit() and len(sent_ls[i+1]) <=2) and sent_ls[i+2] in (']', ')'):\n",
    "#                     feat_arr[ind][i][3] = 1\n",
    "#                 elif sent_ls[i] in ('[', '(') and (sent_ls[i+1].isalpha() and len(sent_ls[i+1]) ==2) and sent_ls[i+2] in (']', ')'):\n",
    "#                     feat_arr[ind][i][4] = 1\n",
    "#     return feat_arr#.reshape(len(data_ls), maxlen, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%time\n",
    "# train_custom_arr = get_custom_feats(xtrain, maxlen)\n",
    "# valid_custom_arr = get_custom_feats(xvalid, maxlen)\n",
    "# test_custom_arr = get_custom_feats(sentences_test, maxlen)\n",
    "# eval1_custom_arr = get_custom_feats(sentences_eval1, maxlen)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Neural parscit features (temporary fix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%time\n",
    "# from feature.ParsCitLSTM import ParsCitLSTM\n",
    "# c = {\n",
    "#         \"model_file\": \"/nlp/parscit/parscit-29-latest.h5\",\n",
    "#         \"label_dict_file\": \"/nlp/parscit/labels.json\",\n",
    "#         \"tfhub_model_dir\": \"/nlp/parscit/resource/\"}\n",
    "# model = ParsCitLSTM(model_config=c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%time\n",
    "# a,b,c = model.predict(sentences_train[0][0:200])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%time\n",
    "# train_df = pd.DataFrame([])\n",
    "# valid_df = pd.DataFrame([])\n",
    "# test_df = pd.DataFrame([])\n",
    "#eval1_df = pd.DataFrame([])\n",
    "\n",
    "# train_df['text'] = np.array(xtrain, dtype='object')\n",
    "# valid_df['text'] = np.array(xvalid, dtype='object')\n",
    "# test_df['text'] = np.array(sentences_test, dtype='object')\n",
    "#eval1_df['text'] = np.array(sentences_eval1, dtype='object')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#%%time\n",
    "# train_parscit = model.get_parscit_blocks(train_df)\n",
    "# valid_parscit = model.get_parscit_blocks(valid_df)\n",
    "# test_parscit  = model.get_parscit_blocks(test_df)\n",
    "#eval1_parscit = model.get_parscit_blocks(eval1_df, chunk_size=40)\n",
    "\n",
    "\n",
    "# train_parscit = pd.read_pickle('/nlp/temp/train_parscit.pickle')\n",
    "# valid_parscit = pd.read_pickle('/nlp/temp/valid_parscit.pickle')\n",
    "# test_parscit = pd.read_pickle('/nlp/temp/test_parscit.pickle')\n",
    "# eval1_parscit = pd.read_pickle('/nlp/temp/eval1_parscit.pickle')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "# def pad_dummy_feats(df, maxlen):\n",
    "#     parscit = df.copy()\n",
    "#     parscit_feats = list(parscit['parscit_feat']) \n",
    "#     parscit_padded = []\n",
    "#     for ind in range(len(parscit_feats)):\n",
    "#         parscit_mask = np.zeros((maxlen, 14), dtype='int8')\n",
    "#         if len(parscit_feats[ind]) <= maxlen:\n",
    "#             parscit_mask[0:len(parscit_feats[ind]), :] = parscit_feats[ind][:]\n",
    "#         else:\n",
    "#             parscit_mask[:] = parscit_feats[ind][0:maxlen,:]\n",
    "#         parscit_padded.append(parscit_mask)\n",
    "#     parscit_arr = np.array([i.tolist() for i in parscit_padded])\n",
    "#     return parscit_arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%time\n",
    "# train_parscit_arr = pad_dummy_feats(train_parscit, maxlen)\n",
    "# valid_parscit_arr = pad_dummy_feats(valid_parscit, maxlen)\n",
    "# test_parscit_arr  = pad_dummy_feats(test_parscit, maxlen)\n",
    "#eval1_parscit_arr = pad_dummy_feats(eval1_parscit, maxlen)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%time\n",
    "# ## load saved files\n",
    "# train_parscit_arr = np.load('/nlp/temp/train_parscit_arr.npy')\n",
    "# valid_parscit_arr = np.load('/nlp/temp/valid_parscit_arr.npy')\n",
    "# test_parscit_arr  = np.load('/nlp/temp/test_parscit_arr.npy')\n",
    "# eval1_parscit_arr = np.load('/nlp/temp/eval1_parscit_arr.npy')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Keras with pre-trained fast text embedding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "#maxlen=100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 0 ns, sys: 91.4 ms, total: 91.4 ms\n",
      "Wall time: 91 ms\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "#wiki_model = FastText.load_fasttext_format('/nlp/cc.en.300.bin')\n",
    "#embedding_matrix = load_embedding_matrix(wiki_model, feat_config['max_vocab'], tokenizer.word_index, 300)\n",
    "embedding_matrix=np.load('/nlp/temp/embedding_matrix.npy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_config = {'embedding': {'use':True, 'matrix':embedding_matrix, 'trainable':True},## trainableTrue is significantly better \n",
    "                'lstm': {'use':True, 'num': 1, 'units':50, 'dropout':0.2},\n",
    "                'dense':{'use':False, 'num': 1, 'units':32, 'activation': 'relu', 'dropout':0.2},\n",
    "                'optimizer': Adam(\n",
    "        learning_rate=0.001,\n",
    "        beta_1=0.0,\n",
    "        beta_2=0.0,\n",
    "        epsilon=1e-05,\n",
    "        amsgrad=False,\n",
    "    ),\n",
    "                'output_activation' : 'sigmoid', \n",
    "                'batch_size': 32, ## lower the better\n",
    "                'aux_feats': {'use':True, 'dim':17, 'place':'before_lstm'}, \n",
    "                'timedistributed':{'use':False}, ## no difference at all True or False\n",
    "                'shuffle': True ## not much different than False\n",
    "                \n",
    "               }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "first line\n",
      "after imput\n",
      "after embedding\n",
      "before bidirectional\n",
      "after bidirectional\n",
      "Model: \"model\"\n",
      "__________________________________________________________________________________________________\n",
      "Layer (type)                    Output Shape         Param #     Connected to                     \n",
      "==================================================================================================\n",
      "input_1 (InputLayer)            [(None, 100)]        0                                            \n",
      "__________________________________________________________________________________________________\n",
      "embedding (Embedding)           (None, 100, 300)     30000000    input_1[0][0]                    \n",
      "__________________________________________________________________________________________________\n",
      "input_2 (InputLayer)            [(None, 100, 17)]    0                                            \n",
      "__________________________________________________________________________________________________\n",
      "concatenate (Concatenate)       (None, 100, 317)     0           embedding[0][0]                  \n",
      "                                                                 input_2[0][0]                    \n",
      "__________________________________________________________________________________________________\n",
      "bidirectional (Bidirectional)   (None, 100, 100)     147200      concatenate[0][0]                \n",
      "__________________________________________________________________________________________________\n",
      "dense (Dense)                   (None, 100, 1)       101         bidirectional[0][0]              \n",
      "==================================================================================================\n",
      "Total params: 30,147,301\n",
      "Trainable params: 30,147,301\n",
      "Non-trainable params: 0\n",
      "__________________________________________________________________________________________________\n",
      "None\n",
      "CPU times: user 1.64 s, sys: 223 ms, total: 1.86 s\n",
      "Wall time: 1.33 s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "model = get_model(feat_config, model_config, maxlen)\n",
    "print(model.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "both spacy and custom not parscit\n",
      "CPU times: user 7.75 s, sys: 5.37 s, total: 13.1 s\n",
      "Wall time: 13.1 s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "if feat_config['spacy_feats'] and not feat_config['parscit_feats'] and not feat_config['custom_feats']:\n",
    "    print('only spacy not parscit not custom')\n",
    "    train_aux = train_dict['spacy_num_feats']\n",
    "    valid_aux = valid_dict['spacy_num_feats']\n",
    "    test_aux  = test_dict['spacy_num_feats']\n",
    "    eval1_aux  = eval1_dict['spacy_num_feats']\n",
    "    \n",
    "elif feat_config['spacy_feats'] and  feat_config['parscit_feats'] and not feat_config['custom_feats']:\n",
    "    print(\"both spacy and parscit not custom\")\n",
    "    train_aux = np.concatenate((train_dict['spacy_num_feats'], train_parscit_arr), axis=-1)\n",
    "    valid_aux = np.concatenate((valid_dict['spacy_num_feats'], valid_parscit_arr), axis=-1)\n",
    "    test_aux  = np.concatenate((test_dict['spacy_num_feats'], test_parscit_arr), axis=-1)\n",
    "    eval1_aux  = np.concatenate((eval1_dict['spacy_num_feats'], eval1_parscit_arr), axis=-1)\n",
    "    \n",
    "elif feat_config['spacy_feats'] and  feat_config['custom_feats'] and not feat_config['parscit_feats']:\n",
    "    print(\"both spacy and custom not parscit\")\n",
    "    train_aux = np.concatenate((train_dict['spacy_num_feats'], train_dict['custom_feats']), axis=-1)\n",
    "    valid_aux = np.concatenate((valid_dict['spacy_num_feats'], valid_dict['custom_feats']), axis=-1)\n",
    "    test_aux  = np.concatenate((test_dict['spacy_num_feats'], test_dict['custom_feats']), axis=-1)\n",
    "    eval1_aux  = np.concatenate((eval1_dict['spacy_num_feats'], eval1_dict['custom_feats']), axis=-1)\n",
    "    \n",
    "elif feat_config['spacy_feats'] and  feat_config['parscit_feats'] and feat_config['custom_feats']:\n",
    "    print(\"all spacy and parscit and custom features\")\n",
    "    train_aux = np.concatenate((train_dict['spacy_num_feats'], train_parscit_arr, train_custom_arr), axis=-1)\n",
    "    valid_aux = np.concatenate((valid_dict['spacy_num_feats'], valid_parscit_arr, valid_custom_arr), axis=-1)\n",
    "    test_aux  = np.concatenate((test_dict['spacy_num_feats'], test_parscit_arr, test_custom_arr), axis=-1)\n",
    "    eval1_aux  = np.concatenate((eval1_dict['spacy_num_feats'], eval1_parscit_arr, eval1_custom_arr), axis=-1)\n",
    "else:\n",
    "    print('only lstm features')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "early_stop= EarlyStopping(monitor='val_loss',patience=3,verbose=0,mode='min',restore_best_weights=False, min_delta=0.0001)\n",
    "\n",
    "## multi input\n",
    "if model_config['aux_feats']['use']:\n",
    "    print(\"multi input model\")\n",
    "    history = model.fit([train_dict['lstm_feats'], train_aux], train_dict['labels'], verbose=1, epochs=100, batch_size= model_config['batch_size'], \\\n",
    "                    validation_data=([valid_dict['lstm_feats'], valid_aux], valid_dict['labels']), callbacks=[early_stop], shuffle=model_config['shuffle'])\n",
    "else:\n",
    "    print(\"single input model\")\n",
    "    history = model.fit(train_dict['lstm_feats'], train_dict['labels'], verbose=1, epochs=100, batch_size= model_config['batch_size'], \\\n",
    "                        validation_data=(valid_dict['lstm_feats'], valid_dict['labels']), callbacks=[early_stop], shuffle=model_config['shuffle'])\n",
    "\n",
    "plot_output(history)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done\n"
     ]
    }
   ],
   "source": [
    "print('done')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "test\n"
     ]
    }
   ],
   "source": [
    "print('test')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#sentences_test[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Analysis on validation data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "#keras_model = mlflow.keras.load_model(\"s3://caps-s3-mlflow/artifacts/4/59e06c5ff45944f5bf56745b2e19cb05/artifacts/models\")\n",
    "if model_config['aux_feats']['use']:\n",
    "    valid_probs = model.predict([valid_dict['lstm_feats'], valid_aux])\n",
    "else:\n",
    "    valid_probs = model.predict(valid_dict['lstm_feats'])\n",
    "\n",
    "valid_probs = valid_probs.reshape(valid_probs.shape[0], valid_probs.shape[1])\n",
    "valid_preds = np.where(valid_probs > 0.5, 1, 0)\n",
    "\n",
    "valid_true_ls = [i[0:len(j)].tolist() for i,j in zip(valid_dict['labels'], valid_dict['tags_window'])]\n",
    "valid_pred_ls = [i[0:len(j)].tolist() for i,j in zip(valid_preds, valid_dict['tags_window'])]\n",
    "result_valid = evaluate(valid_true_ls, valid_pred_ls)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'count': 155569,\n",
       " 'count_citations': 197153,\n",
       " 'mean_ser': 0.07548007585466847,\n",
       " 'mean_jer': 0.07745027547108939,\n",
       " 'mean_acc': 0.9997166608829646,\n",
       " 'num_mistakes_seq': 2097,\n",
       " 'num_mistakes_all': 2199,\n",
       " 'mistakes_per_seq': 1.0486409155937053,\n",
       " 'perc_mistakes_seq': 1.3479549267527593,\n",
       " 'perc_mistake_per_citation': 1.1153773972498517}"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_valid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'count': 18654,\n",
       " 'count_citations': 45668,\n",
       " 'mean_ser': 0.035972241367866975,\n",
       " 'mean_jer': 0.03794619988733852,\n",
       " 'mean_acc': 0.9998070034769974,\n",
       " 'num_mistakes_seq': 274,\n",
       " 'num_mistakes_all': 299,\n",
       " 'mistakes_per_seq': 1.0912408759124088,\n",
       " 'perc_mistakes_seq': 1.4688538651227618,\n",
       " 'perc_mistake_per_citation': 0.6547254094770956}"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_valid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#keras_model = mlflow.keras.load_model(\"s3://caps-s3-mlflow/artifacts/4/977e2cc7b36040f79dde3d6303d26952/artifacts/models\")\n",
    "count_valid = 0\n",
    "#valid_df = pd.DataFrame([])\n",
    "valid_ls = []\n",
    "for ind in range(len(valid_preds)):\n",
    "    temp_ls = []\n",
    "    pred = valid_preds[ind]\n",
    "    true = valid_dict['labels'][ind]\n",
    "    if (true == pred).all():\n",
    "        pass\n",
    "    else:\n",
    "        count_valid += 1\n",
    "        fp_ind = np.where((pred == 1) & (true == 0))[0]\n",
    "        fn_ind = np.where((pred == 0) & (true == 1))[0]\n",
    "        if len(fp_ind) > 0:\n",
    "            for x in fp_ind:\n",
    "                valid_ls.append([ind, 'FP'] + np.array(xvalid[ind].split(\" \"))[max(0, x-3):x+4].tolist())\n",
    "                #print(ind, 'FP', np.array(xvalid[ind].split(\" \"))[max(0, x-3):x+4])\n",
    "        if len(fn_ind) > 0:\n",
    "            #print(ind, 'False negatives:', fn_ind)\n",
    "            for x in fn_ind:\n",
    "                valid_ls.append([ind, 'FN'] + np.array(xvalid[ind].split(\" \"))[max(0, x-3):x+4].tolist())\n",
    "                #print(ind, 'FN', np.array(xvalid[ind].split(\" \"))[max(0, x-3):x+4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "valid_mistakes_df = pd.DataFrame(valid_ls, columns=['index', 'error_type', 'x-3', 'x-2', 'x-1', 'x', 'x+1', 'x+2', 'x+3'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Analysis of test result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "##### CHANGE THE KERAS MODEL LOADING PART ******************************************\n",
    "model = mlflow.keras.load_model(\"s3://caps-s3-mlflow/artifacts/4/aea7f13e822d408e827726c2b972b93c/artifacts/models/\")\n",
    "if model_config['aux_feats']['use']:\n",
    "    test_probs = model.predict([test_dict['lstm_feats'], test_aux])\n",
    "else:\n",
    "    test_probs = model.predict(test_dict['lstm_feats'])\n",
    "test_probs = test_probs.reshape(test_probs.shape[0], test_probs.shape[1])\n",
    "test_preds = np.where(test_probs > 0.5, 1, 0)\n",
    "# test_true_ls = [i[0:len(j.split(\" \"))].tolist() for i,j in zip(test_dict['labels'], sent_tags_test_win)]\n",
    "# test_pred_ls = [i[0:len(j.split(\" \"))].tolist() for i,j in zip(test_preds, sent_tags_test_win)]\n",
    "# result_test = evaluate(test_true_ls, test_pred_ls)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 2.27 s, sys: 34 ms, total: 2.31 s\n",
      "Wall time: 2.25 s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "take_mean = True\n",
    "test_probs_win = []\n",
    "for ind in range(len(test_probs)):\n",
    "    #print(ind)\n",
    "    test_probs_win.append(test_probs[ind][0:len(test_dict['sentences_window'][ind])])\n",
    "\n",
    "test_probs_merged = []\n",
    "window = 100\n",
    "step = 90\n",
    "gind = 0\n",
    "for bigind in range(len(sentences_test)):\n",
    "    temp = []\n",
    "    seqlen =  len(sentences_test[bigind].split(\" \"))\n",
    "    iter = 0\n",
    "    #print(seqlen)\n",
    "    #print(bigind)\n",
    "    for smallind in range(gind, len(test_probs_win)):\n",
    "        if len(temp) < seqlen:\n",
    "            if iter ==0:\n",
    "                temp+=test_probs_win[gind].tolist()\n",
    "                iter +=1\n",
    "                gind += 1\n",
    "            else:\n",
    "                if take_mean:\n",
    "                    temp[-(window-step):] = np.mean([np.array(temp[-(window-step):]), test_probs_win[gind][0:(window-step)]], axis=0).tolist()\n",
    "                    temp += test_probs_win[gind][(window-step):].tolist()\n",
    "                    gind += 1\n",
    "                else:\n",
    "                    temp+=test_probs_win[gind][(window-step):].tolist()\n",
    "                    gind += 1\n",
    "        else:\n",
    "            break\n",
    "    test_probs_merged.append(temp)\n",
    "\n",
    "tag2index = {'B-CIT':1, 'I-CIT':0}\n",
    "test_preds_merged = [[int(t>0.5) for t in test_probs_merged[ind]] for ind in range(len(test_probs_merged))]\n",
    "test_true_merged = [[tag2index[t] for t in sent_tags_test[ind].split(\" \")] for ind in range(len(sent_tags_test))]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 6 µs, sys: 1e+03 ns, total: 7 µs\n",
      "Wall time: 12.4 µs\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "#result_test = evaluate(test_true_merged, test_preds_merged)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'count': 1953,\n",
       " 'count_citations': 46092,\n",
       " 'mean_ser': 0.0007581633612078364,\n",
       " 'mean_jer': 0.002062706327176513,\n",
       " 'mean_acc': 0.999909105171757,\n",
       " 'num_mistakes_seq': 93,\n",
       " 'num_mistakes_all': 121,\n",
       " 'mistakes_per_seq': 1.3010752688172043,\n",
       " 'perc_mistakes_seq': 4.761904761904762,\n",
       " 'perc_mistake_per_citation': 0.26251844137811337}"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'count': 1953,\n",
       " 'count_citations': 46092,\n",
       " 'mean_ser': 0.0007581633612078364,\n",
       " 'mean_jer': 0.002062706327176513,\n",
       " 'mean_acc': 0.999909105171757,\n",
       " 'num_mistakes_seq': 93,\n",
       " 'num_mistakes_all': 121,\n",
       " 'mistakes_per_seq': 1.3010752688172043,\n",
       " 'perc_mistakes_seq': 4.761904761904762,\n",
       " 'perc_mistake_per_citation': 0.26251844137811337}"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2020/09/06 16:19:16 WARNING mlflow.models.model: Logging model metadata to the tracking server has failed, possibly due older server version. The model artifacts have been logged successfully under s3://caps-s3-mlflow/artifacts/4/7d154794377c4317b2c13fe3fb9aaaeb/artifacts. In addition to exporting model artifacts, MLflow clients 1.7.0 and above attempt to record model metadata to the  tracking store. If logging to a mlflow server via REST, consider  upgrading the server version to MLflow 1.7.0 or above.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 5.48 s, sys: 2.1 s, total: 7.58 s\n",
      "Wall time: 12.1 s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "tags = {'dataset':'cs_mixed_20k', 'data_split':'test', 'version':2} \n",
    "del model_config['embedding']['matrix'] \n",
    "opt = model_config['optimizer']\n",
    "model_config['optimizer'] = str(opt.get_config())\n",
    "log_mlflow_results(model, result_test, feat_config, model_config, tags)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "count_test = 0\n",
    "test_ls = []\n",
    "for ind in range(len(test_preds_merged)):\n",
    "    temp_ls = []\n",
    "    pred = np.array(test_preds_merged[ind])\n",
    "    true = np.array(test_true_merged[ind])\n",
    "    if (true == pred).all():\n",
    "        pass\n",
    "    else:\n",
    "        count_test += 1\n",
    "        fp_ind = np.where((pred == 1) & (true == 0))[0]\n",
    "        fn_ind = np.where((pred == 0) & (true == 1))[0]\n",
    "        if len(fp_ind) > 0:\n",
    "            #print(ind, 'fp', fp_ind)\n",
    "            for x in fp_ind:\n",
    "                ls = ['None']*7\n",
    "                if max(0, x-3)==0:\n",
    "                    ls[3-x:] = np.array(sentences_test[ind].split(\" \"))[max(0, x-3):x+4].tolist()\n",
    "                else:\n",
    "                    ls = np.array(sentences_test[ind].split(\" \"))[max(0, x-3):x+4].tolist()\n",
    "                test_ls.append([ind, x, 'FP'] + ls)\n",
    "                #print(ind, 'FP', np.array(sentences_test[ind].split(\" \"))[max(0, x-3):x+4])\n",
    "        if len(fn_ind) > 0:\n",
    "            #print(ind, 'False negatives:', fn_ind)\n",
    "            #print(ind, 'fn', fn_ind)\n",
    "            for x in fn_ind:\n",
    "                ls = ['None']*7\n",
    "                if max(0, x-3)==0:\n",
    "                    ls[3-x:] = np.array(sentences_test[ind].split(\" \"))[max(0, x-3):x+4].tolist()\n",
    "                else:\n",
    "                    ls = np.array(sentences_test[ind].split(\" \"))[max(0, x-3):x+4].tolist()\n",
    "                test_ls.append([ind, x, 'FN'] + ls)\n",
    "                #print(ind, 'FN', np.array(sentences_test[ind].split(\" \"))[max(0, x-3):x+4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_mistakes_df = pd.DataFrame(test_ls, columns=['seq_index', 'token_index', 'error_type', 'x-3', 'x-2', 'x-1', 'x', 'x+1', 'x+2', 'x+3'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_mistakes_df = test_mistakes_df.replace('\\n', 'newline')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#test_mistakes_df[test_mistakes_df.error_type == 'FN']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "FN    86\n",
       "FP    35\n",
       "Name: error_type, dtype: int64"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_mistakes_df.error_type.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_mistakes_df.to_csv('../test_mistakes_windownew.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#test_mistakes_df[test_mistakes_df.error_type == 'FP']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Analysis of eval dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "#keras_model = mlflow.keras.load_model(\"s3://caps-s3-mlflow/artifacts/4/1d953e89279b49fdb63cd7b3cb6c8b0b/artifacts/models\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "auxiliary features ...\n"
     ]
    }
   ],
   "source": [
    "#model = mlflow.keras.load_model(\"s3://caps-s3-mlflow/artifacts/4/aea7f13e822d408e827726c2b972b93c/artifacts/models/\")\n",
    "if model_config['aux_feats']['use']:\n",
    "    print('auxiliary features ...')\n",
    "    eval1_probs = model.predict([eval1_dict['lstm_feats'], eval1_aux])\n",
    "else:\n",
    "    print('only lstm features ...')\n",
    "    eval1_probs = model.predict(eval1_dict['lstm_feats'])\n",
    "eval1_probs = eval1_probs.reshape(eval1_probs.shape[0], eval1_probs.shape[1])\n",
    "eval1_preds = np.where(eval1_probs > 0.5, 1, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 1.83 s, sys: 104 ms, total: 1.94 s\n",
      "Wall time: 1.93 s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "take_mean = True\n",
    "eval1_probs_win = []\n",
    "for ind in range(len(eval1_probs)):\n",
    "    #print(ind)\n",
    "    eval1_probs_win.append(eval1_probs[ind][0:len(eval1_dict['sentences_window'][ind])])\n",
    "\n",
    "eval1_probs_merged = []\n",
    "window = 100\n",
    "step = 90\n",
    "gind = 0\n",
    "for bigind in range(len(sentences_eval1)):\n",
    "    temp = []\n",
    "    seqlen =  len(sentences_eval1[bigind].split(\" \"))\n",
    "    iter = 0\n",
    "    #print(seqlen)\n",
    "    #print(bigind)\n",
    "    for smallind in range(gind, len(eval1_probs_win)):\n",
    "        if len(temp) < seqlen:\n",
    "            if iter ==0:\n",
    "                temp+=eval1_probs_win[gind].tolist()\n",
    "                iter +=1\n",
    "                gind += 1\n",
    "            else:\n",
    "                if take_mean:\n",
    "                    temp[-(window-step):] = np.mean([np.array(temp[-(window-step):]), eval1_probs_win[gind][0:(window-step)]], axis=0).tolist()\n",
    "                    temp += eval1_probs_win[gind][(window-step):].tolist()\n",
    "                    gind += 1\n",
    "                else:\n",
    "                    temp+=eval1_probs_win[gind][(window-step):].tolist()\n",
    "                    gind += 1\n",
    "        else:\n",
    "            break\n",
    "    eval1_probs_merged.append(temp)\n",
    "\n",
    "tag2index = {'B-CIT':1, 'I-CIT':0}\n",
    "eval1_preds_merged = [[int(t>0.5) for t in eval1_probs_merged[ind]] for ind in range(len(eval1_probs_merged))]\n",
    "eval1_true_merged = [[tag2index[t] for t in sent_tags_eval1[ind].split(\" \")] for ind in range(len(sent_tags_eval1))]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "result_eval1 = evaluate(eval1_true_merged, eval1_preds_merged)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'count': 1328,\n",
       " 'count_citations': 56850,\n",
       " 'mean_ser': 1.2982966348151225e-05,\n",
       " 'mean_jer': 0.00041063361934943446,\n",
       " 'mean_acc': 0.9999894054432367,\n",
       " 'num_mistakes_seq': 18,\n",
       " 'num_mistakes_all': 26,\n",
       " 'mistakes_per_seq': 1.4444444444444444,\n",
       " 'perc_mistakes_seq': 1.355421686746988,\n",
       " 'perc_mistake_per_citation': 0.04573438874230431}"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_eval1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'count': 1328,\n",
       " 'count_citations': 56850,\n",
       " 'mean_ser': 0.0007642510339866931,\n",
       " 'mean_jer': 0.0075459564576609486,\n",
       " 'mean_acc': 0.9998296369743739,\n",
       " 'num_mistakes_seq': 353,\n",
       " 'num_mistakes_all': 411,\n",
       " 'mistakes_per_seq': 1.1643059490084986,\n",
       " 'perc_mistakes_seq': 26.58132530120482,\n",
       " 'perc_mistake_per_citation': 0.7229551451187335}"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_eval1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'count': 1328,\n",
       " 'count_citations': 56850,\n",
       " 'mean_ser': 0.0007776320593378298,\n",
       " 'mean_jer': 0.008467544104861686,\n",
       " 'mean_acc': 0.9998095060436722,\n",
       " 'num_mistakes_seq': 422,\n",
       " 'num_mistakes_all': 463,\n",
       " 'mistakes_per_seq': 1.0971563981042654,\n",
       " 'perc_mistakes_seq': 31.77710843373494,\n",
       " 'perc_mistake_per_citation': 0.8144239226033422}"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_eval1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'count': 1328,\n",
       " 'count_citations': 56850,\n",
       " 'mean_ser': 0.0007637693631669535,\n",
       " 'mean_jer': 0.006231187644864529,\n",
       " 'mean_acc': 0.9998530866974515,\n",
       " 'num_mistakes_seq': 300,\n",
       " 'num_mistakes_all': 344,\n",
       " 'mistakes_per_seq': 1.1466666666666667,\n",
       " 'perc_mistakes_seq': 22.59036144578313,\n",
       " 'mistake_per_citation': 0.6051011433597185}"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_eval1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualize errors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "count_eval1 = 0\n",
    "eval1_ls = []\n",
    "for ind in range(len(eval1_preds_merged)):\n",
    "    temp_ls = []\n",
    "    pred = np.array(eval1_preds_merged[ind])\n",
    "    true = np.array(eval1_true_merged[ind])\n",
    "    if (true == pred).all():\n",
    "        pass\n",
    "    else:\n",
    "        count_eval1 += 1\n",
    "        fp_ind = np.where((pred == 1) & (true == 0))[0]\n",
    "        fn_ind = np.where((pred == 0) & (true == 1))[0]\n",
    "        if len(fp_ind) > 0:\n",
    "            #print(ind, 'fp', fp_ind)\n",
    "            for x in fp_ind:\n",
    "                ls = ['None']*7\n",
    "                if max(0, x-3)==0:\n",
    "                    ls[3-x:] = np.array(sentences_eval1[ind].split(\" \"))[max(0, x-3):x+4].tolist()\n",
    "                else:\n",
    "                    ls = np.array(sentences_eval1[ind].split(\" \"))[max(0, x-3):x+4].tolist()\n",
    "                eval1_ls.append([ind, x, 'FP'] + ls)\n",
    "                #print(ind, 'FP', np.array(sentences_eval1[ind].split(\" \"))[max(0, x-3):x+4])\n",
    "        if len(fn_ind) > 0:\n",
    "            #print(ind, 'False negatives:', fn_ind)\n",
    "            #print(ind, 'fn', fn_ind)\n",
    "            for x in fn_ind:\n",
    "                ls = ['None']*7\n",
    "                if max(0, x-3)==0:\n",
    "                    ls[3-x:] = np.array(sentences_eval1[ind].split(\" \"))[max(0, x-3):x+4].tolist()\n",
    "                else:\n",
    "                    ls = np.array(sentences_eval1[ind].split(\" \"))[max(0, x-3):x+4].tolist()\n",
    "                eval1_ls.append([ind, x, 'FN'] + ls)\n",
    "                #print(ind, 'FN', np.array(sentences_eval1[ind].split(\" \"))[max(0, x-3):x+4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "#eval1_ls"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "eval_mistakes_df = pd.DataFrame(eval1_ls, columns=['seq_index', 'token_index', 'error_type', 'x-3', 'x-2', 'x-1', 'x', 'x+1', 'x+2', 'x+3'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "eval_mistakes_df = eval_mistakes_df.replace('\\n', 'newline')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>seq_index</th>\n",
       "      <th>token_index</th>\n",
       "      <th>error_type</th>\n",
       "      <th>x-3</th>\n",
       "      <th>x-2</th>\n",
       "      <th>x-1</th>\n",
       "      <th>x</th>\n",
       "      <th>x+1</th>\n",
       "      <th>x+2</th>\n",
       "      <th>x+3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>171</td>\n",
       "      <td>420</td>\n",
       "      <td>FP</td>\n",
       "      <td>,</td>\n",
       "      <td>Ekpa</td>\n",
       "      <td>newline</td>\n",
       "      <td>,</td>\n",
       "      <td>Cooper</td>\n",
       "      <td>TK</td>\n",
       "      <td>,</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>210</td>\n",
       "      <td>772</td>\n",
       "      <td>FN</td>\n",
       "      <td>)</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>18</td>\n",
       "      <td>Hunger</td>\n",
       "      <td>-</td>\n",
       "      <td>Craig</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>371</td>\n",
       "      <td>1286</td>\n",
       "      <td>FN</td>\n",
       "      <td>34</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>.</td>\n",
       "      <td>A</td>\n",
       "      <td>Consensus</td>\n",
       "      <td>on</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>412</td>\n",
       "      <td>251</td>\n",
       "      <td>FN</td>\n",
       "      <td>76</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>[</td>\n",
       "      <td>8]M.</td>\n",
       "      <td>Preene</td>\n",
       "      <td>,</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>412</td>\n",
       "      <td>561</td>\n",
       "      <td>FN</td>\n",
       "      <td>4260</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>[</td>\n",
       "      <td>17]V.Hamm</td>\n",
       "      <td>,</td>\n",
       "      <td>B.B.Sabet</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>706</td>\n",
       "      <td>828</td>\n",
       "      <td>FN</td>\n",
       "      <td>after</td>\n",
       "      <td>subacute</td>\n",
       "      <td>newline</td>\n",
       "      <td>Monica</td>\n",
       "      <td>,</td>\n",
       "      <td>F.Z.</td>\n",
       "      <td>,</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>712</td>\n",
       "      <td>2833</td>\n",
       "      <td>FN</td>\n",
       "      <td>346−352</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>66</td>\n",
       "      <td>.</td>\n",
       "      <td>http://www.ozm.cz/en/sensitivity-tests/small-s...</td>\n",
       "      <td>newline</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>776</td>\n",
       "      <td>282</td>\n",
       "      <td>FN</td>\n",
       "      <td>257</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>[</td>\n",
       "      <td>10]J.W.Xi</td>\n",
       "      <td>,</td>\n",
       "      <td>Control</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>776</td>\n",
       "      <td>416</td>\n",
       "      <td>FN</td>\n",
       "      <td>129</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>[</td>\n",
       "      <td>14]X.X.Zhao</td>\n",
       "      <td>,</td>\n",
       "      <td>L.J.Li</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>841</td>\n",
       "      <td>1458</td>\n",
       "      <td>FN</td>\n",
       "      <td>remodeling</td>\n",
       "      <td>in</td>\n",
       "      <td>newline</td>\n",
       "      <td>Saccharomyces</td>\n",
       "      <td>cerevisiae</td>\n",
       "      <td>mutants</td>\n",
       "      <td>affected</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>841</td>\n",
       "      <td>1468</td>\n",
       "      <td>FN</td>\n",
       "      <td>stress</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>Appl</td>\n",
       "      <td>Environ</td>\n",
       "      <td>Microbiol</td>\n",
       "      <td>.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>841</td>\n",
       "      <td>1495</td>\n",
       "      <td>FN</td>\n",
       "      <td>Laboratory</td>\n",
       "      <td>Manual</td>\n",
       "      <td>newline</td>\n",
       "      <td>(</td>\n",
       "      <td>Dieffenbach</td>\n",
       "      <td>,</td>\n",
       "      <td>C.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>841</td>\n",
       "      <td>1517</td>\n",
       "      <td>FN</td>\n",
       "      <td>Cold</td>\n",
       "      <td>Spring</td>\n",
       "      <td>newline</td>\n",
       "      <td>Harbor</td>\n",
       "      <td>Laboratory</td>\n",
       "      <td>Press</td>\n",
       "      <td>,</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>841</td>\n",
       "      <td>1548</td>\n",
       "      <td>FN</td>\n",
       "      <td>Genetics</td>\n",
       "      <td>,</td>\n",
       "      <td>newline</td>\n",
       "      <td>Cold</td>\n",
       "      <td>Spring</td>\n",
       "      <td>Harbor</td>\n",
       "      <td>Laboratory</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>841</td>\n",
       "      <td>1706</td>\n",
       "      <td>FN</td>\n",
       "      <td>enzymes</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>Mol</td>\n",
       "      <td>Biol</td>\n",
       "      <td>Cell</td>\n",
       "      <td>.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>931</td>\n",
       "      <td>2055</td>\n",
       "      <td>FN</td>\n",
       "      <td>-</td>\n",
       "      <td>parametric</td>\n",
       "      <td>newline</td>\n",
       "      <td>models</td>\n",
       "      <td>of</td>\n",
       "      <td>production</td>\n",
       "      <td>processes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>954</td>\n",
       "      <td>9223</td>\n",
       "      <td>FN</td>\n",
       "      <td>Experimental</td>\n",
       "      <td>Physiology</td>\n",
       "      <td>newline</td>\n",
       "      <td>79</td>\n",
       "      <td>.</td>\n",
       "      <td>Powers</td>\n",
       "      <td>,</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>954</td>\n",
       "      <td>10553</td>\n",
       "      <td>FN</td>\n",
       "      <td>-</td>\n",
       "      <td>1521</td>\n",
       "      <td>newline</td>\n",
       "      <td>103</td>\n",
       "      <td>.</td>\n",
       "      <td>Krause</td>\n",
       "      <td>,</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>992</td>\n",
       "      <td>0</td>\n",
       "      <td>FN</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "      <td>.</td>\n",
       "      <td>Aziz</td>\n",
       "      <td>,</td>\n",
       "      <td>R.K.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>1002</td>\n",
       "      <td>1800</td>\n",
       "      <td>FN</td>\n",
       "      <td>15</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>The</td>\n",
       "      <td>Chinese</td>\n",
       "      <td>Environmental</td>\n",
       "      <td>Monitoring</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>1007</td>\n",
       "      <td>111</td>\n",
       "      <td>FN</td>\n",
       "      <td>Res</td>\n",
       "      <td>1999;128:112–18</td>\n",
       "      <td>newline</td>\n",
       "      <td>5</td>\n",
       "      <td>Lopez</td>\n",
       "      <td>-</td>\n",
       "      <td>Gonzalez</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>1030</td>\n",
       "      <td>4606</td>\n",
       "      <td>FN</td>\n",
       "      <td>Article</td>\n",
       "      <td>35</td>\n",
       "      <td>newline</td>\n",
       "      <td>112</td>\n",
       "      <td>.</td>\n",
       "      <td>Berman</td>\n",
       "      <td>DE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>1047</td>\n",
       "      <td>418</td>\n",
       "      <td>FN</td>\n",
       "      <td>)</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>17.D.</td>\n",
       "      <td>S.</td>\n",
       "      <td>McLachlan</td>\n",
       "      <td>,</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>1105</td>\n",
       "      <td>40</td>\n",
       "      <td>FN</td>\n",
       "      <td>2011;21:351–5</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>2</td>\n",
       "      <td>Kushner</td>\n",
       "      <td>RF</td>\n",
       "      <td>.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>1198</td>\n",
       "      <td>1890</td>\n",
       "      <td>FN</td>\n",
       "      <td>4</td>\n",
       "      <td>.</td>\n",
       "      <td>newline</td>\n",
       "      <td>Toth</td>\n",
       "      <td>,</td>\n",
       "      <td>A.</td>\n",
       "      <td>,</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>1293</td>\n",
       "      <td>37</td>\n",
       "      <td>FN</td>\n",
       "      <td>,</td>\n",
       "      <td>N.J.</td>\n",
       "      <td>newline</td>\n",
       "      <td>Barr</td>\n",
       "      <td>,</td>\n",
       "      <td>C.S.</td>\n",
       "      <td>,</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    seq_index  token_index error_type            x-3              x-2  \\\n",
       "0         171          420         FP              ,             Ekpa   \n",
       "1         210          772         FN              )                .   \n",
       "2         371         1286         FN             34                .   \n",
       "3         412          251         FN             76                .   \n",
       "4         412          561         FN           4260                .   \n",
       "5         706          828         FN          after         subacute   \n",
       "6         712         2833         FN        346−352                .   \n",
       "7         776          282         FN            257                .   \n",
       "8         776          416         FN            129                .   \n",
       "9         841         1458         FN     remodeling               in   \n",
       "10        841         1468         FN         stress                .   \n",
       "11        841         1495         FN     Laboratory           Manual   \n",
       "12        841         1517         FN           Cold           Spring   \n",
       "13        841         1548         FN       Genetics                ,   \n",
       "14        841         1706         FN        enzymes                .   \n",
       "15        931         2055         FN              -       parametric   \n",
       "16        954         9223         FN   Experimental       Physiology   \n",
       "17        954        10553         FN              -             1521   \n",
       "18        992            0         FN           None             None   \n",
       "19       1002         1800         FN             15                .   \n",
       "20       1007          111         FN            Res  1999;128:112–18   \n",
       "21       1030         4606         FN        Article               35   \n",
       "22       1047          418         FN              )                .   \n",
       "23       1105           40         FN  2011;21:351–5                .   \n",
       "24       1198         1890         FN              4                .   \n",
       "25       1293           37         FN              ,             N.J.   \n",
       "\n",
       "        x-1              x          x+1  \\\n",
       "0   newline              ,       Cooper   \n",
       "1   newline             18       Hunger   \n",
       "2   newline              .            A   \n",
       "3   newline              [         8]M.   \n",
       "4   newline              [    17]V.Hamm   \n",
       "5   newline         Monica            ,   \n",
       "6   newline             66            .   \n",
       "7   newline              [    10]J.W.Xi   \n",
       "8   newline              [  14]X.X.Zhao   \n",
       "9   newline  Saccharomyces   cerevisiae   \n",
       "10  newline           Appl      Environ   \n",
       "11  newline              (  Dieffenbach   \n",
       "12  newline         Harbor   Laboratory   \n",
       "13  newline           Cold       Spring   \n",
       "14  newline            Mol         Biol   \n",
       "15  newline         models           of   \n",
       "16  newline             79            .   \n",
       "17  newline            103            .   \n",
       "18     None              .         Aziz   \n",
       "19  newline            The      Chinese   \n",
       "20  newline              5        Lopez   \n",
       "21  newline            112            .   \n",
       "22  newline          17.D.           S.   \n",
       "23  newline              2      Kushner   \n",
       "24  newline           Toth            ,   \n",
       "25  newline           Barr            ,   \n",
       "\n",
       "                                                  x+2         x+3  \n",
       "0                                                  TK           ,  \n",
       "1                                                   -       Craig  \n",
       "2                                           Consensus          on  \n",
       "3                                              Preene           ,  \n",
       "4                                                   ,   B.B.Sabet  \n",
       "5                                                F.Z.           ,  \n",
       "6   http://www.ozm.cz/en/sensitivity-tests/small-s...     newline  \n",
       "7                                                   ,     Control  \n",
       "8                                                   ,      L.J.Li  \n",
       "9                                             mutants    affected  \n",
       "10                                          Microbiol           .  \n",
       "11                                                  ,          C.  \n",
       "12                                              Press           ,  \n",
       "13                                             Harbor  Laboratory  \n",
       "14                                               Cell           .  \n",
       "15                                         production   processes  \n",
       "16                                             Powers           ,  \n",
       "17                                             Krause           ,  \n",
       "18                                                  ,        R.K.  \n",
       "19                                      Environmental  Monitoring  \n",
       "20                                                  -    Gonzalez  \n",
       "21                                             Berman          DE  \n",
       "22                                          McLachlan           ,  \n",
       "23                                                 RF           .  \n",
       "24                                                 A.           ,  \n",
       "25                                               C.S.           ,  "
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "eval_mistakes_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "eval_mistakes_df.to_csv('../eval_mistakes_windownew.csv', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Save eval results to file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ind = 0\n",
    "eval_path_dict = {}\n",
    "for fpath in os.listdir(eval_without_intra_newline_path):\n",
    "    if fpath not in ['data-gen-config.json', 'data_generation_stats.csv'] and \".csv\" in fpath:\n",
    "        eval_path_dict[ind] = fpath\n",
    "        ind += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "eval_folder_path = '/nlp/eval_data_retok_predictions_f3a9cc61e30f460499b95a4c2b7957ae'\n",
    "if not os.path.exists(eval_folder_path):\n",
    "    os.makedirs(eval_folder_path)\n",
    "for ind in range(len(eval_preds1)):\n",
    "    #print(ind)\n",
    "    df = pd.DataFrame([], columns=['x', 'y'])\n",
    "    seq_len = min(len(sentences_eval1[ind].split(\" \")), maxlen)\n",
    "    df['x'] = sentences_eval1[ind].split(\" \")[0:seq_len]\n",
    "    df['y'] = eval_preds1[ind][0:seq_len]\n",
    "    df.to_csv(os.path.join(eval_folder_path, eval_path_dict[ind][0:-4] + '_pred.csv'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Log Eval results to MLFlow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2020/09/06 16:20:20 WARNING mlflow.models.model: Logging model metadata to the tracking server has failed, possibly due older server version. The model artifacts have been logged successfully under s3://caps-s3-mlflow/artifacts/4/aea7f13e822d408e827726c2b972b93c/artifacts. In addition to exporting model artifacts, MLflow clients 1.7.0 and above attempt to record model metadata to the  tracking store. If logging to a mlflow server via REST, consider  upgrading the server version to MLflow 1.7.0 or above.\n"
     ]
    }
   ],
   "source": [
    "tags = {'dataset':'cs_mixed_20k', 'data_split':'eval_with_intra_newline_path','version':2}\n",
    "#del model_config['embedding']['matrix'] \n",
    "#opt = model_config['optimizer']\n",
    "#model_config['optimizer'] = str(opt.get_config())\n",
    "log_mlflow_results(model, result_eval1, feat_config, model_config, tags)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Saving features to file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%time\n",
    "# np.save('/nlp/temp/train_dict_labels.npy', train_dict['labels'])\n",
    "# np.save('/nlp/temp/train_dict_lstm_feats.npy', train_dict['lstm_feats'])\n",
    "# np.save('/nlp/temp/train_dict_spacy_feats.npy', train_dict['spacy_num_feats'])\n",
    "\n",
    "# np.save('/nlp/temp/valid_dict_labels.npy', valid_dict['labels'])\n",
    "# np.save('/nlp/temp/valid_dict_lstm_feats.npy', valid_dict['lstm_feats'])\n",
    "# np.save('/nlp/temp/valid_dict_spacy_feats.npy', valid_dict['spacy_num_feats'])\n",
    "\n",
    "# np.save('/nlp/temp/test_dict_labels.npy', test_dict['labels'])\n",
    "# np.save('/nlp/temp/test_dict_lstm_feats.npy', test_dict['lstm_feats'])\n",
    "# np.save('/nlp/temp/test_dict_spacy_feats.npy', test_dict['spacy_num_feats'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# np.save('/nlp/temp/eval1_dict_labels.npy', eval1_dict['labels'])\n",
    "# np.save('/nlp/temp/eval1_dict_lstm_feats.npy', eval1_dict['lstm_feats'])\n",
    "# np.save('/nlp/temp/eval1_dict_spacy_feats.npy', eval1_dict['spacy_num_feats'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy as sp\n",
    "import os\n",
    "os.system('python3 -m spacy download en_core_web_sm')\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from spacy.matcher import PhraseMatcher\n",
    "import dask.dataframe as dd\n",
    "from dask.multiprocessing import get\n",
    "from spacy.tokenizer import Tokenizer\n",
    "from feature_window.SpacyFeaturizer_window import SpacyFeaturizer_window, pad_spacy_feats\n",
    "from util.Utils import pad_sequences, sliding_window_list, flatten_3d_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1/1 [00:00<00:00, 2118.34it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 4.43 ms, sys: 14 µs, total: 4.45 ms\n",
      "Wall time: 3.37 ms\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "sp = SpacyFeaturizer_window()\n",
    "testls = [sentences_train[0]]\n",
    "\n",
    "data_dict={}\n",
    "# pad text and tag\n",
    "has_tags = False \n",
    "\n",
    "testls_s = [i.split(\" \") for i in testls]\n",
    "#taglist_s = [i.split(\" \") for i in taglist]\n",
    "testls_win, taglist_win = sliding_window_list(testls_s, [], winSize=feat_config['window'], step=feat_config['step'], has_tags=False)\n",
    "testls_flatten = flatten_3d_list(testls_win)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame([])\n",
    "df['text'] = np.array([\" \".join(i) for i in testls_flatten], dtype='object')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 996 ms, sys: 28 ms, total: 1.02 s\n",
      "Wall time: 1.02 s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "spacy_df1 = sp.get_array_from_df_combined(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(95, 1)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1/1 [00:00<00:00, 2234.58it/s]\n",
      "100%|██████████| 1/1 [00:00<00:00, 698.12it/s]\n",
      "100%|██████████| 1/1 [00:00<00:00, 2502.57it/s]\n"
     ]
    }
   ],
   "source": [
    "bla = featurizer.transform(testls)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dict_keys(['sentences_window', 'tags_window', 'labels', 'custom_feats', 'lstm_feats', 'spacy_num_feats'])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bla.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'1 . Egbor M , Ansari T , Morris N , Green CJ , Sibbons PD . Morphometric placental villous and vascular abnormalities in early- and late - onset pre - eclampsia with and without fetal growth restriction . BJOG . 2006;113:580 - 9 . \\n 2 . Macara L , Kingdom JC , Kaufmann P , Kohnen G , Hair J , More IA , Lyall F , Greer IA . Structural analysis of placental terminal villi from growth - restricted pregnancies with abnormal umbilical artery Doppler waveforms . Placenta . 1996;17:37 - 48 . \\n 3 . Salafia CM , Ernst LM , Pezzullo JC , Wolf EJ , Rosenkrantz TS , Vintzileos AM . The very low birthweight infant : maternal complications leading to preterm birth , placental lesions , and intrauterine growth . Am J Perinatol . 1995;12:106 - 10 . \\n 4 . Salgado SS , Salgado MKR . Structural changes in pre - eclamptic and eclamptic placentas - an ultrastructural study . J Coll Physicians Surg Pak . 2011;21:482 - 6 . \\n 5 . Higgins M , Felle P , Mooney EE , Bannigan J , McAuliffe FM . Stereology of the placenta in type 1 and type 2 diabetes . Placenta . 2011;32:564 - 9 . \\n 6 . Wislocki GB , Dempsey EW . Electron Microscopy of the human placenta . Anat Rec . 1955;123:133 - 67 . \\n 7 . Terzakis J , Rhodin AG . Ultrastructure of Human Term Placenta . Anat Rec . 1961;139:279 . \\n 8 . Leiser R , Luckhardt M , Kaufmann P , Winterhager E , Bruns U. The fetal vascularisation of term human placental villi . I. Peripheral stem villi . Anat Embryol ( Berl ) . 1985;173:71 - 80 . \\n 9 . Leiser R , Krebs C , Ebert B , Dantzer V. Placental vascular corrosion cast studies : A comparison between ruminants and humans . Microsc Res Tech . 1997;38:76 - 87 . \\n 10 . Langheinrich AC , Wienhard J , Vormann S , Hau B , Bohle RM , Zygmunt M. Analysis of the fetal placental vascular tree by X - ray micro - computed tomography . Placenta . 2004;25:95 - 100 . \\n 11 . Hata T , Tanaka H , Noguchi J , Hata K. Three - dimensional ultrasound evaluation of the placenta . Placenta . 2011;32:105 - 15 . \\n 12 . Jirkovská M , Kubínová L , Janáček J , Kaláb J. 3-D study of vessels in peripheral placental villi . Image Anal Stereol . 2007;26:165 - 168 . \\n 13 . Demir R , Kosanke G , Kohnen G , Kertschanska S , Kaufmann P. Classification of human placental stem villi : review of structural and functional aspects . Microsc Res Tech . 1997;38:29 - 41 . \\n 14 . Castellucci M , Scheper M , Scheffen I , Celona A , Kaufmann P. The development of the human placental villous tree . Anat Embryol ( Berl ) . 1990;181:117 - 28 . \\n 15 . Roberts N , Magee D , Song Y , Brabazon K , Shires M , Crellin D , Orsi NM , Quirke R , Quirke P , Treanor D. Toward routine use of 3D histopathology as a research tool . Am J Pathol . 2012;180:1835 - 42 . \\n 16 . Magee D , Treanor D , Quirke P. A New Image Registration algorithm with application to 3D Histopathology . In : Microscopic Image Analysis with Applications in Biology ( 2008 ) . \\n'"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sentences_eval1[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}