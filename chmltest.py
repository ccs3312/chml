# -*- coding: ANSI -*-
import streamlit as st
#import library ����ͧ�������
#from fastai import (
#    load_learner,
#    PILImage,
#)
#from fastai import *
import glob
from random import shuffle
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import urllib.request
from pathlib import Path

#import streamlit ��㹪��� st ������㹡�����ҧ user interface
from fastai.learner import load_learner
from fastai.vision.all import PILImage

# Load the model
#MODEL_URL = "https://github.com/ccs3312/chml/blob/main/fastai_chml.pkl"
#urllib.request.urlretrieve(MODEL_URL, "fastai_chml.pkl")
#lnml = Path('fastai_chml.pkl')
learn_inf = load_learner("fastai_chml.pkl")
st.sidebar.write('### Enter image to classify')

# radio button ����Ѻ���͡��Ҩзӹ���ٻ�ҡ validation set ���� upload �ٻ�ͧ
option = st.sidebar.radio('', ['Use a validation image', 'Use your own image'])
valid_images = glob.glob('allchml/Valid/*/*')
shuffle(valid_images)
if option == 'Use a validation image':
    st.sidebar.write('### Select a validation image')
    fname = st.sidebar.selectbox('', valid_images)

else:
    st.sidebar.write('### Select an image to upload')
    fname = st.sidebar.file_uploader('',
                                     type=['png', 'jpg', 'jpeg'],
                                     accept_multiple_files=False)
    if fname is None:
        fname = valid_images[0]
# ��� title �ͧ main page
st.title("Color Harmony Classifier")
def predict(img, learn):

    # �ӹ�¨ҡ���ŷ�����
    pred, pred_idx, pred_prob = learn.predict(img)

    # ���š�÷ӹ��
    st.success(f"This is {pred} with the probability of {pred_prob[pred_idx]*100:.02f}%")
    
    # ����ٻ���١�ӹ��
    st.image(img, use_column_width=True)
# �Դ�ٻ
img = PILImage.create(fname)
# ���¡ function �ӹ��
predict(img, learn_inf)
