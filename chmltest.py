# -*- coding: UTF-8 -*-
import streamlit as st

import glob
from random import shuffle
#import pathlib
#import urllib.request
#from pathlib import Path

from fastai.learner import load_learner
from fastai.vision.all import PILImage
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

learn_inf = load_learner("fastai_chml.pkl")
st.sidebar.write('### Enter image to classify')

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

st.title("Color Harmony Classifier")
def predict(img, learn):


    pred, pred_idx, pred_prob = learn.predict(img)


    st.success(f"This is {pred} with the probability of {pred_prob[pred_idx]*100:.02f}%")
    

    st.image(img, use_column_width=True)

img = PILImage.create(fname)

predict(img, learn_inf)
