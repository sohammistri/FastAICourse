import gradio as gr
from fastai.vision.all import *

learn = load_learner('resnet18.pkl')

labels = learn.dls.vocab

title = "Pet Breed Classifier"
description = "A pet breed classifier trained on the Oxford Pets dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."

def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(fn=predict, inputs=gr.Image(), outputs=gr.Label(num_top_classes=3),\
title=title, description=description)
demo.launch()
