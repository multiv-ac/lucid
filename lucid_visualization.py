#!/usr/bin/env python
# coding: utf-8

#maybe relevant: https://distill.pub/2020/circuits/early-vision/

import numpy as np
import scipy.ndimage as nd
import tensorflow as tf
import matplotlib.pyplot as plt
import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
from PIL import Image
from matplotlib.image import pil_to_array
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("imagePath", help = "Path to image")
parser.add_argument("--neuron", help = 
"""Inception:
conv2d0, conv2d1, conv2d2, mixed3a, mixed3b, mixed4a, mixed4b, mixed4c, mixed4d, mixed4e, mixed5a, mixed5b, head0_bottleneck, nn0, softmax0, head1_bottleneck, nn1, softmax1, softmax2
\nIf none selected, all will be drawn.
""")
parser.add_argument("-lr", type = float, default = 0.05, help= "Learning rate. 0.05-1")
parser.add_argument("--contrast", type = int, default = 1200)
parser.add_argument("--channel", type =int, default = 50, help = "1-64. Default=60")

args = parser.parse_args()

def search_image(path_img):
    IMAGE = path_img
    img = Image.open(IMAGE)
    img.convert('RGB')

    width, height= img.size

    if height > width:
        d = height - width
        cut = d/2
        (left, top, right, bottom) = (0, cut, width, height-cut)
        img = img.crop((left, top, right, bottom))

    elif width > height:
        d = width - height
        cut = d/2
        (left, top, right, bottom) = ((cut, 0, width-cut, height))
        img = img.crop((left, top, right, bottom))

    img = img.resize((256, 256))
    width, height = img.size
    image = pil_to_array(img)
    
    return(image)

def process_img(IMAGEN):
    # Generar la transformada para usar como punto de partida para la optimizacion
    sess = tf.Session()
    fft2d = tf.signal.rfft2d(np.transpose(IMAGEN, [2, 0, 1]))
    result = sess.run(fft2d)
    real = np.real(result)
    imag = np.imag(result)
    stacked_real_imag = np.stack((real, imag))
    stacked_real_imag = stacked_real_imag[:, np.newaxis, :, :, :]
    escala = 50000 # Ajustar este numero para obtener una "version borrosa" de la imagen original
    # Si escala es muy grande la imagen de output de esta celda se ve blanca, si es muy chica se ve gris oscuro
    stacked_real_imag = (stacked_real_imag/escala).astype(np.float32)
    
    return(stacked_real_imag)

def visualization(learning_rate, neuron, channel, contrast, NRO_IMG, SAVE_P):
    LEARNING_RATE = learning_rate

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    obj  = objectives.neuron(neuron, channel)
    imgs = render.render_vis(model,  obj,
                             optimizer=optimizer,
                             transforms=[],
                             param_f=lambda: param.image(256, fft=True, decorrelate=True, init_val=NRO_IMG),  # 256 es el tamanio de la imagen
                             thresholds=(0,2), verbose=False)


    # Note that we're doubling the image scale to make artifacts more obvious
    plt.figure()
    plt.imshow(imgs[0][0])
    plt.axis('off')
    contraste = contrast # Mover este numero hasta ver algo razonable
    plt.imshow(contraste*(imgs[1][0]-imgs[0][0]) + 0.5)
    plt.savefig(SAVE_P, bbox_inches='tight')
    

model = models.InceptionV1()

filePath, baseName = os.path.split(args.imagePath)
fileName, ext = os.path.splitext(baseName)

imgToProcess = search_image(args.imagePath)
processedImg = process_img(imgToProcess)

if args.neuron:
    savePath = f"{filePath}/{fileName}_{args.neuron}_{args.channel}_{args.lr}{ext}"
    visualization(args.lr, args.neuron, args.channel, args.contrast, processedImg, savePath)
else:
    neuronas = ["conv2d0", "conv2d1", "conv2d2", "mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b", "head0_bottleneck", "nn0", "softmax0", "head1_bottleneck", "nn1", "softmax1", "softmax2"]
    for i in neuronas:
        savePath = f"{filePath}/{fileName}_{i}_{args.channel}_{args.lr}{ext}"
        visualization(args.lr, i, args.channel, args.contrast, processedImg, savePath)
