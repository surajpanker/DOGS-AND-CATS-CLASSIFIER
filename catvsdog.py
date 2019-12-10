
# coding: utf-8

# In[76]:


import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks.

import tkinter 
from tkinter import messagebox
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk,Image

import subprocess as sub

main_win = tkinter.Tk()
main_win.geometry("720x300")
main_win.title("ANIMAL CLASSIFIER")
main_win.sourceFolder = ''
main_win.sourceFolder2 = ''

canvas = Canvas(main_win, width = 800, height = 125)      
canvas.pack()
img = ImageTk.PhotoImage(Image.open("HEAD.png"))  
canvas.create_image(170,5, anchor=NW, image=img)

f=Frame(main_win)
label1=tkinter.Label(main_win,text="Location : ",width = 9, height = 1,anchor=NE)
label1.pack(side=LEFT,fill=X,anchor=NE,expand=NO)
e1=tkinter.Entry(main_win,bd=1,width = 10)
e1.pack(side=LEFT,fill=X,anchor=NW,ipadx=100)
label2=tkinter.Label(main_win,text="Location : ",width = 15, height = 1,anchor=NE)
label2.pack(side=LEFT,fill=X,anchor=NE,expand=NO)
e2=tkinter.Entry(main_win,bd=1,width = 10)
e2.pack(side=LEFT,fill=X,anchor=NW,ipadx=100)
f.pack(fill=X,expand=YES)

def chooseDir():
    main_win.sourceFolder =  filedialog.askdirectory(parent=main_win, initialdir= "/", title='Please select a directory for TRAIN folder')
    e1.insert(0,main_win.sourceFolder)

b_chooseDir = tkinter.Button(main_win, text = "Browse TRAIN Folder", width = 20, height = 2, command = chooseDir)
b_chooseDir.place(x = 185,y = 155)
b_chooseDir.width = 200

def chooseDir2():
    main_win.sourceFolder2 =  filedialog.askdirectory(parent=main_win, initialdir= "/", title='Please select a directory for TEST folder')
    e2.insert(0,main_win.sourceFolder2)

b_chooseDir2 = tkinter.Button(main_win, text = "Browse TEST Folder", width = 20, height = 2, command = chooseDir2)
b_chooseDir2.place(x = 560,y = 155)
b_chooseDir2.width = 200

def close_window (): 
    main_win.destroy()

button = tkinter.Button (main_win, text = "OK", width = 20, height = 2, command = close_window)
button.place(x =560,y = 250)
button.width = 200

main_win.mainloop()

TRAIN_DIR = main_win.sourceFolder
TEST_DIR = main_win.sourceFolder2
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '5conv-basic') # just so we remember which saved model is which, sizes must match


# In[77]:


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very dog]
    elif word_label == 'dog': return [0,1]


# In[78]:


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


# In[79]:


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


# In[87]:


train_data = create_train_data()
# If you have already created the dataset:
#train_data = np.load('train_data.npy')


# In[88]:


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


# In[89]:


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


# In[90]:


train = train_data[:-100]
test = train_data[-100:]


# In[91]:


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]


# In[92]:


model.fit({'input': X}, {'targets': Y}, n_epoch=20, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


# In[93]:


model.save(MODEL_NAME)


# In[94]:


import matplotlib.pyplot as plt

# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:
#test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

import numpy as np
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = tkinter.PhotoImage(master=canvas, width=figure_w, height=figure_h)

    # Position: convert from top-left anchor to center anchor
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

    # Unfortunately, there's no accessor for the pointer to the native renderer
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo
plot_win=tkinter.Tk()
plot_win.geometry("400x300")
plot_win.title("ANIMAL CLASSIFIER")
canvas2 = Canvas(plot_win, width=400, height=300)
canvas2.pack()
fig_photo = draw_figure(canvas2, fig, loc=(0,0))
fig_w, fig_h = fig_photo.width(), fig_photo.height()
plot_win.mainloop()
plt.show()