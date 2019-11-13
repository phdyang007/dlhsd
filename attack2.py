from scipy.misc import imread
import cv2
import numpy as np
from model import *
import configparser as cp
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])
debug = True

def get_image_from_input_id(test_file_list, id):
    '''
    return a image and its label
    '''
    img = cv2.imread(test_file_list[id].split()[0], 0)
    label = int(test_file_list[id].split()[1])
    return img, label
    
def _find_shapes(img_):
    shapes = [] #[upper_left_corner_location_y, upper_left_corner_location_x, y_length, x_length]
    img = np.copy(img_)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i][j] == 255 and img[i-1][j] == 0 and img[i][j-1] == 0:
                j_ = j
                while j_ < img.shape[1]-1 and img[i][j_] == 255:
                    j_ += 1
                x_length = j_ - j
                i_ = i
                while i_ < img.shape[0]-1 and img[i_][j] == 255:
                    i_ += 1
                y_length = i_ - i
                shapes.append([i,j,y_length,x_length])
                img[i:i+y_length][j:j+x_length] = 0
    return np.array(shapes)

def _find_vias(shapes_):
    shapes = np.copy(shapes_)
    squares = shapes[np.where(shapes[:,2]==shapes[:,3])]
    squares_shape = squares[:,2]
    vias_shape = np.amax(squares_shape, axis=0)
    vias_idx = np.where(squares_shape == vias_shape)
    vias = squares[vias_idx]
    srafs = np.delete(shapes, vias_idx, 0)
    return vias, srafs
    
def _generate_sraf_sub(srafs, save_img=False, save_dir="generate_sraf_sub/"):
    sub = []
    black_img_ = cv2.imread("black.png", 0)
    for item in srafs:
        black_img = np.copy(black_img_)
        black_img[item[0]:item[0]+item[2], item[1]:item[1]+item[3]] = -1
        sub.append(black_img)
    if save_img:
        count = 1
        for item in srafs:
            black_img = np.copy(black_img_)
            black_img[item[0]:item[0]+item[2], item[1]:item[1]+item[3]] = 255
            cv2.imwrite(save_dir+str(count)+".png", black_img)
            count += 1
    return np.array(sub, dtype=np.float32)

def _generate_sraf_add(img, vias, srafs, insert_shape=[40,90], save_img=False, save_dir="generate_sraf_add/"):
    add = []
    min_dis_to_vias = 100
    max_dis_to_vias = 500
    min_dis_to_sraf = 60
    black_img_ = cv2.imread("black.png", 0)
    black_img = np.copy(black_img_)
    for item in vias:
        center = [item[0]+int(item[2]/2), item[1]+int(item[3]/2)]
        black_img[max(0, center[0]-max_dis_to_vias):min(black_img.shape[0], center[0]+max_dis_to_vias), max(0, center[1]-max_dis_to_vias):min(black_img.shape[1], center[1]+max_dis_to_vias)] = 255
    for item in vias:
        center = [item[0]+int(item[2]/2), item[1]+int(item[3]/2)]
        black_img[max(0, center[0]-min_dis_to_vias):min(black_img.shape[0], center[0]+min_dis_to_vias), max(0, center[1]-min_dis_to_vias):min(black_img.shape[1], center[1]+min_dis_to_vias)] = 0
    for item in srafs:
        black_img[max(0, item[0]-min_dis_to_sraf):min(black_img.shape[0], item[0]+item[2]+min_dis_to_sraf), max(0, item[1]-min_dis_to_sraf):min(black_img.shape[1], item[1]+item[3]+min_dis_to_sraf)] = 0
    # TODO: implement sampling from valid space
    # iterate the space and add sraf one by one. srafs are generated randomly with width = 40 and length in range insert_shape
    for i in range(1, black_img.shape[0]-1):
        for j in range(1, black_img.shape[1]-1):
            if black_img[i][j] == 0:
                continue
            shape = np.random.randint(insert_shape[0], high=insert_shape[1]+1, size=2)
            shape[np.random.randint(0,high=2)] = 40
            if i+shape[0] <= black_img.shape[0] and j+shape[1] <= black_img.shape[1] and np.all(black_img[i:i+shape[0],j:j+shape[1]] == 255):
                img = np.copy(black_img_)
                img[i:i+shape[0],j:j+shape[1]] = 1
                add.append(img)
                black_img[max(0, i-min_dis_to_sraf):min(black_img.shape[0], i+shape[0]+min_dis_to_sraf), max(0, j-min_dis_to_sraf):min(black_img.shape[1], j+shape[1]+min_dis_to_sraf)] = 0
    if save_img:
        count = 1
        for item in add:
            img = np.copy(item)
            img *= 255
            cv2.imwrite(save_dir+str(count)+".png", img)
            count += 1
    return np.array(add, dtype=np.float32)
    
def generate_input_image(input_image, X, alpha):
    return np.sum(X*alpha.reshape(alpha.shape[0],1,1), axis=0) + input_image

def t_generate_input_image(t_img, t_X, t_alpha):
    return tf.reduce_sum(t_X * tf.reshape(t_alpha, [tf.shape(t_alpha)[0],1,1]), axis=0) + t_img

def generate_candidates(test_file_list, id):
    '''
    gengerate all candidates and save them
    '''
    img, _ = get_image_from_input_id(test_file_list, id)
    vias, srafs = _find_vias(_find_shapes(img))
    add = _generate_sraf_add(img, vias, srafs, save_img=True)
    sub = _generate_sraf_sub(srafs, save_img=True)
    return np.concatenate((add, sub))
    
def load_candidates(sub_dir="generate_sraf_sub/", add_dir="generate_sraf_add/"):
    '''
    load candidates. call this function if candidates have been saved
    by calling gengerate_candidates() in previous run.
    '''
    X = []
    for root, dirs, files in os.walk(add_dir):
        for name in files:
            if ".png" in name:
                img = np.array(cv2.imread(os.path.join(root,name),0),dtype=np.float32) / 255
                X.append(img)
    for root, dirs, files in os.walk(sub_dir):
        for name in files:
            if ".png" in name:
                img = np.array(cv2.imread(os.path.join(root,name),0),dtype=np.float32) / -255
                X.append(img)
    return np.array(X)
    

'''
Initialize Path and Global Params
'''
infile = cp.SafeConfigParser()
infile.read(sys.argv[1])

test_path   = infile.get('dir','train_path_txt')
test_list = open(test_path).readlines()
model_path = infile.get('dir','model_path')
fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
blocksize   = int(infile.get('feature','block_size'))
imgdim   = int(infile.get('feature','img_dim'))
lr = float(infile.get('feature', 'attack_learning_rate'))

'''
Preprocess data: generate DCT from image
'''
def _preprocess(input_image, X, alpha):
    img = generate_input_image(input_image, X, alpha)
    fe = feature(img, blocksize, blockdim, fealen)
    return np.expand_dims(np.rollaxis(fe, 0, 3), axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
'''
Prepare the Input
'''

max_iter = 10000
max_candidates = 50
target_idx = 20 # hotspot target index in test list


input_placeholder = tf.placeholder(tf.float32, shape=[max_candidates+1, blockdim, blockdim, fealen])
alpha = tf.get_variable(name='t_alpha', dtype=tf.float32, shape=[max_candidates], initializer=tf.constant_initializer(0.001))
dlambda = tf.get_variable(name='t_la', dtype=tf.float32, initializer=tf.constant_initializer(0.001)))
perturbation = tf.zeros(tf.float32, shape=[1,blockdim,blockdim,fealen])
for i in range(max_candidates):
    perturbation + = alpha[i]*input_placeholder[i]

input_merged = input_placeholder[0]+perturbation

#X = generate_candidates(test_list, target_idx)
X = load_candidates()
np.random.shuffle(X)
X = X[:max_candidates]
img, _ = get_image_from_input_id(test_list, target_idx)
#alpha = 1.0/X.shape[0] * np.ones((X.shape[0],1))
alpha = np.zeros((X.shape[0],1))
la = 0.0
#alpha_lr = 1
#la_lr = 1

t_alpha = tf.cast(tf.get_variable(name='t_alpha', initializer=alpha), tf.float32)
t_X = tf.cast(tf.convert_to_tensor(X), tf.float32)
t_img = tf.cast(tf.convert_to_tensor(img), tf.float32)
t_la = tf.cast(tf.Variable(la, name='t_la'), tf.float32)

t_feaarray = tf.cast(tf.convert_to_tensor(_preprocess(img, X, sigmoid(alpha))), dtype=tf.float32)

loss_1 = tf.norm(tf.reduce_sum(t_X * tf.reshape(tf.sigmoid(t_alpha), [tf.shape(tf.sigmoid(t_alpha))[0],1,1]), axis=0), 2)

predict = forward(t_feaarray)
fwd = tf.placeholder(tf.float32)
loss = loss_1 + tf.sigmoid(t_la) * fwd

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 't_' in var.name]
m_vars = [var for var in t_vars if 'model' in var.name]

opt = tf.train.RMSPropOptimizer(lr).minimize(loss, var_list=d_vars)

'''
Start attacking
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4


ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    print(ckpt_name)

'''
first attack method by minimizing L(alpha, lambda)
'''
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver    = tf.train.Saver(m_vars)
    saver.restore(sess, os.path.join(model_path, ckpt_name))
    '''
    a = np.zeros(alpha.shape)
    _t_feaarray = tf.cast(tf.convert_to_tensor(_preprocess(img, X, a)), dtype=tf.float32)
    _predict = forward(_t_feaarray).eval()
    print("predict original:")
    print(_predict)
    '''
    for iter in range(max_iter):
        prediction = predict.eval()
        opt.run(feed_dict={fwd: prediction[0][1] - prediction[0][0]})
        
        if debug and iter % 100 == 0:
            print("****************")
            print("alpha:")
            print(t_alpha.eval())
            print("lambda:")
            print(t_la.eval())
            print("loss:")
            print(loss.eval(feed_dict={fwd: prediction[0][1] - prediction[0][0]}))
            
            print("predict new:")
            print(prediction)
            print("fwd:")
            print(prediction[0][1] - prediction[0][0])
            '''
            a = t_alpha.eval()
            idx1 = np.where(a > 0)
            idx2 = np.where(a <= 0)
            a[idx1[0]] = 1.0
            a[idx2[0]] = 0.0
            _t_feaarray = tf.cast(tf.convert_to_tensor(_preprocess(img, X, a)), dtype=tf.float32)
            _predict = forward(_t_feaarray).eval()
            print("adversarial predict:")
            print(_predict)
            print("fwd:")
            print(_predict[0][1] - _predict[0][0])
            '''
exit()
'''
second attack method using cross update. not finish yet
'''
with tf.GradientTape() as tape:
    for iter in range(max_iter):
        input_image = generate_input_image(img, X, alpha)
        input_image = np.expand_dims(np.expand_dims(input_image, axis = 0), axis = -1)
        tape.watch(input_image)
        prediction = y.eval(feed_dict={x_data: input_image})
        s = np.sum(X*alpha, axis=0)
        
        # update alpha
        for idx in range (X.shape[0]):
            # Get the gradients of the loss w.r.t to the individual X.
            gradient = tape.gradient(prediction, X[idx])
            loss_to_alpha_grad = np.sum((s + la*gradient) * X[idx]) # TODO: how to transform matrix to number
            alpha[idx] -= loss_to_alpha_grad * alpha_lr
        
        # update lambda
        loss_to_la_grad = prediction
        la -= loss_to_la_grad * la_lr
        
        if debug:
            print("alpha:")
            print(alpha)
            print("****************")
            print("lambda: " + str(la))
            
        # TODO: stop criteria
        if False:
            break



