from scipy.misc import imread
import cv2
import numpy as np
from model import *
import configparser as cp
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])
debug = False

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
        black_img[item[0]:item[0]+item[2], item[1]:item[1]+item[3]] = -255
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
    # iterate the space and add sraf one by one. srafs are generated randomly with width = 40 and length in range insert_shape
    for i in range(1, black_img.shape[0]-1):
        for j in range(1, black_img.shape[1]-1):
            if black_img[i][j] == 0:
                continue
            shape = np.random.randint(insert_shape[0], high=insert_shape[1]+1, size=2)
            shape[np.random.randint(0,high=2)] = 40
            if i+shape[0] <= black_img.shape[0] and j+shape[1] <= black_img.shape[1] and np.all(black_img[i:i+shape[0],j:j+shape[1]] == 255):
                img = np.copy(black_img_)
                img[i:i+shape[0],j:j+shape[1]] = 255
                add.append(img)
                black_img[max(0, i-min_dis_to_sraf):min(black_img.shape[0], i+shape[0]+min_dis_to_sraf), max(0, j-min_dis_to_sraf):min(black_img.shape[1], j+shape[1]+min_dis_to_sraf)] = 0
    if save_img:
        count = 1
        for item in add:
            cv2.imwrite(save_dir+str(count)+".png", item)
            count += 1
    return np.array(add, dtype=np.float32)

def generate_candidates(test_file_list, id):
    '''
    gengerate all candidates and save them
    '''
    print("Generating candidates...")
    img, _ = get_image_from_input_id(test_file_list, id)
    vias, srafs = _find_vias(_find_shapes(img))
    add = _generate_sraf_add(img, vias, srafs, save_img=False)
    sub = _generate_sraf_sub(srafs, save_img=False)
    print("Generating candidates done. Total candidates: "+str(len(add)+len(sub)))
    return np.concatenate((add, sub))
    
def load_candidates(sub_dir="generate_sraf_sub/", add_dir="generate_sraf_add/"):
    '''
    load candidates. call this function if candidates have been saved
    by calling gengerate_candidates() in previous run.
    '''
    print("Loading candidates...")
    X = []
    for root, dirs, files in os.walk(add_dir):
        for name in files:
            if ".png" in name:
                img = np.array(cv2.imread(os.path.join(root,name),0),dtype=np.float32)
                X.append(img)
    for root, dirs, files in os.walk(sub_dir):
        for name in files:
            if ".png" in name:
                img = np.array(cv2.imread(os.path.join(root,name),0),dtype=np.float32)
                X.append(img)
    print("Loading candidates done. Total candidates: "+str(len(X)))
    return np.array(X)

def generate_adversarial_image(img, X, alpha):
    img = img.astype(np.int32)
    #X = np.absolute(X).astype(np.int32)
    X = X.astype(np.int32)
    alpha = alpha.astype(np.int32)
    return (img+np.sum(X*np.expand_dims(alpha,-1),axis=0)).astype(np.uint8)

'''
Initialize Path and Global Params
'''
infile = cp.SafeConfigParser()
infile.read(sys.argv[1])

test_path   = infile.get('dir','test_path_txt')
test_list = open(test_path).readlines()
model_path = infile.get('dir','model_path')
fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
blocksize   = int(infile.get('feature','block_size'))
imgdim   = int(infile.get('feature','img_dim'))
lr = float(infile.get('attack', 'attack_learning_rate'))
max_iter = int(infile.get('attack', 'max_iter'))
_max_candidates = int(infile.get('attack', 'max_candidates'))
max_perturbation = int(infile.get('attack', 'max_perturbation'))
alpha_threshold = float(infile.get('attack', 'alpha_threshold'))
attack_path = infile.get('attack', 'attack_path_txt')
img_save_dir = 'dct/attack_'+str(_max_candidates)+'_'+str(max_iter)+'/'
    
'''
Prepare the Input
'''
test_list_hs = [int(item.split()[1]) for item in test_list]
test_list_hs = np.array(test_list_hs)
idx = np.where(test_list_hs == 1) #total = 80152, hs = 6107

def _merge_image(dir, savedir, txtfile, test=0, merge=0, id_low=0, id_high=21514):
    with open(txtfile, 'w+') as f:
        if test == 1:
            for id in idx[0]:
                if id < id_low:
                    continue
                if id >= id_high:
                    break
                f.write('./'+savedir+'/'+str(id)+'.png')
                f.write('\n')
                img2, _ = get_image_from_input_id(test_list, id)
                cv2.imwrite(savedir+'/'+str(id)+'.png', img2)

        if merge == 1:
            print("merge mode")
            for root, dirs, files in os.walk(dir):
                for name in files:
                    if ".png" in name:
                        id = int(name[:-4])
                        if id < id_low or id >= id_high:
                            continue
                        if test != 1:
                            f.write('./'+savedir+'/'+str(id)+'.png')
                            f.write('\n')
                        img1 = cv2.imread(dir+'/'+str(id)+'.png', 0)
                        cv2.imwrite(savedir+'/'+str(id)+'.png', img1)
                
def _test_attack():
    imgs = []
    for i in attack_list:
        imgs.append(cv2.imread(i[:-1], 0))
    print("total images: "+str(len(imgs)))
    fearr = feature_mp(np.array(imgs))
    fearr = np.rollaxis(fearr, 1, 4)
    
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, blockdim, blockdim, fealen])
    predict = forward(input_placeholder, is_training=False)
    y      = tf.cast(tf.argmax(predict, 1), tf.int32)
    accu   = tf.reduce_mean(tf.cast(y, tf.float32))
    
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 't_' in var.name]
    m_vars = [var for var in t_vars if 'model' in var.name]
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver    = tf.train.Saver(m_vars)
        saver.restore(sess, os.path.join(model_path, ckpt_name))

        print('Hotspot Detection Accuracy is %f'%accu.eval(feed_dict={input_placeholder: fearr}))

def test_attack(dir='dct/attack_128_100', savedir='dct/merged_128_100', txtfile='dct/attack.txt', test=0, merge=0, id_low=0, id_high=21514):
    merge_image(dir=dir, savedir=savedir, txtfile=txtfile, test=test, merge=merge, id_low=id_low, id_high=id_high)
    attack_list = open(attack_path).readlines()
    _test_attack()

def validate_attack():
    img = cv2.imread("dct/tmp/img.png", 0)
    aimg = cv2.imread("dct/tmp/aimg.png", 0)
    cimg1 = cv2.imread("dct/tmp/12.png", 0)
    cimg2 = cv2.imread("dct/tmp/28.png", 0)
    cimg3 = cv2.imread("dct/tmp/59.png", 0)
    
    v_input_merged = tf.placeholder(dtype=tf.float32, shape=[1, blockdim, blockdim, fealen])
    
    v_predict = forward(v_input_merged,is_training=False)
    v_nhs_pre, v_hs_pre = tf.split(v_predict, [1, 1], 1)
    v_fwd = tf.subtract(v_hs_pre, v_nhs_pre)
    
    # generate candidates
    X = []
    X.append(cimg1)
    X.append(cimg2)
    X.append(cimg3)
    X = np.array(X)
    # dct
    input_images = []
    fe = feature(img, blocksize, blockdim, fealen)
    input_images.append(np.rollaxis(fe, 0, 3))
    for item in X:
        fe = feature(item, blocksize, blockdim, fealen)
        input_images.append(np.rollaxis(fe, 0, 3))
    input_images = np.asarray(input_images)
        
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[4, blockdim, blockdim, fealen])
    perturbation = tf.zeros(dtype=tf.float32, shape=[1, blockdim, blockdim, fealen])
    for i in range(3):
        perturbation += input_placeholder[i+1]
    input_merged = input_placeholder[0]+perturbation

    predict = forward(input_merged,is_training=False)
    nhs_pre, hs_pre = tf.split(predict, [1, 1], 1)
    fwd = tf.subtract(hs_pre, nhs_pre)
    
    t_vars = tf.trainable_variables()
    m_vars = [var for var in t_vars if 'model' in var.name]
    d_vars = [var for var in t_vars if 't_' in var.name]
    
    '''
    Config and model
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9


    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver    = tf.train.Saver(m_vars)
        saver.restore(sess, os.path.join(model_path, ckpt_name))
        
        v_input_images = []
        fe = feature(aimg, blocksize, blockdim, fealen)
        v_input_images.append(np.rollaxis(fe, 0, 3))
        v_input_images = np.asarray(v_input_images)
        v_diff = v_fwd.eval(feed_dict={v_input_merged: v_input_images})
        
        diff = fwd.eval(feed_dict={input_placeholder: input_images})
        
        print("diff: ")
        print(diff)
        print("v_diff: ")
        print(v_diff)

#validate_attack()
#exit()

#test_attack(id_low=10731, id_high=10733)
#exit()

'''
Start attack
'''
    
def attack(target_idx):
    tf.reset_default_graph()
    # test misclassification
    img, _ = get_image_from_input_id(test_list, target_idx)
    v_input_merged = tf.placeholder(dtype=tf.float32, shape=[1, blockdim, blockdim, fealen])
    
    v_predict = forward(v_input_merged,is_training=False)
    v_nhs_pre, v_hs_pre = tf.split(v_predict, [1, 1], 1)
    v_fwd = tf.subtract(v_hs_pre, v_nhs_pre)
    
    t_vars = tf.trainable_variables()
    m_vars = [var for var in t_vars if 'model' in var.name]
    
    '''
    Config and model
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9


    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver    = tf.train.Saver(m_vars)
        saver.restore(sess, os.path.join(model_path, ckpt_name))
        
        v_input_images = []
        fe = feature(img, blocksize, blockdim, fealen)
        v_input_images.append(np.rollaxis(fe, 0, 3))
        v_input_images = np.asarray(v_input_images)
        v_diff = v_fwd.eval(feed_dict={v_input_merged: v_input_images})
        if v_diff < -0.01:
            print("misclassification")
            return -1
    
    print("start attacking on id: "+str(target_idx))
    max_candidates = _max_candidates
    # generate candidates
    X = generate_candidates(test_list, target_idx)
    np.random.shuffle(X)
    if max_candidates > X.shape[0]:
        max_candidates = X.shape[0]
    X = X[:max_candidates]
    t_X = tf.placeholder(dtype=tf.float32, shape=[max_candidates, imgdim, imgdim])
    
    alpha = -10.0 + np.zeros((max_candidates,1))
    la = 100000.0
    t_alpha = tf.sigmoid(tf.cast(tf.get_variable(name='t_alpha', initializer=alpha), tf.float32))
    t_la = tf.cast(tf.Variable(la, name='t_la'), tf.float32)
    
    # dct
    input_images = []
    fe = feature(img, blocksize, blockdim, fealen)
    input_images.append(np.rollaxis(fe, 0, 3))
    for item in X:
        fe = feature(item, blocksize, blockdim, fealen)
        input_images.append(np.rollaxis(fe, 0, 3))
    input_images = np.asarray(input_images)
        
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[max_candidates + 1, blockdim, blockdim, fealen])
    perturbation = tf.zeros(dtype=tf.float32, shape=[1, blockdim, blockdim, fealen])
    for i in range(max_candidates):
        perturbation += t_alpha[i] * input_placeholder[i+1]
    input_merged = input_placeholder[0]+perturbation

    loss_1 = tf.norm(tf.reduce_sum(t_X * tf.reshape(t_alpha, [tf.shape(t_alpha)[0],1,1]), axis=0), 2)

    predict = forward(input_merged,is_training=False)
    nhs_pre, hs_pre = tf.split(predict, [1, 1], 1)
    fwd = tf.subtract(hs_pre, nhs_pre)

    loss = loss_1 + t_la * fwd

    t_vars = tf.trainable_variables()
    m_vars = [var for var in t_vars if 'model' in var.name]
    d_vars = [var for var in t_vars if 't_' in var.name]
    opt = tf.train.RMSPropOptimizer(lr).minimize(loss, var_list=d_vars)

    '''
    first attack method by minimizing L(alpha, lambda)
    '''
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver    = tf.train.Saver(m_vars)
        saver.restore(sess, os.path.join(model_path, ckpt_name))
        
        interval = 10
        
        for iter in range(max_iter):
            opt.run(feed_dict={input_placeholder: input_images, t_X: X})
            
            if iter % interval == 0:
                a = t_alpha.eval()
                diff = fwd.eval(feed_dict={input_placeholder: input_images, t_X: X})
                if debug:
                    print("****************")
                    print("alpha:")
                    print(a)
                    
                    print("lambda:")
                    print(t_la.eval())
                    
                    print("fwd:")
                    print(diff)
                    
                    print("loss_1:")
                    print(loss_1.eval(feed_dict={input_placeholder:
input_images, t_X: X}))
   
                    print("loss:")
                    print(loss.eval(feed_dict={input_placeholder: input_images, t_X: X}))
                
                if diff < -0.0:
                    interval = 5
                    idx = []
                    b = np.copy(a)
                    for i in range(max_perturbation):
                        idx.append(np.argmax(b))
                        b = np.delete(b, idx[-1])
                        c = np.zeros(a.shape)
                        c[idx] = 1.0
                        diff = fwd.eval(feed_dict={input_placeholder: input_images, t_X: X, t_alpha: c})
                        if diff <= -0.01:
                            aimg = generate_adversarial_image(img, X, c)
                            v_input_images = []
                            fe = feature(aimg, blocksize, blockdim, fealen)
                            v_input_images.append(np.rollaxis(fe, 0, 3))
                            v_input_images = np.asarray(v_input_images)
                            v_diff = v_fwd.eval(feed_dict={v_input_merged: v_input_images})
                            #im = input_merged.eval(feed_dict={input_placeholder: input_images, t_X: X})
                            #print("dis: ")
                            #print(np.sum(im-v_input_images))
                            if v_diff > 0:
                                print("False attack")
                                continue
                            cv2.imwrite(img_save_dir+str(target_idx)+'.png', aimg)
                            print("ATTACK SUCCEED: sarfs add: "+str(len(idx)))
                            print("****************")
                            return 1
        
        print("max iteration reached")
        a = t_alpha.eval()
        idx = []
        b = np.copy(a)
        for i in range(max_perturbation):
            idx.append(np.argmax(b))
            b = np.delete(b, idx[-1])
            c = np.zeros(a.shape)
            c[idx] = 1.0
            diff = fwd.eval(feed_dict={input_placeholder: input_images, t_X: X, t_alpha: c})
        
            if diff <= -0.01:
                aimg = generate_adversarial_image(img, X, c)
                v_input_images = []
                fe = feature(aimg, blocksize, blockdim, fealen)
                v_input_images.append(np.rollaxis(fe, 0, 3))
                v_input_images = np.asarray(v_input_images)
                v_diff = v_fwd.eval(feed_dict={v_input_merged: v_input_images})
                #im = input_merged.eval(feed_dict={input_placeholder: input_images, t_X: X})
                #print("dis: ")
                #print(np.sum(im-v_input_images))
                if v_diff > 0:
                    print("False attack")
                    continue
                cv2.imwrite(img_save_dir+str(target_idx)+'.png', aimg)
                print("ATTACK SUCCEED: sarfs add: "+str(len(idx)))
                print("****************")
                return 1
        
        print("ATTACK FAIL: sraf not enough")
        print("****************")
        return 0


success = 0
total = 0
for id in idx[0]:
    if id < 0:
        continue
    if id >= 21514:
        exit()

    ret = attack(id)
    if ret != -1:
        total += 1
        success += ret
    print("success attack: [ "+str(success)+" / "+str(total)+" ]")
