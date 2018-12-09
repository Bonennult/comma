import os
import numpy as np
import json
import matplotlib.pyplot as plt
from data_reader import datagen, read_by_idx

'''
def save_gif(gif_fname, images, fps):
    """
    To generate a gif from image files, first generate palette from images
    and then generate the gif from the images and the palette.
    ffmpeg -i input_%02d.jpg -vf palettegen -y palette.png
    ffmpeg -i input_%02d.jpg -i palette.png -lavfi paletteuse -y output.gif
    Alternatively, use a filter to map the input images to both the palette
    and gif commands, while also passing the palette to the gif command.
    ffmpeg -i input_%02d.jpg -filter_complex "[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse" -y output.gif
    To directly pass in numpy images, use rawvideo format and `-i -` option.
    """
    from subprocess import Popen, PIPE
    head, tail = os.path.split(gif_fname)
    if head and not os.path.exists(head):
        os.makedirs(head)
    h, w, c = images[0].shape
    cmd = ['ffmpeg', '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-r', '%.02f' % fps,
           '-s', '%dx%d' % (w, h),
           '-pix_fmt', {1: 'gray', 3: 'rgb24', 4: 'rgba'}[c],
           '-i', '-',
           '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
           '-r', '%.02f' % fps,
           '%s' % gif_fname]
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = '\n'.join([' '.join(cmd), err.decode('utf8')])
        raise IOError(err)
    del proc

def encode_gif(images, fps):
    from subprocess import Popen, PIPE
    h, w, c = images[0].shape
    cmd = ['ffmpeg', '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-r', '%.02f' % fps,
           '-s', '%dx%d' % (w, h),
           '-pix_fmt', {1: 'gray', 3: 'rgb24', 4: 'rgba'}[c],
           '-i', '-',
           '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
           '-r', '%.02f' % fps,
           '-f', 'gif',
           '-']
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = '\n'.join([' '.join(cmd), err.decode('utf8')])
        raise IOError(err)
    del proc
    return out
'''


def metric_mse(img1,img2):
    assert img1.shape==img2.shape
    return np.mean(np.square(img2-img1))

def metric_l1(img1,img2):
    assert img1.shape==img2.shape
    return np.mean(np.abs(img2-img1))

def metric_psnr(true, pred, keep_axis=None):
    mse = metric_mse(true,pred)
    psnr = 10.0 * np.log10(255**2 / mse)
    return psnr

def metric_ssim(true, pred):
    from skimage.measure import compare_ssim
    assert true.shape == pred.shape
    ssim = compare_ssim(true, pred, multichannel=True)
    return ssim


def save_log(fname, glob, data_type, index):
    try:
        with open(fname, 'r') as f:
            info = json.load(f)
        info['glob'] = glob
        for i in range(len(index)):      # 转为str以便存入json文件
            index[i] = str(index[i])
        if data_type is None:
            pass
        elif len(data_type) == 1:        # 所有类相同时，data_type可以只传入一个元素
            d_type = data_type * len(index)
        else:
            d_type = data_type
        for k in range(len(d_type)):
            if d_type[k] not in info.keys():
                info[d_type[k]] = [index[k]]
            else:
                info[d_type[k]] = info[d_type[k]] + [index[k]]
            info[d_type[k]] = list(set(info[d_type[k]]))

    except FileNotFoundError:
        info = {'glob':'0'}
        pass

    with open(fname, 'w') as f:
        f.write(json.dumps(info))
    
def read_log():
    try:
        with open('./log_modefied/train_index.json', 'r') as f:
            info = json.load(f)
            ratio = {}
            total = 0
            for k,v in info.items():
                if k == 'glob':
                    continue
                ratio[k] = len(v)
                total = total + len(v)
            for k,v in ratio.items():
                print(k + ' : %2.2f %%' % (100*v/total))
    except FileNotFoundError:
        info = {'glob':'0'}
        with open('./log_modefied/train_index.json', 'w') as f:
            f.write(json.dumps(info))
    
    return info['glob']


train_path = [
    './camera/2016-01-30--11-24-51.h5',
    './camera/2016-01-31--19-19-25.h5',
    './camera/2016-02-02--10-16-58.h5',
    './camera/2016-02-08--14-56-28.h5',
    './camera/2016-02-11--21-32-47.h5',
    './camera/2016-03-29--10-50-20.h5',
    './camera/2016-04-21--14-48-08.h5',]

# 2 for validation
validation_path = [
    './camera/2016-06-02--21-39-29.h5',
    './camera/2016-06-08--11-46-01.h5']

# 2 for test
test_path = [
    './camera/2016-01-30--13-46-00.h5',
    './camera/2016-05-12--22-20-00.h5',]

datapath = train_path
time_length = 30
pack_size = 6            # 每个tfrecords文件中有256个视频序列
fps = 6
# yield (X_batch, dset_num, index, glob)
glob = int(read_log())
print(glob)
gen = datagen(datapath, time_len=time_length, batch_size=pack_size, ignore_goods=False, global_idx=glob)
condition_type = {'-1':'false','0':'others','1':'daytime','2':'night_normal','3':'traffic_jam','4':'back_off','5':'turn',}
fname = './log_modefied/train_index.json'

while True:
    try:
        data = next(gen)
        image = np.transpose(data[0],(0,1,3,4,2))  #image=(batch,time_length,height,width,channel)
        #index = data[2]       #a 2-d list : [[30,31,...59],[60,61,...,89],]
        index = data[3]        # a list : [30,60,90]
        print(data[3])
        glob = max(data[3])    # max number of a list
        print(glob)

        dist_mse=[]
        dist_l1=[]
        dist_psnr=[]
        dist_ssim=[]
        for k in range(image.shape[0]):
            #save_gif('./log_modefied/image_%d.gif'%k, image[k], fps=fps)   #保存为gif
            temp_image=image[k]
            dist_mse.append([])
            dist_l1.append([])
            dist_psnr.append([])
            dist_ssim.append([])
            step = 3
            for j in range(0,temp_image.shape[0],step):
                if j+step >= temp_image.shape[0]:
                    break
                dist_mse[k].append(metric_mse(temp_image[j],temp_image[j+step]))
                dist_l1[k].append(metric_l1(temp_image[j],temp_image[j+step]))
                dist_psnr[k].append(metric_psnr(temp_image[j],temp_image[j+step]))
                dist_ssim[k].append(100*metric_ssim(temp_image[j],temp_image[j+step]))

        distance = []
        for k in range(len(dist_ssim)):
            mse = np.mean(dist_mse[k])
            l1 = np.mean(dist_l1[k])
            psnr = np.mean(dist_psnr[k])
            ssim = np.mean(dist_ssim[k])
            mask = [0.1, 0.3, 0.1, 0.5]  # mse, l1, psnr, ssim
            dist = [mse/50, l1/120, psnr/50, 1/(ssim)]
            distance.append(1000*np.dot(mask, dist))

        print(distance)
        threshold = 350
        logic = [k>threshold for k in distance]
        print(logic)
        index = [data[3][k] for k in range(len(logic)) if logic[k]==True]

        d_type = [condition_type['0']]
        #save_log(glob = glob, data_type = None, index = [])
        save_log(fname, glob = str(glob), data_type = d_type, index = index)
        
        index = [data[3][k] for k in range(len(logic)) if logic[k]==False]
        d_type = [condition_type['-1']]
        #save_log(glob = glob, data_type = None, index = [])
        save_log('./log_modefied/false_index.json', glob = str(glob), data_type = d_type, index = index)
    
    except StopIteration :
        print('all done!')
        exit()