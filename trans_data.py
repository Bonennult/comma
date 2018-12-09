import os
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import python_io
from skimage import io, transform

from dask_generator import datagen


train_path = [
    './camera/2016-01-30--11-24-51.h5',
    './camera/2016-01-31--19-19-25.h5',
    './camera/2016-02-02--10-16-58.h5',
    './camera/2016-02-08--14-56-28.h5',
    './camera/2016-02-11--21-32-47.h5',
    './camera/2016-03-29--10-50-20.h5',
    './camera/2016-04-21--14-48-08.h5',
]
'''
train_path = [
    './camera/2016-01-31--19-19-25.h5',]
'''

# 2 for validation
validation_path = [
    './camera/2016-06-02--21-39-29.h5',
    './camera/2016-06-08--11-46-01.h5'
]

# 2 for test
test_path = [
    './camera/2016-01-30--13-46-00.h5',
    './camera/2016-05-12--22-20-00.h5',
]

datapath = test_path
time_length = 30
pack_size = 256   # 每个tfrecords文件中有256个视频序列
gen = datagen(datapath, time_len=time_length, batch_size=pack_size, ignore_goods=False)
data = next(gen)
dataset_num = data[3]

count = 0

while count*30<dataset_num/25:#dataset_num :
    try :
        data = next(gen)  # 取出256*30张图片
        starts = count
        ends = starts + pack_size - 1
        count = count + pack_size
        tfrecords_filename = '../video_prediction/data/comma/test/traj_%d_to_%d.tfrecords'%(starts, ends)
        writer = python_io.TFRecordWriter(tfrecords_filename) # 创建.tfrecord文件，准备写入
            
        for i in range(pack_size):
            #data = next(gen)
            img = np.transpose(data[0][i],(0,2,3,1))
            angle = data[1][i]
            speed = data[2][i]
            img_raw = []
            for k in range(img.shape[0]):
                #img_resize = transform.rescale(img[k],0.25,mode='constant')   # resize to 40*80
                #img_raw.append(img_resize.tostring())
                img_raw.append(img[k].tostring())
                
            example = tf.train.Example(features=tf.train.Features(
                    feature={
                        '0/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[0]])),
                        '0/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[0]])),
                        '0/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[0]])),
                        '1/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[1]])),
                        '1/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[1]])),
                        '1/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[1]])),
                        '2/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[2]])),
                        '2/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[2]])),
                        '2/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[2]])),
                        '3/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[3]])),
                        '3/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[3]])),
                        '3/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[3]])),
                        '4/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[4]])),
                        '4/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[4]])),
                        '4/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[4]])),
                        '5/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[5]])),
                        '5/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[5]])),
                        '5/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[5]])),
                        '6/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[6]])),
                        '6/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[6]])),
                        '6/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[6]])),
                        '7/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[7]])),
                        '7/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[7]])),
                        '7/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[7]])),
                        '8/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[8]])),
                        '8/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[8]])),
                        '8/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[8]])),
                        '9/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[9]])),
                        '9/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[9]])),
                        '9/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[9]])),
                        '10/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[10]])),
                        '10/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[10]])),
                        '10/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[10]])),
                        '11/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[11]])),
                        '11/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[11]])),
                        '11/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[11]])),
                        '12/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[12]])),
                        '12/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[12]])),
                        '12/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[12]])),
                        '13/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[13]])),
                        '13/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[13]])),
                        '13/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[13]])),
                        '14/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[14]])),
                        '14/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[14]])),
                        '14/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[14]])),
                        '15/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[15]])),
                        '15/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[15]])),
                        '15/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[15]])),
                        '16/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[16]])),
                        '16/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[16]])),
                        '16/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[16]])),
                        '17/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[17]])),
                        '17/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[17]])),
                        '17/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[17]])),
                        '18/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[18]])),
                        '18/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[18]])),
                        '18/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[18]])),
                        '19/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[19]])),
                        '19/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[19]])),
                        '19/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[19]])),
                        '20/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[20]])),
                        '20/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[20]])),
                        '20/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[20]])),
                        '21/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[21]])),
                        '21/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[21]])),
                        '21/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[21]])),
                        '22/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[22]])),
                        '22/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[22]])),
                        '22/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[22]])),
                        '23/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[23]])),
                        '23/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[23]])),
                        '23/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[23]])),
                        '24/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[24]])),
                        '24/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[24]])),
                        '24/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[24]])),
                        '25/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[25]])),
                        '25/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[25]])),
                        '25/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[25]])),
                        '26/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[26]])),
                        '26/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[26]])),
                        '26/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[26]])),
                        '27/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[27]])),
                        '27/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[27]])),
                        '27/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[27]])),
                        '28/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[28]])),
                        '28/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[28]])),
                        '28/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[28]])),
                        '29/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[29]])),
                        '29/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[29]])),
                        '29/iamge_aux1/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[29]]))
                    }))
            writer.write(example.SerializeToString()) 
        writer.close()
        print('==================== count = %d / %d\n==================== progress = %2.2f'%(count*30, dataset_num, count*30/dataset_num*100))
        #time.sleep(3)
    except StopIteration :
        writer.close()
        exit()
        
exit()


'''
'sequence_length':tf.train.Feature(int64_list=tf.train.Int64List(value=[20])),
                        'channels':tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
                        '0/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[0]])),
                        '0/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[0]])),
                        '0/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[0]])),
                        '1/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[1]])),
                        '1/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[1]])),
                        '1/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[1]])),
                        '2/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[2]])),
                        '2/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[2]])),
                        '2/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[2]])),
                        '3/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[3]])),
                        '3/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[3]])),
                        '3/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[3]])),
                        '4/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[4]])),
                        '4/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[4]])),
                        '4/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[4]])),
                        '5/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[5]])),
                        '5/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[5]])),
                        '5/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[5]])),
                        '6/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[6]])),
                        '6/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[6]])),
                        '6/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[6]])),
                        '7/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[7]])),
                        '7/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[7]])),
                        '7/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[7]])),
                        '8/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[8]])),
                        '8/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[8]])),
                        '8/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[8]])),
                        '9/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[9]])),
                        '9/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[9]])),
                        '9/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[9]])),
                        '10/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[10]])),
                        '10/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[10]])),
                        '10/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[10]])),
                        '11/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[11]])),
                        '11/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[11]])),
                        '11/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[11]])),
                        '12/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[12]])),
                        '12/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[12]])),
                        '12/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[12]])),
                        '13/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[13]])),
                        '13/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[13]])),
                        '13/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[13]])),
                        '14/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[14]])),
                        '14/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[14]])),
                        '14/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[14]])),
                        '15/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[15]])),
                        '15/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[15]])),
                        '15/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[15]])),
                        '16/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[16]])),
                        '16/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[16]])),
                        '16/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[16]])),
                        '17/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[17]])),
                        '17/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[17]])),
                        '17/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[17]])),
                        '18/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[18]])),
                        '18/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[18]])),
                        '18/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[18]])),
                        '19/angle':tf.train.Feature(float_list = tf.train.FloatList(value=[angle[19]])),
                        '19/speed':tf.train.Feature(float_list = tf.train.FloatList(value=[speed[19]])),
                        '19/iamges/encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw[19]]))
'''