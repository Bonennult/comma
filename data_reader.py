"""
This file is named after `dask` for historical reasons. We first tried to
use dask to coordinate the hdf5 buckets but it was slow and we wrote our own
stuff.
"""
import numpy as np
import h5py
import time
import logging
import traceback

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def concatenate(camera_names, time_len):
    logs_names = [x.replace('camera', 'log') for x in camera_names]

    angle = []  # steering angle of the car
    speed = []  # steering angle of the car
    hdf5_camera = []  # the camera hdf5 files need to continue open
    c5x = []
    filters = []
    lastidx = 0

    for cword, tword in zip(camera_names, logs_names):
        try:
            with h5py.File(tword, "r") as t5:
                c5 = h5py.File(cword, "r")
                hdf5_camera.append(c5)               #打开的文件
                x = c5["X"]
                c5x.append((lastidx, lastidx+x.shape[0], x))

                speed_value = t5["speed"][:]
                steering_angle = t5["steering_angle"][:]
                idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment
                angle.append(steering_angle[idxs])   #angle标记数据
                speed.append(speed_value[idxs])      #speed标记数据

                goods = np.abs(angle[-1]) <= 200

                #print(np.argwhere(goods)[time_len-1:])
                filters.append(np.argwhere(goods)[time_len-1:] + (lastidx+time_len-1))
                #print(filters)
                lastidx += goods.shape[0]
                # check for mismatched length bug
                print("x {} | t {} | f {}".format(x.shape[0], steering_angle.shape[0], goods.shape[0]))
                if x.shape[0] != angle[-1].shape[0] or x.shape[0] != goods.shape[0]:
                    raise Exception("bad shape")

        except IOError:
            import traceback
            traceback.print_exc()
            print("failed to open", tword)

    angle = np.concatenate(angle, axis=0)
    speed = np.concatenate(speed, axis=0)
    filters = np.concatenate(filters, axis=0).ravel()
    print("training on %d/%d examples" % (filters.shape[0], angle.shape[0]))
    dset_num = filters.shape[0]
    #return dset_num, c5x, angle, speed, filters, hdf5_camera
    return dset_num, c5x, filters, hdf5_camera


#first = True
#index

def datagen(filter_files, time_len=1, batch_size=1, ignore_goods=False, global_idx=0):
    """
    Parameters:
    -----------
    leads : bool, should we use all x, y and speed radar leads? default is false, uses only x
    """
    #global first
    #global index
    assert time_len > 0
    filter_names = sorted(filter_files)

    logger.info("Loading {} hdf5 buckets.".format(len(filter_names)))

    dset_num, c5x, filters, hdf5_camera = concatenate(filter_names, time_len=time_len)
    filters_set = set(filters)

    logger.info("camera files {}".format(len(c5x)))

    X_batch = np.zeros((batch_size, time_len, 3, 160, 320), dtype='uint8')
    #angle_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
    #speed_batch = np.zeros((batch_size, time_len, 1), dtype='float32')

    glob = global_idx if global_idx else filters[time_len-1]   #增加的
  

    while glob <= filters[-1]:
        try:
            t = time.time()
            index = []
            glob_idx = []
            count = 0
            start = time.time()
            while count < batch_size:
                index.append([])
                if not ignore_goods:
                    #i = np.random.choice(filters)
                    i = glob
                    # check the time history for goods
                    good = True
                    for j in range(i-time_len+1, i+1):   #加了range
                        if j not in filters_set:
                            good = False
                        if j > filters[-1]:
                            #exit()
                            raise StopIteration
                    if not good:
                        glob = glob + 1             #增加的
                        continue
                    index[count] = range(glob-time_len+1,glob+1)
                    glob = glob + time_len        #增加的
                    glob_idx.append(glob)

                else:
                    #i = np.random.randint(time_len+1, len(angle), 1)
                    raise NotImplementedError 

                # GET X_BATCH
                # low quality loop
                for es, ee, x in c5x:
                    if i >= es and i < ee:
                        X_batch[count] = x[i-es-time_len+1:i-es+1]
                        break

                #angle_batch[count] = np.copy(angle[i-time_len+1:i+1])[:, None]
                #speed_batch[count] = np.copy(speed[i-time_len+1:i+1])[:, None]

                count += 1

            # sanity check
            assert X_batch.shape == (batch_size, time_len, 3, 160, 320)

            logging.debug("load image : {}s".format(time.time()-t))
            print("%5.2f ms" % ((time.time()-start)*1000.0))

            #if first:
                #print("X", X_batch.shape)
                #print("angle", angle_batch.shape)
                #print("speed", speed_batch.shape)
                # time.sleep(3)
                #first = False

            #X_batch = np.transpose(X_batch,(0,1,3,4,2))   # 将图像形状由(3,hwight,width)转换为(height,width,3)
            #yield (X_batch, angle_batch, speed_batch, dset_num, index)
            yield (X_batch, dset_num, index, glob_idx)

        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            pass
        
    exit()

def read_by_idx(filter_files, time_len=1, batch_size=1, ignore_goods=False, global_idx=0):
    assert time_len > 0
    filter_names = sorted(filter_files)

    dset_num, c5x, filters, hdf5_camera = concatenate(filter_names, time_len=time_len)
    filters_set = set(filters)

    X_batch = np.zeros((batch_size, time_len, 3, 160, 320), dtype='uint8')

    index = global_idx   #增加的

    while True:
        try:
            t = time.time()

            glob_idx = []
            count = 0
            start = time.time()
            while count < batch_size:
                if not ignore_goods:
                    i = index
                    good = True
                    for j in range(i-time_len+1, i+1):   #加了range
                        if j not in filters_set:
                            good = False
                    if not good:
                        index = index + 1             #增加的
                        continue
                    glob_idx.append(index)
                    index = index + time_len        #增加的

                else:
                    raise NotImplementedError 

                # GET X_BATCH
                # low quality loop
                for es, ee, x in c5x:
                    if i >= es and i < ee:
                        X_batch[count] = x[i-es-time_len+1:i-es+1]
                        break

                count += 1

            # sanity check
            assert X_batch.shape == (batch_size, time_len, 3, 160, 320)

            print("%5.2f ms" % ((time.time()-start)*1000.0))
            return (np.transpose(X_batch,(0,1,3,4,2)), glob_idx)

        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            pass
