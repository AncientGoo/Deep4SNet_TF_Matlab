import os
import librosa
import gc
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import gc

human_dir = "Training_Data/human/"
spoof_dir = "Training_Data/spoof/"

def make_pathlist(directory):
    pathlist = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        pathlist.append(str(f))
    return pathlist

def save_sound_as_hist(pnf):
    path, n, f = pnf
    sound, _ = librosa.load(path)
    
    fig = plt.figure()
    
    plt.hist(sound/sound.max(), bins=65536)
    plt.axis('off')

    if f == 1:
        plt.savefig('data/human/hist_1_{}.png'.format(n), bbox_inches='tight', pad_inches=0.01)
    else:
        plt.savefig('data/spoof/hist_0_{}.png'.format(n), bbox_inches='tight', pad_inches=0.01)

    fig.clf()
    plt.close('all')
    del path, n, f
    gc.collect()


def make_hist(human_dir, spoof_dir):

    path_list_1 = make_pathlist(human_dir)
    len1 = len(path_list_1)
    #path_list_1 = path_list_1[5100:]
    
    path_list_0 = make_pathlist(spoof_dir)
    len0 = len(path_list_0)
    path_list_0 = path_list_0[28264:]
    

    if __name__ == '__main__':
        
        #with Pool(11) as p:
        #    sound, _ = p.map(save_sound_as_hist, zip(path_list_1, range(len1), np.ones(len1)))

        with Pool(11) as p:
            sound, _ = p.map(save_sound_as_hist, zip(path_list_0, range(28264, len0), np.zeros(len0)))


#print(len(make_pathlist(spoof_dir)))
make_hist(human_dir, spoof_dir)