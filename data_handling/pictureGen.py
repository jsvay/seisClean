import numpy as np
#import matplotlib.pyplot as plt
import segyio
from scipy.signal import decimate


def reader(fname, path, shotlength, downscaling):
    index=0
    with segyio.open(path + fname,ignore_geometry=True) as f:
        start = index*shotlength
        traces = f.trace[start:start+shotlength]
        print(traces)
        traces = np.asarray(list(traces))
        print(traces)
        #traces = decimate(traces, downscaling, zero_phase=True)
    return traces


def tester(fname, path, shotlength, downscaling, index):
    with segyio.open(path + fname,ignore_geometry=True) as f:
        traces = []
        count = 0
        start = index*shotlength
        stop = start + shotlength
        for tr in range(start, stop):
            #d = decimate(f.trace[tr], downscaling, zero_phase=True)
            traces.append(f.trace[tr])
        traces = decimate(np.asarray(traces), downscaling, zero_phase=True)
    return traces


def writeFile(data, savepath, savename):
    mat = np.matrix(data)
    with open('/s0/SI_pictures/no/'+ savename + '.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line)#, fmt='%.*f')


'''
def create_image(data, savepath):
    fig = plt.figure()
    #ax = plt.axes([0,0,1,1])
    #ax.set_axis_off()
    #fig.add_axes(ax)
    plt.imshow(data.T, cmap='gray', aspect='equal')
    #fig.axes.get_xaxis().set_visible(False)
    #fig.axes.get_yaxis().set_visible(False)
    #plt.axis('off')
    plt.savefig("test.png", bbox_inches=0) #bbox_inches='tight'
    #plt.plot()

def reader_image():
    data = np.load("/s0/SI_pictures/yes/50.npy")
    print(data.shape)
    #print(data)
    return data
'''

if __name__ == '__main__':
    fname = "no1_SP_VQH_044_S1C12.segy"
    fname2 = "yes1_SP_QCQAD_000_S1C12.segy"
    path = "/s0/SI_cnn/"
    shotlength = 636
    downscaling = 2
    nShots = 1000
    savepath = "/s0/SI/train/"
    #savepath2 = "/s0/SI_pictures/train/noise/"
    import keras



    for index in range(1):#nShots):
        data = tester(fname, path, shotlength, downscaling, index)
        print(data)
        print(keras.utils.normalize(data, axis=-1, order=2))
        #print(data.shape)
        #data2 = tester(fname2, path, shotlength, downscaling, index)
        #data = data1 + data2
        #print(data.shape)
        #savename = 'n' + str(index)
        #np.save(savepath + savename, data)
        #create_image(data, savepath)

    #data = reader_image()
    #create_image(data, savepath)
