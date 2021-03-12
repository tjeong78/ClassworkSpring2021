import numpy as np
import matplotlib.pyplot as plt
import json


def getData(filename):
    """Reads and parses ECG data and returns as numpy array.

    ECG data is given in two columns as time (s) and voltage (mV). This
    function parses data to be passed to another function which can then
    be passed to another function to plot.

    Args:
        filename (string): a .csv file containing the sample ECG data

    Returns:
        my_data (float array): a numpy array containing the time and voltage
    """
    ecg_data = np.genfromtxt(filename, delimiter=',')
    np.asarray(ecg_data)
    return ecg_data


def frequencyFiltering(ecg_data):
    time = ecg_data[:,0]
    n = len(time)
    dt = max(time/n)
    v = ecg_data[:,1]
    vf = np.fft.fft(v,n)
    PSD = vf * np.conj(vf)/n
    freq = (1/(n*dt)) * np.arange(n)
    # L = np.arange(1,np.floor(n/2),dtype='int')
    # plt.plot(freq,PSD,color='c',label='Noisy')
    peaks = findPeaks(PSD)
    for peak in peaks:
        lowerBound = peak-30
        if lowerBound < 0:
            lowerBound = 0
        upperBound = peak+30
        if upperBound > len(freq):
            upperBound = len(freq)
        vf[lowerBound:upperBound] = 0
    # plt.xlim(-20,400)
    # plt.legend()
    # plt.show()
    # cutoffs = PSD < 0.0001*max(PSD[peaks])
    # vf = cutoffs * vf
    ffilt = np.fft.ifft(vf)
    plt.plot(time, ffilt,'k', linewidth=0.2)
    plt.xlim(5,10)
    plt.show()
    for index, value in enumerate(PSD):
        if value > 0.2*any(PSD[peaks]):
            PSD[index] = 0
    # plt.plot(freq,PSD,'k',label='Clean')
    # plt.xlim(-20,400)
    # plt.legend()
    # plt.show()
    return np.stack((time,ffilt), axis=-1)

    
def findPeaks(y):
    y = np.array(y)
    peaks = []
    for index, data in enumerate(y): # find all possible peaks
        if index == 0:
            pass
        else:
            if (y[index] >= y[index-1]) and (y[index] > y[index+1]):
                peaks.append(index)
    unwanted = []
    for i in peaks: # trim minor peaks
        for j in peaks:
            if np.real(y[i]) < 0.1*np.real(y[j]):
                unwanted.append(i)
                break
    for i in unwanted:
        peaks.remove(i)
    return peaks


def plotECG(ecg_data):
    dist = 2000
    time = ecg_data[0:dist,0]
    voltage = ecg_data[0:dist,1]
    plt.plot(time, voltage, 'b-', linewidth=0.1)
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.title("ECG Data")
    plt.figure(num=None, figsize=(8,6), dpi=800, facecolor='w',edgecolor='k')
    #plt.show()


def heartbeatTimes(data):
    t = data[:,1]
    v = data[:,2]
    index = findPeaks(v)
    return t[findPeaks]


def duration(data):
    time = data[:,1]
    ecg_duration = max(time)-min(time)
    return ecg_duration


def numHeartbeats(data):
    v = data[:,2]
    index = findPeaks(v)
    heartbeats = len(index)
    return heartbeats


def heartrate(data):
    heartbeats = numHeartbeats(data)
    ecg_duration = duration(data)
    bpm = heartbeats/ecg_duration*60
    return bpm


def voltageExtremes(data):
    v = data[:,2]
    max_voltage = max(v)
    min_voltage = min(v)
    extreme_voltages = np.stack((max_voltage,min_voltage))


def makeDict(data,filename):
    metrics = {
        "Duration": duration(data),
        "Maximum Voltage": voltageExtremes(data)[1],
        "Minimum Voltage": voltageExtremes(data)[2],
        "Number of Heartbeats": numHeartbeats(data),
        "Mean Heartrate (bpm)": heartrate(data),
        "Heartbeat Instances": heartbeatTimes(data)}
    filename = filename + ".json"
    out_file = open(filename,"w")
    json.dump(metrics,out_file)
    out_file.close()


for i in range(32):
    i = i + 1
    filename = "test_data" + str(i) + ".csv"    
    test_data = getData(filename)
    filt_data = frequencyFiltering(test_data)
    plotECG(filt_data)
    makeDict(test_data, filename-".csv")