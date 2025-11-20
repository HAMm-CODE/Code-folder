import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr


Fs1, data1 = wav.read('AirportAnnouncements_10.wav')
Fs2, data2 = wav.read('CafeTeria_1.wav')

# Normalize to -1…1 range
data1 = data1 / 32767.0
data2 = data2 / 32767.0


data1_nr = nr.reduce_noise(y=data1, sr=Fs1)
data2_nr = nr.reduce_noise(y=data2, sr=Fs2)


clip1_nr = data1_nr[300:500]

clip2 = data2[300:500]
clip2_nr = data2_nr[300:500]

NFFT = 1024
noverlap = 128
mode = 'psd'
scale = 'dB'

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

def plot_spec(ax, data, Fs, title):
    # same call style as your example
    Pxx, freqs, bins, im = ax.specgram(
        data, NFFT=NFFT, Fs=Fs, noverlap=noverlap,
        mode=mode, scale=scale
    )
    ax.set_title(title)
    ax.set_ylabel('Frequency (Hz)')
    return Pxx, freqs, bins, im

# Airport sample
plot_spec(axes[0,0], clip1,    Fs1, "Airport (Original) 300–500")
plot_spec(axes[0,1], clip1_nr, Fs1, "Airport (Noise Reduced) 300–500")

# Cafeteria sample
plot_spec(axes[1,0], clip2,    Fs2, "Cafeteria (Original) 300–500")
plot_spec(axes[1,1], clip2_nr, Fs2, "Cafeteria (Noise Reduced) 300–500")

axes[1][0].set_xlabel('Time (s)')
axes[1][1].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()
