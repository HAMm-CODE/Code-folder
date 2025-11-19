import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import noisereduce as nr

# ---------------------------------------------------------
# 1. Load the two speech samples
# ---------------------------------------------------------
Fs1, data1 = wav.read('AirportAnnouncements_10.wav')
Fs2, data2 = wav.read('CafeTeria_1.wav')

# Convert to float
data1 = data1.astype(float)
data2 = data2.astype(float)

# ---------------------------------------------------------
# 2. Apply noise reduction using noisereduce
# ---------------------------------------------------------
data1_nr = nr.reduce_noise(y=data1, sr=Fs1)
data2_nr = nr.reduce_noise(y=data2, sr=Fs2)

# ---------------------------------------------------------
# 3. Extract sample window 300–500
# ---------------------------------------------------------
clip1 = data1[300:500]
clip1_nr = data1_nr[300:500]

clip2 = data2[300:500]
clip2_nr = data2_nr[300:500]

# ---------------------------------------------------------
# 4. Plot spectrograms
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

def plot_spec(ax, data, fs, title):
    ax.specgram(data, NFFT=64, Fs=fs, noverlap=32, cmap="inferno")
    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")

plot_spec(axes[0, 0], clip1, Fs1, "Original Airport (300–500)")
plot_spec(axes[0, 1], clip1_nr, Fs1, "Noise-Reduced Airport (300–500)")

plot_spec(axes[1, 0], clip2, Fs2, "Original Cafeteria (300–500)")
plot_spec(axes[1, 1], clip2_nr, Fs2, "Noise-Reduced Cafeteria (300–500)")

plt.tight_layout()
plt.show()
