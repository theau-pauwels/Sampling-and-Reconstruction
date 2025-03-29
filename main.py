import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sc
from scipy.io import wavfile
import io

#%% Definitions of functions
# Generate signals
def generate_signal(sig_type, f_signal, duration, f_e):
    t = np.linspace(0, duration, int(f_e * duration))
    if sig_type == "Sinus":
        y = np.sin(2 * np.pi * f_signal * t)
    elif sig_type == "Squared":
        y = sc.square(2 * np.pi * f_signal * t)
    elif sig_type == "Sawtooth":
        y = sc.sawtooth(2 * np.pi * f_signal * t)
    elif sig_type == "Triangle":
        y = sc.sawtooth(2 * np.pi * f_signal * t, width=0.5)
    return t, y

def reconstruct_shannon_nyquist(y_sampled, f_e, t_reconstructed):
    """
    Reconstructs a discrete (sampled) signal into a continuous signal using the Shannon-Nyquist theorem.

    Args:
        y_sampled (numpy.ndarray): A NumPy array containing the sampled values of the signal.
        f_e (float): The sampling frequency (in Hz).
        t_reconstructed (numpy.ndarray): A NumPy array containing the times at which the signal should be reconstructed.

    Returns:
        numpy.ndarray: A NumPy array containing the values of the reconstructed signal.
    """

    # 1. Calculate the sampling period (T_e)
    # The sampling period is the inverse of the sampling frequency.
    T_e = 1 / f_e

    # 2. Create the sampling indices (n)
    # We create an array of indices from 0 to the length of the samples array.
    # These indices correspond to the positions of the samples in time.
    n = np.arange(len(y_sampled))

    # 3. Construct the sinc matrix
    # This step is crucial for applying the Shannon-Nyquist theorem.
    # We calculate a matrix where each element is the sinc function evaluated at (t - n*T_e) / T_e.
    # - t_reconstructed / T_e : Calculates the reconstruction time relative to the sampling period.
    # - n[:, None] : Transforms the indices array 'n' into a column matrix, allowing subtraction with t_reconstructed / T_e.
    # - np.sinc(...) : Applies the sinc function (sin(pi*x) / (pi*x)) element-wise.
    sinc_matrix = np.sinc(t_reconstructed / T_e - n[:, None])

    # 4. Weighted sum for reconstruction
    # For each time 't' in t_reconstructed, we calculate the sum of the samples weighted by the sinc function.
    # - y_sampled[:, None] : Transforms the samples array into a column matrix.
    # - y_sampled[:, None] * sinc_matrix : Multiplies each sample by its contribution (the corresponding row in the sinc matrix).
    # - np.sum(..., axis=0) : Sums the contributions of all samples for each time 't'.
    y_reconstructed = np.sum(y_sampled[:, None] * sinc_matrix, axis=0)

    # 5. Return the reconstructed signal
    # We return the array containing the values of the reconstructed signal.
    return y_reconstructed

def reconstruct_sample_and_hold(y_sampled, f_e, t_reconstructed):
    """
    Reconstructs a sampled signal using a simple sample-and-hold method.

    Args:
        y_sampled (numpy.ndarray): A NumPy array containing the sampled values of the signal.
        f_e (float): The sampling frequency (in Hz).
        t_reconstructed (numpy.ndarray): A NumPy array containing the times at which the signal should be reconstructed.

    Returns:
        numpy.ndarray: A NumPy array containing the values of the reconstructed signal.
    """

    # 1. Calculate sample indices
    # For each time in t_reconstructed, we determine the index of the nearest sample in y_sampled.
    # - t_reconstructed * f_e : Calculates the sample index (not necessarily an integer) for each time.
    # - np.floor(...) : Rounds down to the nearest integer, giving the index of the sample to use.
    # - .astype(int) : Converts the indices to integers.
    sample_indices = np.floor(t_reconstructed * f_e).astype(int)

    # 2. Clip sample indices
    # We ensure that the calculated sample indices are within the valid range of indices for y_sampled.
    # - np.clip(..., 0, len(y_sampled) - 1) : Clips the indices to be between 0 and the last valid index of y_sampled.
    sample_indices = np.clip(sample_indices, 0, len(y_sampled) - 1)

    # 3. Return reconstructed signal
    # We create the reconstructed signal by using the sample values from y_sampled at the calculated indices.
    # - y_sampled[sample_indices] : Retrieves the sample values corresponding to the calculated indices.
    return y_sampled[sample_indices]

def apply_anti_aliasing_filter(signal, f_e_original, cutoff_f_e, filter_order=5):
    """
    Applies a Butterworth low-pass filter to a signal to prevent aliasing.

    Args:
        signal (numpy.ndarray): The input signal to be filtered.
        f_e_original (float): The original sampling frequency of the signal (in Hz).
        cutoff_f_e (float): The cutoff frequency of the filter (in Hz).
        filter_order (int, optional): The order of the Butterworth filter. Defaults to 5.

    Returns:
        numpy.ndarray: The filtered signal.
    """

    # 1. Normalize the cutoff frequency
    # We need to normalize the cutoff frequency to be between 0 and 1, where 1 corresponds to the Nyquist frequency (f_e_original / 2).
    # - cutoff_f_e : The desired cutoff frequency.
    # - f_e_original / 2 : The Nyquist frequency, which is half the original sampling frequency.
    normal_cutoff = cutoff_f_e / (f_e_original / 2)

    # 2. Calculate Butterworth filter coefficients
    # We use the scipy.signal.butter function to calculate the coefficients of a Butterworth low-pass filter.
    # - filter_order : The order of the filter, which determines its steepness.
    # - normal_cutoff : The normalized cutoff frequency.
    # - btype='low' : Specifies a low-pass filter.
    # - analog=False : Specifies a digital filter.
    # The function returns two arrays, 'b' and 'a', which are the numerator and denominator coefficients of the filter.
    b, a = sc.butter(filter_order, normal_cutoff, btype='low', analog=False)

    # 3. Apply the filter to the signal
    # We use the scipy.signal.filtfilt function to apply the filter to the input signal.
    # - b, a : The filter coefficients calculated in the previous step.
    # - signal : The input signal to be filtered.
    # filtfilt applies the filter twice, once forward and once backward, to eliminate phase distortion.
    filtered_signal = sc.filtfilt(b, a, signal)

    # 4. Return the filtered signal
    # We return the filtered signal.
    return filtered_signal
#%% Printings on streamlit

st.set_page_config(layout="wide")                                              # puts streamlit in widescreen by default
st.title("Sampling and Reconstruction")

# Section 1: Sampling and reconstruction
st.markdown("""
         This Streamlit application interactively demonstrates audio signal sampling and reconstruction, showcasing the effects of sampling rate and reconstruction methods. Users can generate various signals, adjust sampling parameters, and compare reconstructions using Shannon-Nyquist sinc interpolation (theoretically accurate) and sample-and-hold (a simpler, less precise method). The app also features anti-aliasing filtering, visualizes time-domain and frequency spectra, and enables audio playback, providing an educational tool to understand the practical implications of signal sampling and the Shannon-Nyquist theorem.
""")
st.latex(r"s(t) = \sum_{n=-\infty}^{+\infty} s(n T_e) \operatorname{sinc} \left( \frac{t - n T_e}{T_e} \right)")

celParam, celSignal = st.columns(2)
        
##### Affichage du tableau sur streamlit

celA, celB = st.columns(2)

#%% Definition of the parameters
duration = 0.5
with celParam:
    sig_type = st.selectbox("Type of signal", ["Sinus", "Squared", "Sawtooth", "Triangle"]) #, "Music"
    
    # Sampling parameters
    f_signal = st.slider("Frequency of the signal (Hz)", 20, 500, 240)
    f_e = st.slider("Sampling rate (Hz)", 10, 4410, 1500)
    f_e_original = 44100                                                       # CD sampling rate to avoid distorsion
    cutoff_f_e = st.slider("Cutoff frequency of Butterworth filter (Hz)", 1, 4410, 480)   # cutoff_f_e of Butterworth filter


#%% Signal processing

### base signal
# cel1 - generate base signal and its sampling data
t_signal, y_signal = generate_signal(sig_type, f_signal, duration, f_e_original)
t_sampled, y_sampled = generate_signal(sig_type, f_signal, duration, f_e)

# cel2 - spectrum of base signal
fft_signal = np.fft.fft(y_signal)
freqs_signal = np.fft.fftfreq(len(y_signal), 1 / f_e_original)
amp_signal = np.abs(fft_signal)  

### Shannon-Nyquist
# cel3 - signal reconstrcution with Shannon-Nyquist theorem
t_reconstructed = np.linspace(0, duration, int(f_e_original * duration), endpoint=False)
y_reconstructed = reconstruct_shannon_nyquist(y_sampled, f_e, t_reconstructed)

# cel4 - spectrum of Shannon-Nyquist signal
fft_rec = np.fft.fft(y_reconstructed)
freqs_rec = np.fft.fftfreq(len(y_reconstructed), 1 / f_e_original)
amp_rec = np.abs(fft_rec)

### Filtrage du signal de Shannon-Nyquist
# cel9 - Filtrage du signal reconstruit à l'aide de Shannon-Nyquist
y_reconstructed2 = apply_anti_aliasing_filter(y_reconstructed, f_e_original, cutoff_f_e)

# cel10 - Calcul du spectre du signal filtré de Shannon-Nyquist
fft_rec2 = np.fft.fft(y_reconstructed2)
freqs_rec2 = np.fft.fftfreq(len(y_reconstructed2), 1 / f_e_original)
amp_rec2 = np.abs(fft_rec2)

### 'sample & hold' signal
# cel5 - signal reconstruction with 'sample & hold'
y_simpler_reconstructed = reconstruct_sample_and_hold(y_sampled, f_e, t_reconstructed)

# cel6 - spectrum of 'sample & hold' signal
fft_simpler_rec = np.fft.fft(y_simpler_reconstructed)
freqs_simpler_rec = np.fft.fftfreq(len(y_simpler_reconstructed), 1 / f_e_original)
amp_simpler_rec = np.abs(fft_simpler_rec)

### filtered 'sample and hold' signal
# cel7 - filtration 'sample & hold' signal
y_simpler_reconstructed2 = apply_anti_aliasing_filter(y_simpler_reconstructed, f_e_original, cutoff_f_e)

# cel8 - spectrum of filtered 'sample & hold' signal
fft_simpler_rec2 = np.fft.fft(y_simpler_reconstructed2)
freqs_simpler_rec2 = np.fft.fftfreq(len(y_simpler_reconstructed2), 1 / f_e_original)
amp_simpler_rec2 = np.abs(fft_simpler_rec2)

#%% audios generation
        
# cel1 and cel2 - base signal
audio_signal_original = np.int16(y_signal * 32767)                             # convert to 16-bit integers
wav_file_original = io.BytesIO()
wavfile.write(wav_file_original, f_e_original, audio_signal_original)
wav_bytes_original = wav_file_original.getvalue()

# cel3 and cel4 - reconstructed signal with 'Shannon-Nyquist'
audio_signal_reconstructed = np.int16(y_reconstructed * 32767)                 # convert to 16-bit integers
wav_file_reconstructed = io.BytesIO()
wavfile.write(wav_file_reconstructed, f_e_original, audio_signal_reconstructed)
wav_bytes_reconstructed = wav_file_reconstructed.getvalue()
        
# cel9 - Signal reconstruit avec 'Shannon-Nyquist' filtré
audio_signal_reconstructed2 = np.int16(y_reconstructed2 * 32767)               # convert to 16-bit integers
wav_file_reconstructed2 = io.BytesIO()
wavfile.write(wav_file_reconstructed2, f_e_original, audio_signal_reconstructed2)
wav_bytes_reconstructed2 = wav_file_reconstructed2.getvalue()

# cel5 and cel6 - 'sample & hold' signal
audio_simpler_signal_reconstructed = np.int16(y_simpler_reconstructed * 32767) # convert to 16-bit integers
wav_file_simpler_reconstructed = io.BytesIO()
wavfile.write(wav_file_simpler_reconstructed, f_e_original, audio_simpler_signal_reconstructed)
wav_bytes_simpler_reconstructed = wav_file_simpler_reconstructed.getvalue()

# cel7 and cel8 - 'sample & hold' filtered signal
audio_simpler_signal_reconstructed2 = np.int16(y_simpler_reconstructed2 * 32767) # convert to 16-bit integers
wav_file_simpler_reconstructed2 = io.BytesIO()
wavfile.write(wav_file_simpler_reconstructed2, f_e_original, audio_simpler_signal_reconstructed2)
wav_bytes_simpler_reconstructed2 = wav_file_simpler_reconstructed2.getvalue()

#%% content of each cell of the table
with celSignal:
    # create two columns for the base signal and its fft
    st.subheader("Base signal and its fft")
    cel1, cel2 = st.columns(2)
    st.write("Listen the above signal :")
    st.audio(wav_bytes_original, format="audio/wav")
    
    with cel1:
        # plot the base signal
        fig1, ax1 = plt.subplots()
        ax1.set_xlim(10/f_signal, 15/f_signal)                                 # limit the plot to 5 periods of the signal, not plotting the first ones to avoid reconstruction artefacts
        ax1.stem(t_sampled, y_sampled, linefmt='b-', markerfmt='bo', basefmt=' ') # add sample points
        ax1.plot(t_signal, y_signal, color='orange', linestyle='-', label='original signal') # plot the signal
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid()
        st.pyplot(fig1)
    
    with cel2:
        # plot the spectrum of the base signal
        fig2, ax2 = plt.subplots()
        ax2.set_xlim(0, f_e)
        ax2.plot(freqs_signal[:len(freqs_signal)//2], 20*np.log(amp_signal[:len(freqs_signal)//2]), color='steelblue', linestyle='-')
        ax2.axvline(f_e/2, color='red', linestyle='--', label='f_e/2')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude (dB)")
        ax2.set_ylim(-75, 225)
        ax2.legend(loc='upper right')
        ax2.grid()
        st.pyplot(fig2)
        
with celA:
    # plotting Shannon-Nyquist reconstructed signal and its fft, cells 3 and 4
    st.subheader("Shannon-Nyquist reconstructed signal and its fft")
    cel3, cel4 = st.columns(2)
    st.write("Listen the above signal :")
    st.audio(wav_bytes_reconstructed, format="audio/wav")
    
    with cel3:
        # plot Shannon-Nyquist reconstructed signal
        fig3, ax3 = plt.subplots()
        ax3.set_xlim(10/f_signal, 15/f_signal)                                 # Print 5 periods of the signal, not the first ones to avoid reconstruction artefacts
        ax3.stem(t_sampled, y_sampled, linefmt='b-', markerfmt='bo', basefmt=' ')
        ax3.plot(t_reconstructed, y_reconstructed, color='steelblue', linestyle='-', label='reconstructed signal')
        ax3.plot(t_signal, y_signal, color='sandybrown', linestyle='--', label='original signal')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Amplitude")
        ax3.grid()
        st.pyplot(fig3)
    
    with cel4:
        # plot Shannon-Nyquist reconstructed signal fft
        fig4, ax4 = plt.subplots()
        ax4.set_xlim(0, f_e)
        ax4.plot(freqs_rec[:len(freqs_rec)//2], 20*np.log(amp_rec[:len(amp_rec)//2]))
        ax4.axvline(f_e/2, color='red', linestyle='--', label='f_e/2')
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Amplitude (dB)")
        ax4.set_ylim(-75, 225)
        ax4.legend(loc='upper right')
        ax4.grid()
        st.pyplot(fig4)
        
    # Affichage du signal et du spectre reconstruit, cellule 9 & 10
    st.subheader("Signal et spectre reconstruits avec Shannon-Nyquist")
    cel9, cel10 = st.columns(2)
    st.write("Listen the above signal :")
    st.audio(wav_bytes_reconstructed2, format="audio/wav")
    
    with cel9:
        #Affichage du signal reconstitué à partir des sinc()
        fig9, ax9 = plt.subplots()
        ax9.set_xlim(10/f_signal, 15/f_signal)                                 # Print 5 periods of the signal, not the first ones to avoid reconstruction artefacts
        ax9.stem(t_sampled, y_sampled, linefmt='b-', markerfmt='bo', basefmt=' ')
        ax9.plot(t_reconstructed, y_reconstructed2, color='steelblue', linestyle='-', label='reconstructed signal')
        ax9.plot(t_signal, y_signal, color='sandybrown', linestyle='--', label='original signal')
        ax9.set_xlabel("Time (s)")
        ax9.set_ylabel("Amplitude")
        ax9.grid()
        ax9.legend(loc='upper right')
        st.pyplot(fig9)
    
    with cel10:
        fig10, ax10 = plt.subplots()
        ax10.set_xlim(0, f_e)
        ax10.plot(freqs_rec2[:len(freqs_rec2)//2], 20*np.log(amp_rec2[:len(amp_rec2)//2]))
        ax10.axvline(f_e/2, color='red', linestyle='--', label='f_e/2')
        ax10.axvline(cutoff_f_e, color='blue', linestyle='--', label='cutoff_f_e')
        ax10.set_xlabel("Frequency (Hz)")
        ax10.set_ylabel("Amplitude (dB)")
        ax10.set_ylim(-75, 225)
        ax10.legend(loc='upper right')
        ax10.grid()
        st.pyplot(fig10)
        
with celB:
    # plotting 'sample & hold' reconstructed signal and its fft, cells 5 and 6
    st.subheader("'Sample & hold' reconstructed signal and its fft")
    cel5, cel6 = st.columns(2)
    st.write("Listen the above signal :")
    st.audio(wav_bytes_simpler_reconstructed, format="audio/wav")
        
    with cel5:
        # plot 'sample & hold' reconstructed signal
        fig5, ax5 = plt.subplots()
        ax5.set_xlim(10/f_signal, 15/f_signal)                                 # Print 5 periods of the signal, not the first ones to avoid reconstruction artefacts
        ax5.stem(t_sampled, y_sampled, linefmt='b-', markerfmt='bo', basefmt=' ')
        ax5.plot(t_reconstructed, y_simpler_reconstructed, color='steelblue', linestyle='-', label='reconstructed signal')
        ax5.plot(t_signal, y_signal, color='sandybrown', linestyle='--', label='original signal')
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Amplitude")
        ax5.grid()
        ax5.legend(loc='upper right')
        st.pyplot(fig5)
    
    with cel6:
        # plot 'sample & hold' reconstructed signal fft
        fig6, ax6 = plt.subplots()
        ax6.set_xlim(0, f_e)
        ax6.plot(freqs_simpler_rec[:len(freqs_simpler_rec)//2], 20*np.log(amp_simpler_rec[:len(amp_simpler_rec)//2]))
        ax6.axvline(f_e/2, color='red', linestyle='--', label='f_e/2')
        ax6.set_xlabel("Frequency (Hz)")
        ax6.set_ylabel("Amplitude (dB)")
        ax6.set_ylim(-75, 225)
        ax6.legend(loc='upper right')
        ax6.grid()
        st.pyplot(fig6)
        
    # plotting 'sample & hold' filtered reconstructed signal and its fft, cells 7 and 8
    st.subheader("'Sample & hold' filtered reconstructed signal and its fft")
    cel7, cel8 = st.columns(2)
    st.write("Listen the above signal :")
    st.audio(wav_bytes_simpler_reconstructed2, format="audio/wav")    
    
    with cel7:
        # plot 'sample & hold' filtered reconstructed signal
        fig7, ax7 = plt.subplots()
        ax7.set_xlim(10/f_signal, 15/f_signal)                                 # Print 5 periods of the signal, not the first ones to avoid reconstruction artefacts
        ax7.stem(t_sampled, y_sampled, linefmt='b-', markerfmt='bo', basefmt=' ')
        ax7.plot(t_reconstructed, y_simpler_reconstructed2, color='steelblue', linestyle='-', label='reconstructed signal')
        ax7.plot(t_signal, y_signal, color='sandybrown', linestyle='--', label='original signal')
        ax7.set_xlabel("Time (s)")
        ax7.set_ylabel("Amplitude")
        ax7.grid()
        ax7.legend(loc='upper right')
        st.pyplot(fig7)

    with cel8:
        # plot 'sample & hold' filtered reconstructed signal fft
        fig8, ax8 = plt.subplots()
        ax8.set_xlim(0, f_e)
        ax8.plot(freqs_simpler_rec2[:len(freqs_simpler_rec2)//2], 20*np.log(amp_simpler_rec2[:len(amp_simpler_rec2)//2]))
        ax8.axvline(f_e/2, color='red', linestyle='--', label='f_e/2')
        ax8.axvline(cutoff_f_e, color='blue', linestyle='--', label='cutoff_f_e')
        ax8.set_xlabel("Frequency (Hz)")
        ax8.set_ylabel("Amplitude (dB)")
        ax8.set_ylim(-75, 225)
        ax8.legend(loc='upper right')
        ax8.grid()
        st.pyplot(fig8)

#%% detailed explanations

with st.expander("More detailed explanation"):
    st.markdown("""
        **1. Sampling the signal:**
        - An analog signal is converted into a discrete sequence of values.
        - For the sampling to allow a faithful reconstruction of the original signal, the Shannon-Nyquist theorem requires that the useful signal is band-limited (i.e., it does not contain frequencies higher than a certain $f_{max}$) and that:
    """)
    st.latex(r"f_e > 2 \cdot f_{max}")
    st.markdown("""
        This condition prevents the phenomenon of **aliasing** (spectrum folding) which would lead to distortion during reconstruction.

        **2. Signal reconstruction by sinc interpolation, Shannon-Nyquist theorem:**
        - Once the signal is sampled, we want to recover an approximation of the original continuous signal.
        - The interpolation (or reconstruction) formula is given by:
    """)
    st.latex(r"s(t) = \sum_{n=-\infty}^{+\infty} s[n] \cdot \operatorname{sinc}\left(\frac{t - nT_e}{T_e}\right)")
    st.markdown(r"""
        where:
        - $s[n]$ is the signal value at time $nT_e$ (with $T_e = \frac{1}{f_e}$ the sampling period),
        - The **sinc** function is defined by $\operatorname{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$.
        - The sinc function has the property of being equal to 1 when $x = 0$ and 0 for all non-zero integers. This means that each sample contributes significantly to the reconstruction only around its sampling time.
        - By multiplying each sample by a shifted sinc function (centered on the sample time) and summing all these contributions, we theoretically reconstruct the initial continuous signal.
        - Since it requires to know all the signal data and its heavily intensive computational need, it can't be used in real time application.

        **3. Signal reconstruction by sample-and-hold:**
        - This is a simpler, but less accurate, method of reconstructing a signal.
        - Each sampled value is held constant until the next sample is taken, creating a series of steps.
        - This introduces a "staircase" effect, which can be heard as distortion, especially at higher frequencies.
        - In the code, this is implemented by assigning the value of the nearest sample to each point in the reconstructed time array.
        - The 'reconstruct_sample_and_hold' function finds the index of the nearest sample and uses that sample's value for the reconstructed signal.
        - This method is computationally less intensive than sinc interpolation but results in a less faithful reconstruction.
        - It can be used in real time application.

        **4. Technical details of the implementation:**
        - **Calculation of the sampling period ($T_e$)**:
        We calculate $T_e$ as the inverse of the sampling frequency, i.e., $T_e = 1 / f_e$. This gives us the time between two samples.
        - **Creation of sampling indices ($n$)**:
        The function `np.arange(len(y_sampled))` generates an array containing the indices (0, 1, 2, …) corresponding to each sample of the signal.
        - **Construction of the sinc matrix (for Shannon-Nyquist):**
        To reconstruct the signal at precise times (defined in `t_reconstructed`), we use the `np.sinc()` function.
        - The calculation `t_reconstructed / T_e - n[:, None]` creates a 2D matrix where each row corresponds to a sample index and each column to a reconstruction time.
        - The `np.sinc()` function is applied element by element, calculating $\operatorname{sinc}\left(\frac{t - nT_e}{T_e}\right)$ for each combination.
        - **Weighted sum for reconstruction (for Shannon-Nyquist):**
        By multiplying each sample by its contribution (its row in the sinc matrix) and summing over all indices, we obtain the reconstructed signal.
        - **Sample-and-hold reconstruction:**
        The 'reconstruct_sample_and_hold' function finds the nearest sample index for each point in the reconstructed time and uses that sample's value.

        This sinc interpolation mechanism is the theoretical basis that guarantees, in theory, that we can recover a continuous signal from its samples, provided that the signal is band-limited and the sampling frequency respects the Shannon-Nyquist condition. Sample and hold, although less accurate, is a common and computationally efficient reconstruction method.
    """)
