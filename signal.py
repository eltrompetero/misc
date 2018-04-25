
import numpy as np
from scipy.signal import get_window,detrend,fftconvolve,convolve2d

def tfcohf(x,y,nfft,spec_win,sm_win,tstep,fs):
    """
    Ported from MATLAB function
    https://www.mathworks.com/matlabcentral/fileexchange/38537-time-frequency-coherency

    TFCOHF Time-frequency coherency
    Estimates the complex coherency coefficients using Fourier decomposition 
    of vector X and vector Y. The cross and auto spectra are smoothed with 
    identical smoothing windows (sm_win).
    
    Time-frequency coherency is computed by smoothing the cross- and autospectra
    using a smoothing kernel specficied by SM_WIN. The cross- and autospectra
    are estimated using Welch's periodogram method. Signals are dived into overlapping
    sections, each of which is detrended and windowed by the SPEC_WIN parameter,
    then zero padded to len NFFT. TSTEP defines the number of samples the
    window is slided forward. The spectra X, Y, and XY are estimated for each
    segment.Spectral coefficients are then smoothed for the estimation of 
    time-frequency coherency using identical smoothing windows.
    
    ARGUMENTS:
              x           --  signal 1 (vector)
              y           --  signal 2 (vector)
              nfft (int)  --  len of fft, zero-padded if spec_win has less
                              than n points
              spec_win    --  len of window in samples used for spectral
                              decomposition (Hamming window)
              sm_win      --  len of window used for smoothing the auto
                              and cross spectra (Gauss window)
              tstep       --  number of samples the window is slided forward
              fs          --  sample frequeny
    
    If len(sm_win)==1 it specifies the len of the smoothing window in 
    seconds. If len(sm_win)==2 sm_win(1) specifies the height of the kernel 
    in hertz and sm_win(2) the width of the kernel in seconds. Otherwise 
    sm_win specifies the actual smoothing kernel.
    
    OUTPUTS:
              C           --  complex valued time-frequency coherency
                              [N,M] matrix with N frequencies and M
                              time-points
              F           --  frequency vector
              T           --  time vector
    
    If no outputs are specified, time-frequency coherency is plotted.
    
    EXAMPLE:
    >>fs = 200 spec_win = fs nfft = fs*3 tstep = fs/5
    >>x1 = sin(2*pi*20*(1:fs*10)/fs) x2 = sin(2*pi*40*(1:fs*10)/fs)
    >>x = [x1,x1,x2]+randn(1,fs*30)/20 y = [x1,x2,x2]+randn(1,fs*30)/20
    >>sm_win = [3,2]
    >>tfcohf(x,y,nfft,spec_win,sm_win,tstep,fs)
      
    Time-frequency coherency between two signals sampled at 200 Hz of 30s 
    duration for which synchronization jumps from 20 to 40 Hz. Data is 
    decomposed using a 1s window and smoothed over an time-frequency area of 
    3Hz by 2s. 
    
    Please cite the following paper when using this code:
    Mehrkanoon S, Breakspear M, Daffertshofer A, Boonstra TW (2013). Non-
    identical smoothing operators for estimating time-frequency interdependence 
    in electrophysiological recordings. EURASIP Journal on Advances in Signal 
    Processing 2013, 2013:73. doi:10.1186/1687-6180-2013-73
    
    T.W. Boonstra and S. Mehrkanoon          9-October-2012
    Systems Neuroscience Group, UNSW, Australia.
    
    See also FFT CONV
    """
    if type(spec_win) is int:
        wl = spec_win
    else:
        wl = len(spec_win)

    # Zero-padding of signal
    x_new = np.zeros((len(x)+wl))
    y_new = np.zeros((len(x)+wl))
    x_new[wl//2:wl//2+len(x)] = x
    y_new[wl//2:wl//2+len(x)] = y
    
    # Compute Fourier coefficients
    if nfft%2:    # nfft odd
        select = list(range((nfft+1)//2))
    else:
        select = list(range(nfft//2+1))   # include DC AND Nyquist

    X = np.zeros((len(select),len(x)//tstep+1),dtype=np.complex)
    Y = np.zeros((len(select),len(x)//tstep+1),dtype=np.complex)

    if type(spec_win) is int:
       window = get_window('hamming',spec_win)
    else:
        window = spec_win
    index = np.arange(wl,dtype=int)
    for k in range(len(x)//tstep+1):
        # Not sure why this detrending is necessary.
        temp = np.fft.fft( detrend(x_new[index],type='constant')*window,nfft )
        X[:,k] = temp[select]
        temp = np.fft.fft( detrend(y_new[index],type='constant')*window,nfft )
        Y[:,k] = temp[select]
        
        index = index+tstep
    
    # compute cross and auto spectra
    XY = X * Y.conjugate()
    X = np.abs(X)**2
    Y = np.abs(Y)**2

    # smooth spectra using sm_win
    if len(sm_win) == 1:
        gwl = int(sm_win[0]*fs/tstep)
        window = get_window(('gaussian',(gwl-1)/2/2.5),gwl)
    elif len(sm_win) == 2:
        gwl1 = round(sm_win[1]*nfft/fs)
        gwl2 = round(sm_win[2]*fs/tstep)
        window = ( get_window(('gaussian',(gwl1-1)/2/2.5),gwl1)[:,None]*
                   get_window(('gaussian',(gwl2-1)/2/2.5),gwl2)[None,:] )
    else:
        window = sm_win
    window = window/window.sum()
    print(window) 
    if window.ndim==1:
        for f in range(X.shape[0]):
            X[f,:] = fftconvolve(X[f,:],window,'same')
            Y[f,:] = fftconvolve(Y[f,:],window,'same')
            XY[f,:] = fftconvolve(XY[f,:],window,'same')
    else:
        X = convolve2d(X,window,'same')
        Y = convolve2d(Y,window,'same')
        XY = convolve2d(XY,window,'same')
        
    # compute tfcoh
    Cxy = XY/np.sqrt(X*Y)

    return Cxy,np.array(select)*fs/nfft,np.arange(0,len(x),tstep)/fs

def csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
        detrend='constant', return_onesided=True, scaling='density', axis=-1):
    r"""
    Estimate the cross power spectral density, Pxy, using Welch's
    method.

    Mostly copied from scipy.signal.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    y : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` and `y` time series. Defaults
        to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows
        and required parameters. If `window` is array_like it will be
        used directly as the window and its length must be nperseg.
        Defaults to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Note that for complex
        data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross spectrum
        ('spectrum') where `Pxy` has units of V**2, if `x` and `y` are
        measured in V and `fs` is measured in Hz. Defaults to 'density'
    axis : int, optional
        Axis along which the CSD is computed for both inputs; the
        default is over the last axis (i.e. ``axis=-1``).

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxy : ndarray
        Cross spectral density or cross power spectrum of x,y.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method. [Equivalent to
           csd(x,x)]
    coherence: Magnitude squared coherence by Welch's method.

    Notes
    --------
    By convention, Pxy is computed with the conjugate FFT of X
    multiplied by the FFT of Y.

    If the input series differ in length, the shorter series will be
    zero-padded to match.

    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default 'hann' window an overlap
    of 50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] Rabiner, Lawrence R., and B. Gold. "Theory and Application of
           Digital Signal Processing" Prentice-Hall, pp. 414-419, 1975

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    Generate two test signals with some common features.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 20
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / fs
    >>> b, a = signal.butter(2, 0.25, 'low')
    >>> x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    >>> y = signal.lfilter(b, a, x)
    >>> x += amp*np.sin(2*np.pi*freq*time)
    >>> y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)

    Compute and plot the magnitude of the cross spectral density.

    >>> f, Pxy = signal.csd(x, y, fs, nperseg=1024)
    >>> plt.semilogy(f, np.abs(Pxy))
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('CSD [V**2/Hz]')
    >>> plt.show()
    """

    freqs, _, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft,
                                     detrend, return_onesided, scaling, axis,
                                     mode='psd')

    # Average over windows.
    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-1] > 1:
            Pxy = Pxy.mean(axis=-1)
        else:
            Pxy = np.reshape(Pxy, Pxy.shape[:-1])

    return freqs, Pxy


