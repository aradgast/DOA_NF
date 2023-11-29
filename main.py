from src.methods import MUSIC
from src.signal import Signal

if __name__ == '__main__':
    T = 100
    S = 2
    M = 8
    wavelength = 1
    array_geometry = 'ULA'
    snr = 20
    theta = [15, 60]

    x = Signal(num_samples=T, num_sources=S, num_sensors=M, wavelength=wavelength, array_geometry=array_geometry)
    samples = x.generate(snr=snr, angles=theta)
    music = MUSIC(array_geometry=array_geometry, num_sensors=M, wavelength=wavelength, num_sources=S)
    predictions = music.compute(samples)

    print(predictions)
