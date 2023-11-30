from src.methods import MUSIC
from src.signal import Signal
from utils.functions import plot_angles_on_unit_circle
if __name__ == '__main__':
    T = 500
    S = 2
    M = 8
    wavelength = 1
    array_geometry = 'ULA'
    snr = 20
    theta = [-10, 50]

    x = Signal(num_samples=T, num_sources=S, num_sensors=M, wavelength=wavelength, array_geometry=array_geometry)
    samples = x.generate(snr=snr, angles=theta)
    music = MUSIC(array_geometry=array_geometry, num_sensors=M, wavelength=wavelength, num_sources=S)
    predictions = music.compute_predictions(samples)
    print(predictions)
    plot_angles_on_unit_circle(theta, predictions)

