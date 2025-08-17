import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.constants import h, c
import csv
import warnings

#REFERENCE: https://www.researchgate.net/publication/330429i600_Behavioral_Modeling_of_Photon_Arrival_Time_for_Time-of-Flight_Measurement_Circuit_Simulation

#constants
acquisition_time = 10e-3  #10ms
repetition_rate = 5e6 #5Mhz
pump_power = 0.29*1e-3  #0.29Mhz
pulses = int(acquisition_time * repetition_rate) #Total laser pulses
jitter = 0.3e-9  #timing uncertainty of SPAD (in Gaussian); μ += tof
acquisition_window_ns = acquisition_time * 1e9
DCR = 20000
detector_eff = 0.45
lambda_p = 940e-9 #wavelength
h
def photon_per_pulse():
    count = (pump_power*lambda_p)/(h*c*repetition_rate)
    return count

def gaussian(x, a, mu, sigma):
    return a * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


def arrival_estimation(QE1, tof, SR, tau_amb_mean, window_start, window_end,  DE, seed=None, bin_width=0.1, fit_bins=7):
    if seed is not None:
        np.random.seed(seed)

    count = photon_per_pulse()

    #everything is in second
    total_eff = QE1 * DE
    lambda_sig = total_eff * SR * count * pulses  #counts per sec [poisson]

    window_duration = window_end - window_start
    lambda_dark = DCR * window_duration

    num_signal = np.random.poisson(lambda_sig)
    num_dark = np.random.poisson(lambda_dark)

    signal_arrivals = tof + np.random.normal(0, jitter, size=num_signal) #array

    ambient_arrivals = []
    current_time = window_start
    while current_time < window_end:
        interval = np.random.exponential(tau_amb_mean)
        next_arrival = current_time + interval
        if next_arrival > window_end:
            break
        ambient_arrivals.append(next_arrival)
        current_time = next_arrival

    dark_arrivals = np.random.uniform(window_start, window_end, size=num_dark)

        # Combine & sort
    all_arrivals = np.concatenate([signal_arrivals, ambient_arrivals, dark_arrivals])
    all_arrivals.sort()  # stable sort for tagged events if needed
    all_arrivals_ns = all_arrivals*1e9

    bins = np.arange(0, (window_end * 1e9) + bin_width, bin_width)
    hist, bin_edges = np.histogram(all_arrivals_ns, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit Gaussian to peak region
    if not np.any(hist):
        estimated_tof = np.nan
    else:
        peak_index = np.argmax(hist)
        half_width = fit_bins // 2
        fit_range = slice(max(0, peak_index - half_width), min(len(hist), peak_index + half_width + 1))
        x_fit = bin_centers[fit_range]
        y_fit = hist[fit_range]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=OptimizeWarning)
                p0 = [np.max(y_fit), bin_centers[peak_index], bin_width * 2]
                popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
                estimated_tof = popt[1]
        except (RuntimeError, ValueError):
            estimated_tof = bin_centers[peak_index]

    return estimated_tof, num_signal, num_dark, all_arrivals_ns
#csv file
with open("tof_trials_results0.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Trial",
        "Distance (m)",
        "Reflectivity",
        "B:S Ratio",
        "QE",
        "Signal Photons",
        "Background Photons",
        "Number of simulated ambient photons",
        "Est. TOF (ns)",
        "True TOF (ns)",
        "Percent Error (%)"
    ])



    for trial in range(1, 31):
        subject_reflectivity = np.random.uniform(0.05, 0.75)
        distance = np.random.uniform(1, 5)
        n_s = np.random.uniform(60, 80) #60-80 : 1
        quantum_eff = np.random.uniform(0.2, 0.7)

        window_start_s = 0.0
        window_end_s = acquisition_time

        count = photon_per_pulse()
        total_eff = quantum_eff * detector_eff
        lambda_sig = total_eff * subject_reflectivity * count * pulses  # counts per sec [poisson]
        signal_rate = lambda_sig / acquisition_time
        background_rate = n_s * signal_rate
        tau_amb_mean = 1.0 / background_rate

        true_TOF = (2*distance)/3e8

        # Run simulation
        (estimated_tof,
         num_signal,
         num_dark,
         all_arrivals_ns) = arrival_estimation(
         quantum_eff,
         true_TOF,
         subject_reflectivity,
         tau_amb_mean,
         window_start_s,
         window_end_s,
         detector_eff,
         seed=123 # Different seed for this part to differentiate from the initial setup
        )

        N_background = num_signal * n_s

        percent_error = (abs(estimated_tof - (true_TOF*1e9))/(true_TOF*1e9))*100

        writer.writerow([
                trial,
                f"{distance:.3f}",
                f"{subject_reflectivity:.3f}",
                f"{n_s:.1f}",
                f"{quantum_eff:.3f}",
                int(num_signal),
                int(N_background),
                int(num_dark),
                f"{estimated_tof:.3f}",
                f"{true_TOF * 1e9:.3f}",
                f"{percent_error:.3f}%"
            ])

        #histogram
        bin_width = 0.5  # ns
        bin_edges = np.arange(0, acquisition_window_ns + bin_width, bin_width)

        plt.figure(figsize=(8, 5))
        plt.hist(all_arrivals_ns, bins=bin_edges, range=(0, acquisition_window_ns), color='skyblue', edgecolor='black')
        plt.axvline(true_TOF*1e9, color='red', linestyle='--', linewidth=2, label=f'True TOF ({true_TOF*1e9:.3f}ns)')
        if not np.isnan(estimated_tof):
            plt.axvline(estimated_tof, color = 'green', linestyle=':', linewidth=2, label=f'Estimated TOF ({estimated_tof:.1f}ns)')
        plt.title("Simulated Photon Arrival Times in High Noise Environment")
        plt.xlabel("Arrival Time (ns)")
        plt.ylabel("Photon Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"histogram_trial_{trial}.png", dpi=300)
        plt.close()


print("Simulation complete!")



