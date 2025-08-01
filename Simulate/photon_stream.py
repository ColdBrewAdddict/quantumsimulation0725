import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import csv
import warnings

#constants
acquisition_time = 0.01  #10ms
repetition_rate = 5_000_000 #5Mhz
pulses = int(acquisition_time * repetition_rate) #Total laser pulses
jitter = 0.3e-9  #timing uncertainty of SPAD (in Gaussian); μ = tof
acquisition_window_ns = acquisition_time * 1e9
DCR = 20000



def gaussian(x, a, mu, sigma):
    return a * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


def arrival_estimation(QE1, tof, SR, dark_arrival_ns, tau_laser_mean, laser_pulse_duration, tau_amb_mean, window_start, window_end, seed=None,
                       bin_width=0.1, fit_bins=7):
    if seed is not None:
        np.random.seed(seed)

    expected_photons = np.random.binomial(n=pulses, p=SR)
    raw_depth = tof + np.random.normal(0, jitter, size=expected_photons)  # full array of photon arrival times
    detected_signal = np.random.rand(len(raw_depth)) < QE1
    signal_arrival = raw_depth[detected_signal]

    # First laser photon with Gaussian jitter
    if len(signal_arrival) == 0:
        return (np.nan, np.nan, [], [])

    first_laser_arrival = tof + np.random.normal(0, jitter)
    laser_arrivals = [first_laser_arrival]

    # Step 2: Generate additional laser photons within pulse duration using exponential intervals
    current_time = first_laser_arrival
    pulse_end_time = first_laser_arrival + laser_pulse_duration

    while True:
        # Generate next photon with exponential interval
        interval = np.random.exponential(tau_laser_mean)
        next_arrival = current_time + interval

        # Stop if we exceed the pulse duration
        if next_arrival > pulse_end_time:
            break

        laser_arrivals.append(next_arrival)
        current_time = next_arrival

    laser_arrivals = np.array(laser_arrivals)
        
    # Step 3: Generate ambient photons across entire measurement window
    ambient_arrivals = []
    current_time = window_start
    
    while current_time < window_end:
        # Generate next ambient photon with exponential interval
        interval = np.random.exponential(tau_amb_mean)
        next_arrival = current_time + interval
        
        if next_arrival > window_end:
            break
            
        ambient_arrivals.append(next_arrival)
        current_time = next_arrival
    
    ambient_arrivals = np.array(ambient_arrivals)
    
    # Step 4: Apply quantum efficiency to laser and ambient photons
    laser_detected_mask = np.random.rand(len(laser_arrivals)) < QE1
    laser_detected = laser_arrivals[laser_detected_mask]
    
    if len(ambient_arrivals) > 0:
        ambient_detected_mask = np.random.rand(len(ambient_arrivals)) < QE1
        ambient_detected = ambient_arrivals[ambient_detected_mask]
    else:
        ambient_detected = np.array([])
    
    # Step 5: Apply quantum efficiency to dark counts (already in ns)
    dark_arrivals_s = dark_arrivals / 1e9  # Convert to seconds
    if len(dark_arrivals_s) > 0:
        dark_detected_mask = np.random.rand(len(dark_arrivals_s)) < QE1
        dark_detected = dark_arrivals_s[dark_detected_mask]
    else:
        dark_detected = np.array([])
    
    # Step 6: Combine all detected photons and convert to nanoseconds
    all_detected = []
    if len(laser_detected) > 0:
        all_detected.append(laser_detected)
    if len(ambient_detected) > 0:
        all_detected.append(ambient_detected)
    if len(dark_detected) > 0:
        all_detected.append(dark_detected)
    
    if len(all_detected) == 0:
        return (np.nan, np.nan, [], [], [])
    
    all_arrivals_s = np.concatenate(all_detected)
    all_arrivals_ns = all_arrivals_s * 1e9

    bins = np.arange(0, (window_end * 1e9) + bin_width, bin_width)
    hist, bin_edges = np.histogram(all_arrivals, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit Gaussian to peak region
    if not np.any(hist):
        estimated_tof = np.nan  # No photons detected
    else:
        peak_index = np.argmax(hist)
        half_width = fit_bins // 2
        fit_range = slice(max(0, peak_index - half_width), min(len(hist), peak_index + half_width + 1))
        x_fit = bin_centers[fit_range]
        y_fit = hist[fit_range]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category= OptimizeWarning)
                p0 = [np.max(y_fit), bin_centers[peak_index], bin_width * 2]
                popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
                estimated_tof = popt[1]  # μ (mean) is the TOF estimate
        except (RuntimeError, ValueError):
            estimated_tof = bin_centers[peak_index]  # Fallback to histogram peak

    return (estimated_tof, first_laser_arrival * 1e9, ambient_arrivals)

#csv file
with open("tof_trials_results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Trial",
        "Distance (m)",
        "Reflectivity",
        "B:S Ratio",
        "QE",
        "Signal Photons",
        "Background Photons",
        "Dark Counts",
        "First laser photon arrival (ns)",
        "Number of simulated ambient photons",
        "Est. TOF (ns)",
        "True TOF (ns)",
        "Percent Error (%)"
    ])



    for trial in range(1, 31):
        subject_reflectivity = np.random.uniform(0.05, 0.75)
        distance = np.random.uniform(1, 5)
        b_s_ratio = np.random.uniform(60, 80) #60-80 : 1
        quantum_efficiency = np.random.uniform(0.2, 0.7)

        expected_photons = np.random.binomial(n=pulses, p=subject_reflectivity)
        true_TOF = (2*distance)/3e8
        raw_depth = true_TOF + np.random.normal(0, jitter, size=expected_photons)  # full array of photon arrival times

        detected_signal = np.random.rand(len(raw_depth)) < quantum_efficiency
        signal_arrival = raw_depth[detected_signal]

        background_noise = int(b_s_ratio * len(signal_arrival))
        background_arrivals = np.random.uniform(0, acquisition_time*1e9, background_noise)
        background_mask = np.random.rand(background_noise) < quantum_efficiency
        background_arrivals = background_arrivals[background_mask]
        dark_counts = int(DCR * acquisition_time) #expected in the situation
        dark_arrivals = np.random.uniform(0, acquisition_time * 1e9, dark_counts)
        dark_mask = np.random.rand(dark_counts) < quantum_efficiency
        dark_arrivals = dark_arrivals[dark_mask]

        all_arrivals = np.concatenate([signal_arrival*1e9, background_arrivals, dark_arrivals])


        #Calculation of time duration
        signal_arrival_sorted = np.sort(signal_arrival) #ensure in time order
        signal_intervals = np.diff(signal_arrival_sorted) * 1e9
        avg_signal_interval = np.mean(signal_intervals)
        background_sorted = np.sort(background_arrivals)
        background_intervals = np.diff(background_sorted)
        avg_background_interval = (acquisition_time) / len(background_arrivals)

        min_arrival = signal_arrival_sorted[0]
        max_arrival = signal_arrival_sorted[-1]
        pulse_duration = (max_arrival - min_arrival)  # in seconds

        measurement_window_start_s = 0.0
        measurement_window_end_s = acquisition_time

        # Run simulation
        (estimated_tof,
         first_laser_photon,
         simulated_ambient_photons) = arrival_estimation(
         quantum_efficiency,
         true_TOF,
         subject_reflectivity,
         dark_arrivals,
         avg_signal_interval,
         pulse_duration,
         avg_background_interval,
         measurement_window_start_s,
         measurement_window_end_s,
         seed=123 # Different seed for this part to differentiate from the initial setup
        )

        percent_error = (abs(estimated_tof - (true_TOF*1e9))/(true_TOF*1e9))*100

        writer.writerow([
                trial,
                f"{distance:.3f}",
                f"{subject_reflectivity:.3f}",
                f"{b_s_ratio:.1f}",
                f"{quantum_efficiency:.3f}",
                len(signal_arrival),
                len(background_arrivals),
                len(dark_arrivals),
                f"{first_laser_photon :.3f}",
                len(simulated_ambient_photons),
                f"{estimated_tof:.3f}",
                f"{true_TOF * 1e9:.3f}",
                f"{percent_error:.3f}"
            ])


        #histogram
        plt.figure(figsize=(8, 5))
        plt.hist(all_arrivals, bins=100, range=(0, acquisition_window_ns), color='skyblue', edgecolor='black')
        plt.axvline(true_TOF*1e9, color='red', linestyle='--', linewidth=2, label='True TOF (~10ns)')
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



