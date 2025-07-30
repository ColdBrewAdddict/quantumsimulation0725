import numpy as np
import matplotlib.pyplot as plt

#constants
acquisition_time = 0.01  #10ms
repetition_rate = 5_000_000 #5Mhz
pulses = int(acquisition_time * repetition_rate) #Total laser pulses

#signal photon
subject_reflectivity = 0.07   #human skin
expected_photons = np.random.binomial(n=pulses, p=subject_reflectivity)
true_TOF = (2*1.5)/3e8 #1.5 meter; light speed = 3e8; round trip
jitter = 0.3e-9 #timing uncertainty of SPAD (in Gaussian)
raw_depth = true_TOF + np.random.normal(0, jitter, size=expected_photons) #full array of photon arrival times

acquisition_window_ns = acquisition_time * 1e9     # Total detection window in ns
quantum_efficiency = 0.7
DCR = 20000

detected_signal = np.random.rand(len(raw_depth)) < quantum_efficiency
signal_arrival = raw_depth[detected_signal]
b_s_ratio = np.random.uniform(60, 80) #60-80 : 1
dark_counts = int(DCR * acquisition_time) #expected in the situation
background_noise = int(b_s_ratio * len(signal_arrival))
background_arrivals = np.random.uniform(0, acquisition_time*1e9, background_noise)
background_mask = np.random.rand(background_noise) < quantum_efficiency
background_arrivals = background_arrivals[background_mask]
dark_arrivals = np.random.uniform(0, acquisition_time*1e9, dark_counts)
dark_mask = np.random.rand(dark_counts) < quantum_efficiency
dark_arrivals = dark_arrivals[dark_mask]

all_arrivals = np.concatenate([signal_arrival*1e9, background_arrivals, dark_arrivals])

#estimation of TOF
def tof_estimation(TOF, jitter_mean, tau_laser_mean, laser_pulse_duration, tau_amb_mean, measurement_window_start, measurement_window_end, seed=None):
    """
    Args:
        jitter_mean (float): The mean value for the Gauss distribution, acting as the standard deviation
                                   for laser timing jitter in sec.
        tau_laser_mean (float): The expected average interval between laser photons within a pulse in seconds.
                                This is the 'mean' for the exponential distribution.
        laser_pulse_duration (float): The duration of the laser pulse in seconds.
        tau_amb_mean (float): The expected average interval between ambient photons in seconds.
                              This is the 'mean' for the random distribution of ambient photons.
        measurement_window_start (float): The start time of the TOF measurement window in seconds.
        measurement_window_end (float): The end time of the TOF measurement window in seconds.
        seed (int, optional): Seed for the random number generator for reproducibility. Defaults to None.

    Returns:
        A tuple containing:
            - ideal_first_photon_arrival_time (float): The theoretical first photon arrival time without jitter (in seconds).
            - first_laser_photon_arrival_time_with_jitter (float): The simulated arrival time of the first
                                                                   laser photon, including jitter (in seconds).
            - simulated_laser_photon_arrival_times (list): Sorted list of all simulated laser photon arrival times (in seconds).
            - simulated_ambient_photon_arrival_times (list): Sorted list of all simulated ambient photon arrival times (in seconds).
    """
    if seed is not None:
        np.random.seed(seed)

    ideal_first_photon_arrival_time = TOF

    # "$rdist_normal (seed, 0, mean)" implies a Gaussian distribution with mean 0 and standard deviation 'mean'.
    laser_jitter_effect = np.random.normal(loc=0, scale=jitter_mean)
    first_laser_photon_arrival_time_with_jitter = ideal_first_photon_arrival_time + laser_jitter_effect

    simulated_laser_photon_arrival_times = [first_laser_photon_arrival_time_with_jitter]

    # 3. Generate arrival times for the series of subsequent laser photons
    # These arrive within the specified laser pulse duration, starting from the jittered first photon.
    # "$rdist_exponential (seed, mean)" implies an exponential distribution with 'mean' as its scale parameter.
    current_laser_photon_time = first_laser_photon_arrival_time_with_jitter
    while True:
        # Generate the next interval using an exponential distribution
        interval = np.random.exponential(scale=tau_laser_mean)
        current_laser_photon_time += interval

        # Check if the new photon arrives within the laser pulse duration relative to the first photon
        if current_laser_photon_time <= (first_laser_photon_arrival_time_with_jitter + laser_pulse_duration):
            simulated_laser_photon_arrival_times.append(current_laser_photon_time)
        else:
            break
    simulated_laser_photon_arrival_times.sort() # Ensure the list is sorted by arrival time

    # 4. Generate ambient photon arrival times
    simulated_ambient_photon_arrival_times = []
    # "$random (seed) with the expected average interval τAmb as a mean value".
    # An exponential distribution is suitable for generating random events with an average interval.
    current_ambient_time = measurement_window_start
    while current_ambient_time < measurement_window_end:
        interval_ambient = np.random.exponential(scale=tau_amb_mean)
        current_ambient_time += interval_ambient

        if measurement_window_start <= current_ambient_time <= measurement_window_end:
            simulated_ambient_photon_arrival_times.append(current_ambient_time)
    simulated_ambient_photon_arrival_times.sort() # Ensure the list is sorted by arrival time

    return (ideal_first_photon_arrival_time,
            first_laser_photon_arrival_time_with_jitter,
            simulated_laser_photon_arrival_times,
            simulated_ambient_photon_arrival_times)


signal_arrival_sorted = np.sort(signal_arrival) #ensure in time order
signal_intervals = np.diff(signal_arrival_sorted) * 1e9
laser_pulse_model_duration = 50e-9  # Example: 50 ns duration for the simulated laser pulse train
background_sorted = np.sort(background_arrivals)
background_intervals = np.diff(background_sorted) * 1e9

measurement_window_start_s = 0.0
measurement_window_end_s = acquisition_time

# Run the simulation based on the described method
(ideal_tof_model,
 first_laser_photon_tof_model,
 simulated_laser_photons_model,
 simulated_ambient_photons_model) = tof_estimation(
    true_TOF,
    jitter,
    signal_intervals,
    laser_pulse_model_duration,
    background_intervals,
    measurement_window_start_s,
    measurement_window_end_s,
    seed=123 # Different seed for this part to differentiate from the initial setup
)

print("\n--- TOF Estimation ---")
print(f"Ideal TOF (based on model distance): {ideal_tof_model * 1e9:.3f} ns")
print(f"First laser photon arrival (with jitter from model): {first_laser_photon_tof_model * 1e9:.3f} ns")
print(f"Number of simulated laser photons (model): {len(simulated_laser_photons_model)}")
print(f"Number of simulated ambient photons (model): {len(simulated_ambient_photons_model)}")

# 1. Histogram the arrival times to create a TOF histogram
bin_width = 0.1 # ns
bins = np.arange(0, acquisition_window_ns + bin_width, bin_width)
hist, bin_edges = np.histogram(all_arrivals, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 2. Find the peak (a simple method for TOF estimation)
# This assumes the signal peak is dominant enough to be easily identified.
estimated_tof_from_histogram = bin_centers[np.argmax(hist)]
percent_error = abs(estimated_tof_from_histogram - true_TOF*1e9) / (true_TOF*1e9)
print(f"\nEstimated TOF from all_arrivals histogram peak: {estimated_tof_from_histogram:.3f} ns")
print(f"True TOF from setup: {true_TOF * 1e9:.3f} ns")
print(f"Percent error: {percent_error:.3f}%")


# Print results
"""
print(f"B:S ratio: {b_s_ratio:.2f} : 1")
print(f"Signal photons: {len(signal_arrival)}")
print(f"Background photons: {len(background_arrivals)}")
print(f"Dark counts: {len(dark_arrivals)}")
"""

#histogram
plt.hist(all_arrivals, bins=100, range=(0, acquisition_window_ns), color='skyblue', edgecolor='black')
plt.axvline(true_TOF*1e9, color='red', linestyle='--', label='True TOF (~10ns)')
plt.title("Simulated Photon Arrival Times in High Noise Environment")
plt.xlabel("Arrival Time (ns)")
plt.ylabel("Photon Count")
plt.legend()
plt.tight_layout()
plt.show()



