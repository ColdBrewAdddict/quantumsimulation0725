import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, pi, c, epsilon_0
import csv

class HeraldGating:

    def __init__(self):
        self.acquisition_time = 0.01  # 10ms
        self.repetition_rate = 5e6  # 5pMhz
        self.pulses = int(self.acquisition_time * self.repetition_rate)  # Total laser pulses
        self.jitter = 0.3e-9  # timing uncertainty of SPAD (in Gaussian); μ = tof
        self.DCR = 20000

        self.L = 3e-3 #crystal length mm
        self.pump_power = 30e-3
        self.coe = 10e-12 #x^2
        self.pulse_duration = 1.5e-12 #when laser pulse hits crystal during SPDC
        self.n_o = 1.824441
        self.detector_eff = 0.45
        self.subject_reflectivity = np.random.uniform(0.05, 0.75)
        self.b_s = np.random.uniform(40, 60)  # 60-80 : 1
        self.w_o = 1e6
        self.prob_per_pulse = self.SPDC()

    def electric_field_amplitude(self):
        Ep = np.sqrt(2*self.pump_power/(epsilon_0 * c * pi * self.w_o**2*self.n_o))
        return Ep


    def SPDC(self):
        # Laser parameters
        lambda_pump = 404e-9  # 404 nm pump laser
        lambda_signal = 808e-9
        lambda_idler = 808e-9

        w_pump = 2 * pi * c / lambda_pump
        w_signal = 2 * pi * c / lambda_signal
        w_idler = 2 * pi * c / lambda_idler

        k_p = self.n_o * w_pump / c  #
        k_s = self.n_o * w_signal / c
        k_i = self.n_o * w_idler / c
        delta_k = k_p - k_s - k_i

        #Effective interaction length
        if abs(delta_k * self.L) < 1e-10:
            L_eff = self.L
        else:
            L_eff = np.sin(delta_k * self.L / 2) / (delta_k / 2)
        Ep = self.electric_field_amplitude()
        # Pair generation rate (photons per second)
        rate = (w_signal**2 * w_idler**2 * self.coe**2 * Ep**2 * self.L**4 * L_eff**2)/(pi**3 * epsilon_0**2 * c**5 * self.n_o)
        print(f"rate ={rate}")

        self.prob_per_pulse = rate * self.pulse_duration

        return self.prob_per_pulse

    def generate_spdc_pairs(self):
        spdc_success = np.random.rand(self.pulses) <= 0.002 #boolean array
        successful_indices = np.where(spdc_success)[0]
        pair_times = successful_indices / self.repetition_rate
        z = np.random.uniform(0, self.L, size=len(successful_indices))

        print(f"Generated {len(successful_indices)} SPDC pairs out of {self.pulses} pulses")
        return pair_times, z

    def idler_detection(self, pair_times, z, QE):
        total_eff = QE * self.detector_eff

        idler_detected = np.random.rand(len(pair_times)) <= total_eff #boolean array return [mask]
        idler_delay = ((self.L-z)*self.n_o)/c
        idler_times = np.full(len(pair_times), np.nan)
        idler_times[idler_detected] = (
                pair_times[idler_detected]
                + idler_delay[idler_detected]
                + np.random.normal(0.0, self.jitter, size=np.sum(idler_detected)))

        return idler_times, idler_detected

    def signal_detection(self, pair_times, z, tof, QE):
        total_eff = QE * self.detector_eff*self.subject_reflectivity

        signal_detected = np.random.random(len(pair_times)) < total_eff

        t2 = ((self.L - z) * self.n_o) / c
        t3 = tof  # only tof

        signal_times = np.full(len(pair_times), np.nan)
        signal_times[signal_detected] = (
                pair_times[signal_detected]
                + t2[signal_detected]
                + t3
                + np.random.normal(0.0, self.jitter, size=np.sum(signal_detected))
        )
        return signal_times, signal_detected

    def false_detection(self, pair_times):

        true_photon_rate = len(pair_times)/self.acquisition_time
        background_rate = self.b_s * true_photon_rate
        tau_amb_mean = 1.0 / background_rate

        # Actual counts sampled from Poisson distribution
        num_dark = np.random.poisson(self.DCR * self.acquisition_time)
        ambient_arrivals = []
        current_time = 0.0
        while current_time < self.acquisition_time:
            interval = np.random.exponential(tau_amb_mean)
            next_arrival = current_time + interval
            if next_arrival > self.acquisition_time:
                break
            ambient_arrivals.append(next_arrival)
            current_time = next_arrival

        ambient_arrivals = np.array(ambient_arrivals)

        # Uniform arrival times over the acquisition window
        dark_counts = np.random.uniform(0, self.acquisition_time, size=num_dark)
        # Merge and sort timestamps
        false_counts = np.sort(np.concatenate([ambient_arrivals, dark_counts]))
        return false_counts

    def timestamps(self, pair_times, z, tof, QE, p_idler=0.5, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # 1) Real detections
        idler_times, idler_detected = self.idler_detection(pair_times, z, QE)
        signal_times, signal_detected = self.signal_detection(pair_times, z, tof, QE)

        idler_timestamps = idler_times[idler_detected] #got rid of the NaNs
        signal_timestamps = signal_times[signal_detected]

        # 2) Background / false counts split into channels
        false_counts = self.false_detection(pair_times)  # 1D array of timestamps
        mask = rng.random(false_counts.size) < p_idler
        false_idler = np.sort(false_counts[mask])
        false_signal = np.sort(false_counts[~mask])

        # 3) Concatenate channels once
        all_idler_timestamps = np.concatenate((idler_timestamps, false_idler))
        all_signal_timestamps = np.concatenate((signal_timestamps, false_signal))

        # 4) Build labels in the SAME concatenation order as above
        idler_labels = np.concatenate((
            np.ones(len(idler_timestamps), dtype=bool),  # real
            np.zeros(len(false_idler), dtype=bool)  # background
        ))
        signal_labels = np.concatenate((
            np.ones(len(signal_timestamps), dtype=bool),
            np.zeros(len(false_signal), dtype=bool)
        ))

        return (all_idler_timestamps, all_signal_timestamps,
                idler_labels, signal_labels,
                idler_timestamps, signal_timestamps)

    def find_coincidences(self, pair_times, z_positions, tof, QE, window_center, window_width):
        """
        Return:
          true_coincidences:  dt = (signal_time - idler_time) for pairs where both photons were detected
          noisy_coincidences: dt from greedy one-to-one matches in noisy (true+false) streams
        """
        # --- True coincidences from per-pair alignment ---
        idler_times, idler_det = self.idler_detection(pair_times, z_positions, QE)
        signal_times, signal_det = self.signal_detection(pair_times, z_positions, tof, QE)

        both = idler_det & signal_det  # NaNs are already excluded by the masks
        true_coincidences = signal_times[both] - idler_times[both]

        # --- Noisy streams (sorted) ---
        (all_i, all_s, idler_labels, signal_labels, _, _) = self.timestamps(
            pair_times, z_positions, tof, QE
        )
        # all_i and all_s must be sorted; timestamps() should already ensure this.
        # Greedy one-to-one matching using binary search per idler
        delta = window_width * 0.5
        used_s = np.zeros(len(all_s), dtype=bool)
        noisy = []

        for ti in all_i:
            expected = ti + window_center
            left = np.searchsorted(all_s, expected - delta, side="left")
            right = np.searchsorted(all_s, expected + delta, side="right")
            if right <= left:
                continue
            # candidates in window that are not yet used
            cands = np.arange(left, right)
            cands = cands[~used_s[cands]]
            if cands.size == 0:
                continue
            # pick closest to expected
            k = cands[np.argmin(np.abs(all_s[cands] - expected))]
            noisy.append(all_s[k] - ti)
            used_s[k] = True

        return np.asarray(true_coincidences), np.asarray(noisy)

    def simulation(self):
        with open("herald_model_results.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Trial",
                "Distance (m)",
                "Reflectivity",
                "B:S Ratio",
                "QE",
                "True TOF (ns)",
                "True Coincidences",
                "All Coincidences",
                "Windowed Coincidences",
                "Signal Preservation Ratio"
            ])

            for trial in range(1, 31):
                pair_times, z = self.generate_spdc_pairs()

                distance = np.random.uniform(1, 5)  # crystal to target
                quantum_efficiency = np.random.uniform(0.2, 0.7)

                true_TOF = (2 * distance) / c
                (all_idler_timestamps, all_signal_timestamps,
                 idler_labels, signal_labels,
                 real_idler_timestamps, real_signal_timestamps) = self.timestamps(pair_times, z, true_TOF,
                                                                                  quantum_efficiency)

                # 1. TRUE SIGNALS ONLY - calculate delta_t from real detections
                delta_t_true = []
                for real_idler in real_idler_timestamps:
                    for real_signal in real_signal_timestamps:
                        delta_t = real_signal - real_idler
                        # The 0 < delta_t check is good, but a wider range is better for a general plot
                        if -100e-9 < delta_t < 100e-9:
                            delta_t_true.append(delta_t)

                delta_t_true = np.array(delta_t_true)
                if len(delta_t_true) == 0:
                    print("No true coincidences found!")
                    continue

                # Calculate window parameters from true signals
                mean_dt = np.mean(delta_t_true)
                std_dt = np.std(delta_t_true)
                window_center = mean_dt
                window_width = 6 * std_dt  # ±3σ window

                # 2. TRUE + FALSE SIGNALS - all possible coincidences
                delta_t_all = []
                for idler_time in all_idler_timestamps:
                    for signal_time in all_signal_timestamps:
                        delta_t = signal_time - idler_time
                        if -100e-9 < delta_t < 100e-9:
                            delta_t_all.append(delta_t)

                delta_t_all = np.array(delta_t_all)

                # 3. COINCIDENT WINDOW APPLIED - only coincidences within window
                # Fix the assignment here:
                true_coincidences, noisy_coincidences = self.find_coincidences(
                    pair_times, z, true_TOF, quantum_efficiency, window_center, window_width
                )

                # Use the correct variable for plotting
                delta_t_windowed = noisy_coincidences

                # Calculate signal preservation ratio
                signal_ratio = len(true_coincidences) / len(noisy_coincidences) if len(noisy_coincidences) > 0 else 0

                writer.writerow([
                    trial,
                    f"{distance:.3f}",
                    f"{self.subject_reflectivity:.3f}",
                    f"{self.b_s:.1f}",
                    f"{quantum_efficiency:.3f}",
                    f"{true_TOF * 1e9:.3f}",
                    len(delta_t_true),
                    len(delta_t_all),
                    len(delta_t_windowed),
                    f"{signal_ratio:.3f}"
                ])


                # [Histograms plotting section remains the same, but now `delta_t_windowed` is correct]
                fig, axs = plt.subplots(3, 1, figsize=(12, 10))
                plt.subplots_adjust(hspace=0.8)

                # Histogram 1: TRUE SIGNALS ONLY
                if len(delta_t_true) > 0:
                    axs[0].hist(delta_t_true * 1e9, bins=30, alpha=0.7, edgecolor='black', color='blue')
                    axs[0].axvline(mean_dt * 1e9, color='red', linestyle='--',
                                   label=f'Mean: {mean_dt * 1e9:.2f} ns')
                    axs[0].axvline((mean_dt - 3 * std_dt) * 1e9, color='orange', linestyle='--',
                                   alpha=0.7, label=f'±3σ window')
                    axs[0].axvline((mean_dt + 3 * std_dt) * 1e9, color='orange', linestyle='--', alpha=0.7)
                    axs[0].legend()
                else:
                    axs[0].text(0.5, 0.5, 'No true coincidences found',
                                ha='center', va='center', transform=axs[0].transAxes, fontsize=14)

                axs[0].set_xlabel("Time Difference Δt (ns)")
                axs[0].set_ylabel("Counts")
                axs[0].set_title(f"Trial {trial}: TRUE SIGNALS ONLY\n" +
                                 f"Distance: {distance:.2f}m, QE: {quantum_efficiency:.2f}, " +
                                 f"True coincidences: {len(delta_t_true)}")
                axs[0].grid(True, alpha=0.3)


                # Histogram 2: TRUE + FALSE SIGNALS
                if len(delta_t_all) > 0:
                    axs[1].hist(delta_t_all * 1e9, bins=50, alpha=0.7, edgecolor='black', color='green')
                    axs[1].axvline(mean_dt * 1e9, color='red', linestyle='--',
                                   label=f'Expected: {mean_dt * 1e9:.2f} ns')
                    axs[1].axvline((mean_dt - 3 * std_dt) * 1e9, color='orange', linestyle='--',
                                   alpha=0.7, label=f'±3σ window')
                    axs[1].axvline((mean_dt + 3 * std_dt) * 1e9, color='orange', linestyle='--', alpha=0.7)
                    axs[1].legend()
                else:
                    axs[1].text(0.5, 0.5, 'No coincidences found',
                                ha='center', va='center', transform=axs[1].transAxes, fontsize=14)

                axs[1].set_xlabel("Time Difference Δt (ns)")
                axs[1].set_ylabel("Counts")
                axs[1].set_title(f"Trial {trial}: TRUE + FALSE SIGNALS\n" +
                                 f"All coincidences: {len(delta_t_all)}, " +
                                 f"Background ratio: {self.b_s:.1f}:1")
                axs[1].grid(True, alpha=0.3)

    # Histogram 3: COINCIDENT WINDOW APPLIED
                if len(delta_t_windowed) > 0:
                    # This line will now work correctly
                    axs[2].hist(delta_t_windowed * 1e9, bins=50, alpha=0.7, edgecolor='black', color='purple')
                    axs[2].set_title(
                        f"3. COINCIDENT WINDOW APPLIED\n(Window width: {window_width * 1e9:.2f} ns, Windowed coincidences: {len(delta_t_windowed)})")
                else:
                    axs[2].text(0.5, 0.5, 'No coincidences found in window',
                                horizontalalignment='center', verticalalignment='center',
                                transform=axs[2].transAxes, fontsize=14)
                    axs[2].set_title("3. COINCIDENT WINDOW APPLIED\n(No coincidences found)")

                axs[2].set_xlabel("Time Difference Δt (ns)")
                axs[2].set_ylabel("Counts")
                axs[2].grid(True, alpha=0.3)

                # Add window indicators
                axs[2].axvline(mean_dt * 1e9, color='red', linestyle='--',
                               label=f'Window center: {mean_dt * 1e9:.2f} ns', zorder =10)
                axs[2].axvline((mean_dt - window_width / 2) * 1e9, color='orange', linestyle='-', alpha=0.7,
                               label=f'Window edges', zorder =10)
                axs[2].axvline((mean_dt + window_width / 2) * 1e9, color='orange', linestyle='-', alpha=0.7, zorder =10)
                axs[2].legend()
                plt.savefig(f"histogram_trial_{trial}.png", dpi=300)
                plt.tight_layout()
                plt.close()

                """
                # Print summary for each trial
                print(f"\nTrial {trial} - Distance: {distance:.2f}m")
                print(f"Window center: {mean_dt * 1e9:.3f} ns")
                print(f"Window width: {window_width * 1e9:.3f} ns")
                print(f"True coincidences: {len(delta_t_true)}")
                print(f"All possible coincidences: {len(delta_t_all)}")
                print(f"Windowed coincidences: {len(delta_t_windowed)}")
                print(f"Signal preservatihon ratio: {signal_ratio:.3f}")
                """

    def window_width(self, idler_times, signal_times, window_center, half_width):
        both_detected = np.isfinite(idler_times) & np.isfinite(signal_times)

        delta_t = signal_times - idler_times

        lower = window_center - half_width
        upper = window_center + half_width
        in_window = (delta_t >= lower) & (delta_t <= upper)  # true

        keep_mask = both_detected & in_window
        indices_kept = np.nonzero(keep_mask)[0]
        delta_t_kept = delta_t[keep_mask]
        return keep_mask, delta_t_kept, indices_kept

# Example usage
if __name__ == "__main__":
    herald_sim = HeraldGating()
    herald_sim.simulation()