import signals
from math_utilities import differentiate, integrate, median

# Physionet MIT-BIH database files
files = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "121",
    "122",
    "123",
    "124",
    "200",
    "201",
    "202",
    "203",
    "205",
    "207",
    "208",
    "209",
    "210",
    "212",
    "213",
    "214",
    "215",
    "217",
    "219",
    "220",
    "221",
    "222",
    "223",
    "228",
    "230",
    "231",
    "232",
    "233",
    "234",
]


def main():
    record_name = input("Record name: ")
    if record_name not in files:
        print("Invalid record name.")
        return
    window_start_sec = input("Sample from: ")
    window_length_sec = input("Sample for: ")
    try:
        window_start_sec = int(window_start_sec)
        window_length_sec = int(window_length_sec)
    except ValueError:
        print("Invalid input. Please enter integer values for start time and length.")
        return
    if window_start_sec < 0 or window_length_sec < 60:
        print(
            "Invalid input. Start time must be non-negative and length must be at least 60 seconds."
        )
        return
    try:
        record, sampling_rate, reference_peaks = load(
            record_name, window_start_sec, window_length_sec
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    # Clean record, 5hz-15hz bandpass
    cleaned = clean(record, sampling_rate)
    # Detect R-peaks
    detected_peaks = detect(cleaned, sampling_rate)
    # Compare with annotated beats, returns sensitivity, positive predictive values,
    # true and false positives, and false negatives
    sensitivity, ppv, tp, fp, fn = validate(
        detected_peaks, reference_peaks, sampling_rate
    )
    # Computes heartrate variability
    hrv = compute_hrv(detected_peaks, sampling_rate)
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"PPV: {ppv:.4f}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    if hrv is None:
        print("Not enough data to compute HRV metrics.")
    else:
        rmssd, sdnn, excluded, total = hrv
        share = 100 * excluded / total
        print(f"RMSSD: {rmssd:.2f} ms")
        print(f"SDNN: {sdnn:.2f} ms")
        print(f"Excluded beats: {excluded}/{total} ({share:.2f}%)")
        if share > 30:
            print("Warning: More than 30% of beats were excluded from HRV calculation.")


def load(record_name, window_start_sec, window_length_sec):
    """Load the relevant window, channel, and beats for the given record.

    Args:
        record_name (str): The name of the record to load.
        window_start_sec (int): The start time of the window in seconds.
        window_length_sec (int): The length of the window in seconds.

    Returns:
        tuple: A tuple containing the loaded record, sampling rate, and reference peaks.

    Raises:
        ValueError: If the start time or window length is invalid.
    """
    # Parse header file for sampling rate, sample count and recording data for each channel
    with open(f"mitdb/{record_name}.hea", "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    sampling_rate = int(lines[0].split()[2])
    n_samples = int(lines[0].split()[3])
    # Find indexes for the given window, raise errors if invalid
    start_index = int(window_start_sec) * sampling_rate
    if start_index >= n_samples:
        raise ValueError("Start time exceeds the length of the record.")
    end_index = start_index + int(window_length_sec) * sampling_rate
    if end_index > n_samples:
        raise ValueError("End time exceeds the length of the record.")
    c1_info = lines[1].split()
    c1_gain = int(c1_info[2])
    c1_baseline = int(c1_info[4])
    c1_lead = c1_info[8]
    c2_info = lines[2].split()
    c2_gain = int(c2_info[2])
    c2_baseline = int(c2_info[4])
    c2_lead = c2_info[8]
    c1 = []
    c2 = []
    # Read ECG data
    with open(f"mitdb/{record_name}.dat", "rb") as f:
        data = f.read()
        data = data[start_index * 3 : end_index * 3]  # Restrict to relevant window
        for n in range(
            0, len(data) - 2, 3
        ):  # Format 212 stores 2 12-bit samples in 3 bytes
            # Byte 1 and the low-nibble of byte 2 form 1 12-bit sample
            c1_sample = ((data[n + 1] & 0x0F) << 8) | data[n]
            # Account for negative values, baseline, and gain
            if c1_sample >= 2048:
                c1_sample -= 4096
            c1_sample -= c1_baseline
            c1_sample /= c1_gain
            c1.append(c1_sample)  # Append to relevant channel
            # Byte 3 and the high-nibble of byte 2 for the second 12-bit sample
            c2_sample = ((data[n + 1] & 0xF0) << 4) | data[n + 2]
            # Account for negative values, baseline, and gain
            if c2_sample >= 2048:
                c2_sample -= 4096
            c2_sample -= c2_baseline
            c2_sample /= c2_gain
            c2.append(c2_sample)  # Append to relevant channel
    # Physionet annotations indicative of beats
    beat_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 34, 35, 38]
    reference_peaks = []
    # Read annotations
    with open(f"mitdb/{record_name}.atr", "rb") as f:
        sample = 0
        while True:
            data = f.read(2)  # Formatted as 1 or more 16 bit (2 byte) samples
            if len(data) < 2:
                break
            combined = (data[1] << 8) | data[0]  # Combine bytes
            ann_type = (combined >> 10) & 0x3F  # Annotation type in high 6 bits
            delta = combined & 0x3FF  # Time since last annotation in low 10 bits
            # "0" indicates no annotation, continue
            if ann_type == 0:
                continue
            # "59" indicates SKIP - 10 bits can only store a delta up to 1023 samples, 
            # SKIP is used for anything longer
            if ann_type == 59:
                data = f.read(4)  # SKIP is a 32-bit signaled interval
                if len(data) < 4:
                    break
                # Assemble 32-bit value, proceed past it
                low = (data[3] << 8) | data[2]
                high = (data[1] << 8) | data[0]
                sample += (high << 16) | low
                continue
            # "63" indicates auxiliary text, skip past it
            if ann_type == 63:
                aux_len = delta
                f.read(aux_len + (aux_len % 2))
                continue
            # Field changes, continue
            if ann_type in (60, 61, 62):
                continue
            sample += delta
            if ann_type in beat_set and sample >= start_index and sample <= end_index:
                reference_peaks.append(sample - start_index)
    # Determine and return correct lead, sampling rate, and annotated peaks
    if c1_lead == "MLII":
        return c1, sampling_rate, reference_peaks
    elif c2_lead == "MLII":
        return c2, sampling_rate, reference_peaks
    elif c1_lead == "V5":
        return c1, sampling_rate, reference_peaks
    elif c2_lead == "V5":
        return c2, sampling_rate, reference_peaks
    else:
        raise ValueError("No compatible lead found.")


def clean(record, sampling_rate):
    """Clean the signal, attenuating unwanted frequencies.
    Args:
        record (list): The signal to clean.
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        list: The cleaned signal.
    """
    # Find filter coefficients to attenuate frequencies below 5hz
    hpb0, hpb1, hpb2, hpa1, hpa2 = signals.calculate_filter_coefficients(
        5, sampling_rate, "high"
    )
    high_passed = signals.filtfilt(record, hpb0, hpb1, hpb2, hpa1, hpa2)
    # Find filter coefficients to attenuate frequencies above 15hz
    lpb0, lpb1, lpb2, lpa1, lpa2 = signals.calculate_filter_coefficients(
        15, sampling_rate, "low"
    )
    cleaned = signals.filtfilt(high_passed, lpb0, lpb1, lpb2, lpa1, lpa2)
    return cleaned


def detect(cleaned, sampling_rate):
    """Detect R-peaks.

    Args:
        cleaned (list): The cleaned signal.
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        list: The indices of the detected R-peaks.
    """

    def is_local_max(i):
        """Determines if a point is a local maximum, including the end of a plateau."""
        return integrated[i] >= integrated[i - 1] and integrated[i] > integrated[i + 1]

    def locate_r(peak):
        """Finds peak within cleaned signal since the integrated signal lags."""
        start = max(0, peak - integration_window)
        window = cleaned[start : peak + 1]
        best = max(range(len(window)), key=lambda k: abs(window[k]))
        return start + best

    def qrs_slope(peak):
        """Finds highest slope for a given peak."""
        lo = max(0, peak - slope_window)
        hi = min(len(derivative), peak + slope_window)
        return max(abs(v) for v in derivative[lo:hi])

    def is_t_wave(peak, prev_peak, prev_slope):
        """Determines if a peak is a t-wave."""
        if (
            prev_peak is None or peak - prev_peak >= t_wave_range
        ):  # T-waves come within 360ms of an r-peak
            return False
        return (
            qrs_slope(peak) < 0.5 * prev_slope
        )  # T-wave if less than half the slope of the last r-peak

    def accept(peak, searchback=False):
        """Appends index to r-peaks, adjusts relevant values."""
        nonlocal spki, threshold_i1, last_slope, avg_rr, noise_head, refractory_until
        gain = 0.25 if searchback else 0.125
        spki = gain * integrated[peak] + (1.0 - gain) * spki
        threshold_i1 = npki + 0.25 * (spki - npki)
        r_peaks_i.append(peak)
        last_slope = qrs_slope(peak)
        refractory_until = (
            peak + refrac
        )  # Window during which another r-peak is impossible
        if len(r_peaks_i) > 1:
            k = min(8, len(r_peaks_i) - 1)
            avg_rr = (
                r_peaks_i[-1] - r_peaks_i[-1 - k]
            ) / k  # Determine rr-interval if possible
        while noise_head < len(noise_peaks) and noise_peaks[noise_head] <= peak:
            noise_head += 1

    def reject(peak):
        """Appends index to noise peaks, adjusts relevant values."""
        nonlocal npki, threshold_i1
        npki = 0.125 * integrated[peak] + 0.875 * npki
        threshold_i1 = npki + 0.25 * (spki - npki)
        noise_peaks.append(peak)

    def find_missed():
        """Searchback function."""
        threshold_i2 = (
            0.5 * threshold_i1
        )  # Original threshold missed a peak, so reduce it
        best = None
        # Searches noise peaks for possible r-peaks
        for j in range(noise_head, len(noise_peaks)):
            p = noise_peaks[j]
            if p <= refractory_until or integrated[p] <= threshold_i2:
                continue
            if best is None or integrated[p] > integrated[best]:
                best = p
        return best

    def prominence(p):
        """Finds amplitude within 50ms of a peak inside the filtered signal."""
        return max(
            abs(v)
            for v in cleaned[max(0, p - prominence_half) : p + prominence_half + 1]
        )

    fs = float(sampling_rate)
    n_samples = len(cleaned)
    refrac = int(0.20 * fs)
    t_wave_range = int(0.36 * fs)
    slope_window = int(0.075 * fs)
    integration_window = int(0.15 * fs)
    learning_end = min(int(2.0 * fs), n_samples)
    if n_samples < 5 or learning_end < 3:
        return []
    derivative = differentiate(cleaned)  # Five point derivative
    squared = [v * v for v in derivative]
    integrated = integrate(squared, integration_window)
    # Seed SPKI (typical r-peak) and NPKI (typical noise peak)
    # Max and mean respectively
    spki = max(integrated[:learning_end])
    npki = sum(integrated[:learning_end]) / learning_end
    threshold_i1 = npki + 0.25 * (spki - npki)
    r_peaks_i = []
    noise_peaks = []
    noise_head = 0
    last_slope = 0.0
    avg_rr = None
    refractory_until = 0
    searchback_mark = -1
    index = learning_end
    # Main detection loop
    while index < len(integrated) - 1:
        # Searchback if current interval exceeds 166% of avg interval
        if (
            avg_rr is not None
            and index - r_peaks_i[-1] > 1.66 * avg_rr
            and len(noise_peaks) != searchback_mark
        ):
            searchback_mark = len(
                noise_peaks
            )  # Ensures searchback can't reinitialize without new noise peaks
            candidate = find_missed()
            # Accept candidate if found and adjust values accordingly
            if candidate is not None:
                accept(candidate, searchback=True)
        # Regular detection
        if is_local_max(index):
            # R-peak impossible during refractory period
            if index < refractory_until:
                pass
            # Verify if a peak passes the threshold and is not a t-wave
            elif integrated[index] > threshold_i1:
                prev = r_peaks_i[-1] if r_peaks_i else None
                if is_t_wave(index, prev, last_slope):
                    reject(index)
                else:
                    accept(index)
            else:
                reject(index)
        index += 1
    # Scan learning end with thresholds built
    early = []
    prev_peak, prev_slope = None, 0.0
    limit = (
        r_peaks_i[0] - refrac if r_peaks_i else learning_end
    )  # Ensure r-peak can't be detected within refrac of first main r-peak
    i = 1
    while i < min(learning_end, len(integrated) - 1) and i < limit:
        # Check if peak passes threshold
        if is_local_max(i) and integrated[i] > threshold_i1:
            # Check if peak is a t-wave
            if not is_t_wave(i, prev_peak, prev_slope):
                early.append(i)
                prev_peak, prev_slope = i, qrs_slope(i)
                i += refrac
                continue
        i += 1
    r_peaks_i = early + r_peaks_i  # Prepend learning end peaks to main peaks
    r_peaks = []
    # Find peaks within cleaned signal
    for p in r_peaks_i:
        loc = locate_r(p)
        if not r_peaks or loc - r_peaks[-1] >= refrac:
            r_peaks.append(loc)
    # Filter p-waves from detected peaks
    prominence_half = max(1, int(0.05 * fs))
    pruned = []
    for p in r_peaks:
        # If two detections are within 300ms, keep that with higher amplitude
        if pruned and p - pruned[-1] < 0.30 * fs:
            if prominence(p) > prominence(pruned[-1]):
                pruned[-1] = p
        else:
            pruned.append(p)
    return pruned


def validate(detected_peaks, reference_peaks, sampling_rate):
    """Check detected peaks against annotated peaks.

    Args:
        detected_peaks (list): The list of detected R-peak indices.
        reference_peaks (list): The list of annotated R-peak indices.
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        tuple: A tuple containing the sensitivity, positive predictive value, true positives, false positives, and false negatives.
    """
    i = j = 0
    tp = fp = fn = 0
    tolerance = int(
        0.15 * sampling_rate
    )  # Detected peaks must come within 150ms of annotated peaks
    # Walk through detected and annotated peaks
    while i < len(reference_peaks) and j < len(detected_peaks):
        difference = detected_peaks[j] - reference_peaks[i]
        # Within tolerance, detected peak is valid
        if abs(difference) <= tolerance:
            tp += 1
            i += 1
            j += 1
        # Negative difference indicates false beat detected
        elif difference < 0:
            fp += 1
            j += 1  # Only increment detected peaks
        # Difference greater than tolerance indicates a beat when undetected
        else:
            fn += 1
            i += 1  # Only increment annotated peaks
    fn += len(reference_peaks) - i  # Account for false negatives outside loop
    fp += len(detected_peaks) - j  # Account for false positives outside loop
    total_ref = tp + fn
    sensitivity = tp / total_ref if total_ref else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    return sensitivity, ppv, tp, fp, fn


def flag(rr, window=2, tolerance=0.20, min_rr=300.0, max_rr=2000.0):
    """Flag normal beats within all detected beats.

    Args:
        rr (list): The list of RR intervals.
        window (int): The window size for local median calculation.
        tolerance (float): The tolerance for flagging abnormal intervals.
        min_rr (float): The minimum acceptable RR interval.
        max_rr (float): The maximum acceptable RR interval.

    Returns:
        list: A list of boolean values indicating valid beats.
    """
    n = len(rr)
    valid = [True] * n
    # Invalidate impossible heart rates
    for i in range(n):
        if rr[i] < min_rr or rr[i] > max_rr:
            valid[i] = False
    # Reject intervals that deviate from the median by more than 20%
    for i in range(n):
        if not valid[i]:
            continue
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbors = [rr[j] for j in range(lo, hi) if j != i]
        if not neighbors:
            continue
        local = median(neighbors)
        if local <= 0 or abs(rr[i] - local) > tolerance * local:
            valid[i] = False
    # Flag invalid peaks
    flagged = [i for i in range(n) if not valid[i]]
    # NN interval can't follow invalid beats
    for i in flagged:
        if i + 1 < n:
            valid[i + 1] = False
    return valid


def compute_hrv(detected_peaks, sampling_rate):
    """Compute HRV metrics.

    Args:
        detected_peaks (list): The list of detected R-peak indices.
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        tuple: A tuple containing the RMSSD, SDNN, number of removed intervals, and total number of intervals.
        None: If there are not enough detected peaks to compute HRV metrics.
    """
    # Check if beats are sufficient to compute
    if len(detected_peaks) < 3:
        return None
    rr_intervals = [
        (detected_peaks[i] - detected_peaks[i - 1]) / sampling_rate * 1000
        for i in range(1, len(detected_peaks))
    ]
    valid = flag(rr_intervals)
    # Assemble list of NN intervals
    kept = [rr_intervals[i] for i in range(len(rr_intervals)) if valid[i]]
    diffs = [
        rr_intervals[i] - rr_intervals[i - 1]
        for i in range(1, len(rr_intervals))
        if valid[i] and valid[i - 1]
    ]
    # Check if data still sufficient for compute
    if len(kept) < 2 or not diffs:
        return None
    rmssd = (sum(d**2 for d in diffs) / len(diffs)) ** 0.5
    rr_mean = sum(kept) / len(kept)
    sdnn = (
        sum((interval - rr_mean) ** 2 for interval in kept) / (len(kept) - 1)
    ) ** 0.5
    return rmssd, sdnn, len(rr_intervals) - len(kept), len(rr_intervals)


if __name__ == "__main__":
    main()
