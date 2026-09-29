# ECG Analysis and HRV Computation from Scratch

Reads raw PhysioNet ECG recordings, detects heartbeats, and computes heart-rate variability metrics in pure Python, with neither the standard or third-party libraries. 99.66% beat-detection sensitivity across all 48 records of the MIT-BIH Arrhythmia Database.

## Why

Reading an ECG and reporting HRV is a handful of lines with WFDB and NeuroKit2. Doing it without them means learning and implementing what they actually do.

- **Bitwise operations:** PhysioNet stores ECGs in the 212 format, where two 12-bit samples share three bytes.
- **Digital signal processing:** A Butterworth bandpass is the first step in beat detection, isolating QRS enegy from noise. Applying it means placing poles on a circle in the complex plane, prewarping the cutoff to correct for bilinear-transform frequency distortion, and mapping analog poles to the z-domain. That requires complex arithmetic, exponentials, and trigonometric functions, all defined here from their Taylor series.
- **QRS detection:** A modified version of the 1985 Pan-Tompkins algorithm differentiates, squares, integrates, and detects peaks with two adaptive thresholds, discriminating T-waves, tracking RR-intervals, and rescanning as needed.
- **HRV computation:** RMSSD and SDNN, the reported metrics, are defined over normal-to-normal intervals. Ectopic beats must be filtered before computation, requiring basic beat classification.

Implementing each means learning each, and such was the goal of this project.

## Quick start
 
Requires Python 3.x. No installation step — there are no dependencies.
 
The MIT-BIH Arrhythmia Database is freely available from PhysioNet and is not included in this repository (~91 MB). Download the `.hea`, `.dat`, and `.atr` files into `mitdb/`:
 
```bash
wget -r -np -nH --cut-dirs=4 -P mitdb/ \
  https://physionet.org/files/mitdb/1.0.0/
```
 
Then analyze a single record:
 
```
$ python project.py
Record name:  100        # any record in the MIT-BIH database
Start second: 0          # second at which to begin scanning
End second:   300        # second at which to stop
```
 
```
Sensitivity: 1.0000
PPV: 1.0000
True Positives: 371
False Positives: 0
False Negatives: 0
RMSSD: 26.09 ms
SDNN: 25.41 ms
Excluded beats: 10/370 (2.70%)
```
 
### Reproducing the published results
 
```
python benchmark.py
```
 
```bash
python benchmark.py --all
```
## Results
 
### QRS detection
 
Benchmarked against the cardiologist-annotated reference beats of the MIT-BIH Arrhythmia Database. A detection counts as correct if it falls within 150 ms of an annotated beat, the tolerance specified by ANSI/AAMI EC57.
 
| | Sensitivity | PPV | Beats | FN | FP |
|---|---|---|---|---|---|
| All 48 records | **99.66%** | **99.61%** | 109,150 | 366 | 425 |
| 44-record EC57 subset | 99.65% | 99.58% | 100,415 | 354 | 421 |
 
Runtime is 0.34 s for a 5-minute record and 2.0 s for a full 30-minute record, in pure Python on a single core.
 
For context, the original Pan & Tompkins paper reports 99.3% sensitivity and 99.5% PPV. That figure relates to different records and is **not** directly comparable, but indicates the range a correct implementation should land in.
 
### HRV accuracy
 
Measured across the first five minutes of ten records, chosen to show a range of ectopy. *Unfiltered* takes every detected interval at face value with no ectopic filtering; *This project* applies the filter described below; *Reference* is computed from the annotations.
 
#### RMSSD (ms)
 
| Record | Unfiltered | This project | Reference | Excluded |
|---|---|---|---|---|
| 103 | 29.0 | 29.0 | 29.1 | 0.0% |
| 234 | 17.4 | 17.4 | 17.1 | 0.0% |
| 100 | 55.9 | 26.1 | 25.9 | 2.7% |
| 113 | 96.9 | 83.2 | 87.8 | 2.8% |
| 106 | 361.9 | 60.7 | 61.2 | 33.3% |
| 119 | 477.6 | 35.2 | 37.5 | 64.6% |
| 208 | 193.3 | 56.9 | 25.6 | 61.3% |
| 233 | 235.6 | 32.2 | 15.4 | 65.8% |
| 221 | 304.2 | 91.2 | 216.4 | 70.7% |
| 203 | 283.4 | 78.2 | 227.0 | 76.1% |
 
#### SDNN (ms)
 
| Record | Unfiltered | This project | Reference | Excluded |
|---|---|---|---|---|
| 103 | 38.0 | 38.0 | 38.3 | 0.0% |
| 234 | 14.3 | 14.3 | 14.3 | 0.0% |
| 100 | 38.7 | 25.4 | 25.4 | 2.7% |
| 113 | 80.4 | 71.7 | 74.9 | 2.8% |
| 106 | 234.9 | 108.1 | 107.2 | 33.3% |
| 119 | 261.5 | 27.7 | 29.1 | 64.6% |
| 208 | 131.9 | 41.9 | 30.5 | 61.3% |
| 233 | 131.4 | 28.8 | 18.5 | 65.8% |
| 221 | 197.0 | 83.3 | 156.9 | 70.7% |
| 203 | 202.0 | 155.9 | 187.0 | 76.1% |
 
**Read the exclusion column before the others.** The ectopic filter assumes intervals should resemble their neighbors, which holds up to roughly 60% exclusion and then stops holding. Records 203 and 221 are atrial fibrillation: the rhythm is irregular by nature, so "deviates from its neighbors" stops meaning "ectopic" and the filter discards genuine variability — hence 78.2 against a reference of 227.0. Records 208 and 233 fail in the opposite direction, with frequent ventricular ectopy surviving into the retained intervals. These are the filter's known failure mode, not noise in the measurement, and the exclusion percentage is printed at runtime specifically so the breakdown is visible rather than silent.

## How it works

The flow is five stages: **load**, **clean**, **detect**, **validate**, and **HRV computation**.

### Load

PhysioNet ECGs are three files each: header, data, and annotations. Header contains key information about the recording, data contains the actual ECG signal, and annotations contains just that: beat classifications and expert-annotated R-peak indices. Before either data or annotations can be parsed, key information must be extracted from the header file: the sampling rate, total sample count, the gain, baseline, and lead for each channel. 

As mentioned, PhysioNet stores ECGs in the 212 format - as two, 12-bit samples spread across three bytes. Each sample corresponds to a channel. 'load' iterates over three bytes at a time, combining byte 1 with the low-nibble of byte 2 and byte 3 with the high nibble of byte 2 to form each sample. Each is appended to the relevant channel. Values are two's complement, so anything at or above 2048 is negative. Each is then shifted by the baseline and divided by the gain to give millivolts.

The annotations are formatted as one or more 16-bit/2-byte samples, the high 6 bits storing the annotation type, and the lower 10 the delta since the last annotation. The code accounts for several notable annotations, including SKIP: as 10 bits can only store a delta of 1023 samples, SKIP indicates a longer delta, which the next 4 bytes encode. If the annotation type is amongst those indicating beats, that index is appended to a list of reference peaks.

### Clean

Pan-Tompkins calls for a 5–15 Hz bandpass to isolate QRS energy. This implementation uses a second-order Butterworth cascade in place of the paper's integer-coefficient design, designed from scratch rather than with precomputed coefficients:
 
1. Prewarp the cutoff: The bilinear transform compresses the infinite analog frequency axis onto the digital one, which stops at Nyquist. That bends every frequency except DC, so a filter designed directly at the cutoff lands below it once digital. Applying the inverse picks the analog frequency that will bend onto the target.
2. Place the analog poles: Butterworth spaces poles evenly around a circle; second order takes the two in the left half-plane, at 3π/4 and 5π/4. Left-plane poles guarantee stability — a positive exponent would grow without bound under feedback.
3. Map to the z-domain: via the bilinear transform, which sends the left half-plane inside the unit circle, so the response decays rather than diverging.
4. Recover coefficients: by multiplying the roots back out, and **normalize the gain** so the passband comes through untouched — at DC for the lowpass, at Nyquist for the highpass.

The result is a difference equation, y[n] = (b0)(x[n]) + (b1)(x[n - 1]) + (b2)(x[n - 2]) − (a1)(y[n - 1]) − (a2)(y[n - 2]), applied forward and backward for zero phase. Below is the derivation, which dives deeper:

#### Derivation

H(z) equals Y(z) / X(z) - output over input, a transfer function. Y(z) / X(z) = B(z) / A(z) = ((b0) + (b1)(z^-1) + (b2)(z^-2)) / ((a0) + (a1)(z^-1) + (a2)(z^-2)), where B(z) is the product of the distance to the zeros and A(z) is the product of the distance to the poles. The output thus emphasizes and attenuates the desired frequencies. The a and b coefficients are those of a polynomial whose roots are the poles/zeros respectively. b0 is the gain, the weight on the current input, and a0 is the weight on the current output, which is normalized to one.
Cross multiplying gets rid of the fraction:
Y(z) / X(z) = ((b0) + (b1)(z^-1) + (b2)(z^-2)) / ((a0) + (a1)(z^-1) + (a2)(z^-2)) 
becomes
(Y(z))((a0) + (a1)(z^-1) + (a2)(z^-2)) = (X(z))((b0) + (b1)(z^-1) + (b2)(z^-2)).
Distribute:
(Y(z))(a0) + (Y(z))(a1)(z^-1) + (Y(z))(a2)(z^-2) = (X(z))(b0) + (X(z))(b1)(z^-1) + (X(z))(b2)(z^-2)
and now solve for the current output:
(Y(z)) = (((X(z))(b0) + (X(z))(b1)(z^-1) + (X(z))(b2)(z^-2) - (Y(z))(a1)(z^-1) - (Y(z))(a2)(z^-2))) / a0.
All coefficients are divided by a0 in their computation - so the same name is retained, though a0 can be removed from the equation. Rearranging the equation to place functions of z next to terms of z results in:
Y(z) = (b0)(X(z)) + (b1)(z^-1)(X(z)) + (b2)(z^-2)(X(z)) - (a1)(z^-1)(Y(z)) - (a2)(z^-2)(Y(z)).
Computationally, this z^-1 is a delay of one unit. The Z-transform explains why:

The Z-transform (X(z) = sum(x[n] * z^-n) for n = -inf to inf) converts this time domain signal into a frequency-domain representation.
Let y[n] = x[n - 1], giving the Z-transform: 
Y(z) = sum(x[n - 1] * z^-n) for n = -inf to inf. 
Substitute the time index m = n - 1: 
Y(z) = sum(x[m] * z^-(m + 1)) for m = -inf to inf. 
Using exponent rules, z^-(m + 1) = z^-m * z^-1. Pull z^-1 out of the summation: 
Y(z) = (z^-1)(sum(x[m] * z^-m) for m = -inf to inf),
which is the original definition of X(z), only with m instead of n. Therefore, y[n] = x[n - 1] gives Y(z) = (z^-1)(X(z)). Similarly, x[n - k] gives (z^-k)(X(z)).
Apply this to the difference equation:
y[n] = (b0)(x[n]) + (b1)(x[n - 1]) + (b2)(x[n - 2]) - (a1)(y[n - 1]) - (a2)(y[n - 2]), where x[n] is the input and y[n] is the output. This is the filter - it now needs the right coefficients.

First, the cutoff must be prewarped. The bilinear transform ((1 + x/(2 * the sampling rate)) / (1 - x/(2 * the sampling rate))) squeezes the analog frequency axis, which runs to infinity, into the digital one, which stops at Nyquist (half the sampling rate, and the highest frequency such a sampling rate can record). That compression bends every frequency except DC, meaning that a filter designed directly at the cutoff would land below it once made digital. Applying the inverse of the bilinear transform (2 * the sampling rate * tan(pi * x / the sampling rate)) to the cutoff picks the analog frequency that will bend onto it. tan is defined by the sine/cosine identity - each of which are, alongside several other functions, implemented via their Taylor series.

Next, the analog poles are placed. A Butterworth filter spaces poles evenly around a circle; a second order example takes the two in the left plane: (3pi/4) and (5pi/4). Left-plane poles ensure stability: since the filter is fed previous outputs, a positive exponent would continuously grow. A negative one decays. The poles scale these angles to a circle with a radius equal to the cutoff. A highpass filter maps s to (the cutoff^2)/s. This reduces to conjugation when |s|= the cutoff, as it does here. As a result, a highpass just flips these poles across the horizontal axis - however, since the poles sit on the circle of radius wc, this maps the pair onto itself. Only the numerator, the zeros, distinguishes a highpass from a lowpass at the same cutoff.

Then, the bilinear transform is applied, converting the analog poles to digital ones. It maps the left half of the analog plane to the inside of the unit circle, as in, to values less than one. This ensures stability - values more than one would cause positive feedback. This way, they decay.

These values are the roots, the transfer function needs coefficients. The roots are multiplied back out to find them: (z - d0)(z - d1) = z^2 - (d0+d1)z + d0·d1. Matching that against 1 + a1·z^-1 + a2·z^-2 gives a1 = -(d0 + d1) and a2 = d0·d1. As complex conjugates, their sum and product are always mathematically real.

Lastly, place the zeros and normalize. Butterworth zeros are always placed at the far end of the spectrum from the passband: DC (z = 1) for a highpass and Nyquist (z = -1) for a lowpass. The highpass polynomial (1 - z^-1)^2 expands to 1 - 2z^-1 + z^-2, giving the coefficients 1, -2, and 1. The lowpass polynomial (1 + z^-1)^2 expands to 1 + 2z^-1 + z^-2, giving the coefficients 1, 2, and 1. Gain is chosen to ensure the passband comes through untouched. For the lowpass, that's DC, where every z^-1 becomes 1: (b0+b1+b2)/(1+a1+a2) = 4G/(1+a1+a2). Set that equal to one and G = (1+a1+a2)/4. The highpass does the same at Nyquist, where z^-1 becomes -1, and the alternating signs give (1-a1+a2)/4.

With the filter designed, it can be applied, and the signal cleaned.

### Detect

A typical heartbeat produces a recognizable signature in an ECG: a P-wave, a small, gentle bump, a QRS complex, a sharp, tall spike, and a T-wave, another small, gentle bump. A QRS complex has three parts: a small, but still sharp, dip down (Q), a big spike up (R), then another small dip down below the baseline (S). The goal of the Pan-Tompkins algorithm is to find the peak of the QRS complex, the R-peak.

The signal is first processed, to make the QRS stand out. A QRS complex is the steepest part of a heartbeat, and a derivative highlights it. PT specifically used a 5-point, weighted derivative: relatively resilient to noise, light on computing, weighted near the current value. Squaring the differentiated signal makes all values positive, and disproportionately amplifies large numbers, further emphasizing the already high slopes of the QRS complex. The separate Q, R, and S spikes can be unified into a single hump by integrating across a 150ms window - the maximum expected width of a QRS.

Peaks are then classified against an adaptive threshold made up of two running estimates: SPKI, a typical R-peak, and NPKI, a typical noise peak, with the threshold between them. The first two seconds of the signal are used to seed the SPKI and NPKI and first compute the threshold. As the recording proceeds, each R-peak updates the SPKI, each noise peak updates the NPKI, and the threshold is recomputed after each update. A few nuances further improve the algorithm:

- A refractory window: a new R-peak within 200ms of the last one is physiologically impossible.
- T-wave discrimination: a candidate arriving within 360ms of the previous beat but with less than half its slope is a T-wave, not an R-peak.
- Searchback: if the current interval exceeds 166% of the average RR-interval, the window is re-examined at half threshold to recover a missed beat.

The aforementioned signal processing means the R-peaks within are slightly delayed. Before returning, the actual peak is found by searching for the highest point in the cleaned signal within 150ms of that detected within the processed signal.

This implementation adds to PT in two ways:

#### Addition 1: Prominence
While PT has a rule for T-waves after an R-peak, it has no equivalent for P-waves before it. P-waves never normally reach threshold, so the paper doesn't need one - but on record 108, they do. Prominence solves this: after detected peaks have been mapped back to the raw signal, the output is walked, and when detections occur within 300ms of one another, only that with a higher amplitude is kept - a QRS is taller than the P-wave preceding it. As it's done post-detection, this can only remove detections, not add them. Overall it reduced false positives by 534 at the cost of increasing false negatives by 55.

#### Addition 2: Rescan
The first two seconds seed SPKI and NPKI, and the main scan starts after them. PT describes this learning phase, but not what happens to beats within it. This implementation rescans the learning window after the threshold has been tuned by the whole record to recover missed beats. Overall, it recovered 110 beats, at no cost.

### Validate

Two pointers are walked through the detected and reference peaks, calculating the delta between the two, in that order. If the beat was detected within 150ms of the annotated beat, it's considered valid, and true positives and the two pointers incremented. A negative delta indicates an incorrectly detected beat - false positives and the detected beats are incremented. If the delta is positive, but greater than 150ms, a beat went undetected. False negatives and the reference peaks are incremented. If the lists were of uneven lengths, there are false positives or negatives outside the loop. Once accounted for, sensitivity and positive predictive value are calculated.

### HRV Computation

As mentioned, RMSSD and SDNN, the reported metrics, are defined over normal-to-normal intervals. Ectopic beats must be filtered before computation, requiring basic beat classification. Intervals are rejected based on a few criteria:

- Length: An interval shorter than 300ms or longer than 2000ms is physiologically unlikely.
- Comparison: An interval that deviates more than 20% from the median of its four nearest neighbors is likely ectopic.
- Neighbors: Ectopic beats are followed by a compensatory pause, meaning the interval after a rejection is also invalid.

With NN-intervals found, HRV is computed normally: SDNN is the standard deviation of NN-intervals (N-1), and RMSSD is the root mean square of successive differences - meaning differences that were adjacent in the original series. Compacting the interval list and differencing it pairs intervals that were never successive; thus differences must be confirmed as neighbors before computation.

## Limitations

- The ectopic filter cannot handle irregular rhythms. It assumes intervals should resemble their neighbors. Exclusion is printed to show when the filter breaks down, and begins invalidating usable intervals.
- Only the PhysioNet MIT-BIH database is supported. This is not a general WFDB reader.
- The zero-phase filter starts from zero-initial conditions rather than reflecting the signal at the boundaries, resulting in a small transient at each end of the window.
- Detection assumes a single dominant QRS morphology. Records with abnormal shapes degrade.
- Only time-domain HRV is computed.

## Project Structure

```
project.py  main file
math_utilities.py   sin, cos, tan, exp, etc
complex.py  Complex class and methods
signals.py  Butterworth design and filter application
benchmark.py  Benchmarks detection across sets of records
test_project.py  test for project.py
test_math_utilities.py  tests for math_utilities.py
test_complex.py  tests for complex.py
test_signals.py  tests for signals.py
mitdb/  records, see below
```


## References

- Pan, J. and Tompkins, W. J. (1985). A Real-Time QRS Detection Algorithm. *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230–236.
- Moody, G. B. and Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology*, 20(3), 45–50.
- Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology (1996). Heart rate variability: standards of measurement, physiological interpretation, and clinical use. *Circulation*, 93(5), 1043–1065.
- Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.