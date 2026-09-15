This exposes a simplification I made. The receiver does not magically know −3.5 MHz. I glossed
over a whole subsystem. And you've also spotted a conflation: the carrier frequency and the symbol
center are two different problems. Let me untangle both — and then point out that none of this
happens in your actual code.

Two distinct synchronization problems
I sloppily mixed these together. They're separate:

Carrier recovery — finding the −31.5°/sample rotation rate (the carrier frequency and phase). This
is about which frequency the signal sits at.

Symbol timing recovery — finding which of the 40 samples is the symbol center (where to take
the measurement). This is about when to sample within each symbol period.

Sampling at the center of the 40-sample window does not determine the carrier frequency. It solves
the timing problem, not the frequency problem. These are handled by completely different
mechanisms.

How carrier frequency is actually found
A real receiver gets the carrier frequency in two stages.

Stage 1 — nominal a priori value. The satellite operator assigns carrier 5 to a known slot. The
receiver is configured to tune there — it knows the nominal center is −3.5 MHz (relative to the
transponder center). This is just channel assignment; it's written in the link plan.

Stage 2 — fine correction via carrier recovery. The nominal value is never exact. Doppler shift
(the satellite and ground station move relative to each other), local oscillator drift (the receiver's
crystal isn't perfect), and atmospheric effects shift the actual carrier away from nominal by some
unknown offset Δf. The receiver must measure and track this offset with a feedback loop.

The loop works by exploiting a property of QPSK. The data modulation switches the phase among
4 values that are all multiples of 90°. If you raise the received signal to the 4th power, those 90°
increments become 360° increments — which are indistinguishable from 0°. The modulation
cancels out:

QPSK phases:        45°, 135°, 225°, 315°
After ^4:           180°, 540°, 900°, 1260°
mod 360°:           180°, 180°, 180°, 180°   ← all identical, data erased

With the data stripped away, what remains is a pure tone at 4× the carrier frequency. The receiver
measures that tone's frequency, divides by 4, and recovers the true carrier — including the unknown
Δf offset. This is the 4th-power (or Costas loop) carrier recovery method used in QPSK
receivers.

So the receiver doesn't "know" −3.5 MHz from the start. It starts with the nominal −3.5 MHz, then a
Costas loop continuously refines and tracks the exact frequency and phase, locking onto the real
carrier despite Doppler and drift.

How symbol timing is found (the separate problem)
Once the carrier is recovered and the signal is de-rotated to baseband, the receiver still has to decide
which sample in each 40-sample window to read. The symbol center is where the RRC pulse peaks
and where ISI from neighbors is zero — sampling anywhere else gives a degraded, ISI-
contaminated value.

The receiver finds the optimal sampling instant with a separate timing recovery loop (e.g., a
Gardner detector or early-late gate). It works by exploiting the symmetry of the RRC pulse: at the
correct sampling instant, the pulse is at a peak, so the signal value just before and just after the
sample should be equal. If they're not equal, the timing is off, and the loop nudges the sampling
phase until they balance. This locks onto the symbol centers.

Carrier recovery:   removes the −31.5°/sample rotation (frequency/phase lock)
                          ↓
Timing recovery:    finds which sample is the symbol center (when to read)
                          ↓
Decision:           read the de-rotated value at the symbol center, map to bits

Both loops run continuously and independently, tracking their respective parameters as conditions
drift.

Why none of this is in your code
Here's the important part for your thesis: your system has no receiver and does no demodulation.
There's no carrier recovery, no timing recovery, no symbol decisions anywhere in your pipeline.

Your main.py reads the raw I/Q file and immediately computes an FFT to get the power
spectrum. That's it. You never de-rotate any carrier, never recover symbol timing, never decode a
single bit. You're building a spectrum monitor / anomaly detector, not a communications receiver.

This matters because it changes what you need to know about the carriers. A receiver needs to
exactly recover each carrier's frequency and phase to decode the data. Your detector only needs to
find where the carriers sit in the spectrum — which the FFT reveals directly as power concentrated
at certain frequencies. You detect carrier 5 by seeing a power bump centered around −3.5 MHz in
the averaged spectrum; you never need to lock onto its exact carrier phase.

So my earlier statement — "the receiver knows the carrier is −3.5 MHz" — was a simplification to
explain how QPSK demodulation works in principle. In a real receiver, that knowledge comes from
channel assignment plus a carrier recovery loop. In your detection system, the question never arises,
because you analyze the spectrum directly without demodulating anything. The carrier frequency
isn't something your code recovers — it's something your FFT reveals as a peak location, which is
all your CUC detector needs.

The FFT extracts frequencies by testing every possible rotation rate and measuring which ones
the signal matches. It's essentially doing the same de-rotation a receiver does — but for all
frequencies at once, and just measuring the strength of each rather than decoding it.

The core mechanism: correlation with test rotations
The DFT formula is:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j2\pi kn/N}$$

Each output bin k corresponds to a test frequency f_k = k·fs/N. That e^{-j2π kn/N} term
is a probe — a complex exponential spinning at frequency −f_k. The FFT multiplies your signal
by this probe and sums all the products.

The key insight is what happens when the probe's spin rate matches a carrier's spin rate.

Why carrier 5 lights up the −3.5 MHz bin
Carrier 5's contribution to the samples spins at −31.5° per sample (that's the −3.5 MHz carrier).
Now consider the FFT bin that corresponds to −3.5 MHz. Its probe spins in the opposite direction,
+31.5° per sample (the probe is designed to cancel rotation at its target frequency).

When the FFT multiplies carrier 5's samples by this probe, the two rotations cancel:

Carrier 5 component:  spinning at  −31.5°/sample
Probe at −3.5 MHz:    spinning at  +31.5°/sample
                      ─────────────────────────
Product:              spinning at    0°/sample   ← stationary!

Every product term points in the same direction. When you sum them, they add coherently — the
magnitudes pile up into a large value:

n=0:  product points at, say, 45°
n=1:  product points at 45°   (rotation cancelled)
n=2:  product points at 45°
...
n=8191: product points at 45°

Sum: 8192 vectors all pointing at 45° → huge magnitude → big peak at the −3.5
MHz bin

Why other bins stay small
Now take a bin that does NOT match carrier 5 — say the +5 MHz bin. Its probe spins at −45° per
sample. Multiplying carrier 5's samples by this probe:

Carrier 5 component:  spinning at  −31.5°/sample
Probe at +5 MHz:      spinning at  −45°/sample
                      ─────────────────────────
Product:              spinning at  −76.5°/sample   ← still rotating

The product keeps spinning. As you sum over all 8192 samples, the product vectors point in all
different directions around the circle and cancel out:

n=0:  product points at 0°
n=1:  product points at −76.5°
n=2:  product points at −153°
n=3:  product points at −229.5°
...vectors scattered all around the circle...

Sum: ≈ 0  → near-zero magnitude → no peak at the +5 MHz bin

This cancellation is why non-matching frequencies produce small FFT values. Only the bin whose
probe exactly de-rotates a carrier produces coherent addition and a large peak.

The FFT tries every frequency simultaneously
So the FFT is effectively asking, at every bin: "If I assume the signal contains a component at this
frequency, and I spin a probe to cancel that frequency, do the samples line up coherently or scatter?"

Bin at −16.5 MHz:  probe cancels carrier 1's rotation  → coherent → peak
Bin at −14.5 MHz:  probe cancels carrier 2's rotation  → coherent → peak
...
Bin at −3.5 MHz:   probe cancels carrier 5's rotation  → coherent → peak
...
Bin at +16.0 MHz:  probe cancels carrier 13's rotation → coherent → peak
Bin at +5.0 MHz:   no carrier there                    → scatter  → ~0
Bin at −9.0 MHz:   no carrier there (guard band)        → scatter  → ~0

Each carrier's rotation rate gets cancelled by exactly one bin's probe, producing a peak there. The
empty guard bands match no probe and stay near zero. The result is the spectrum: peaks where
carriers sit, valleys where they don't.

The connection to what a receiver does
A receiver de-rotates one carrier — it spins a probe at exactly carrier 5's recovered frequency to
bring that carrier to a standstill, then reads the data. The FFT does the same de-rotation operation,
but for all 8192 frequencies in parallel, and instead of reading data it just measures "how
stationary did this make the signal?" (the magnitude).

Receiver:  de-rotate ONE known carrier, read the data underneath
FFT:       de-rotate at EVERY frequency, measure coherence at each

That's why the FFT reveals the carrier frequencies without any prior knowledge of where they are.
It doesn't need to be told −3.5 MHz — it tries every frequency, and −3.5 MHz is simply the one
where carrier 5's samples stop spinning and add up. The peak location is the discovered carrier
frequency.

Why this gives you the power spectrum
The QPSK data modulation slightly smears each peak — because the data phase jumps every 40
samples, the coherent addition isn't perfect; it spreads over a band of bins matching the carrier's
bandwidth. That's why carrier 5 appears as a ~1.35 MHz-wide flat-topped bump rather than a single
infinitely-sharp spike. The RRC bandwidth determines the width of the smear. Taking |X[k]|²
converts the coherent-sum magnitudes into power, and Welch-averaging 100 frames smooths the
data-induced fluctuations — leaving the clean carrier shapes your detector segments.

The last paragraph compressed a lot. Let me unpack why the data turns a sharp spike into a band.

What a pure carrier with no data would do
Imagine carrier 5 transmitting no data — just a constant phase, rotating steadily at −31.5°/sample
forever. When the FFT de-rotates at exactly −3.5 MHz, the probe cancels the rotation perfectly, and
every single de-rotated sample points in the identical direction:

De-rotated samples (no data):
n=0:    points at 45°
n=1:    points at 45°
n=2:    points at 45°
...
n=8191: points at 45°

All 8192 identical → perfect coherent sum → one razor-sharp spike at −3.5 MHz
exactly

A pure unmodulated carrier is a single frequency, so it produces a single infinitely-narrow spike. No
width at all.

What the data does to this
But carrier 5 does carry data. The QPSK phase offset jumps every 40 samples to a new constellation
value. So when the FFT de-rotates at exactly −3.5 MHz, the carrier rotation cancels but the data
jumps remain:

De-rotated samples (with QPSK data):
samples 0–39:    all point at 45°    (symbol 0)
samples 40–79:   all point at 135°   (symbol 1)  ← jumped
samples 80–119:  all point at 315°   (symbol 2)  ← jumped again
samples 120–159: all point at 45°    (symbol 3)  ← jumped again
...

The de-rotated samples are no longer all identical. They're piecewise-constant — steady within each
symbol, jumping between symbols. When you sum them, you still get a large value (most of the
coherence survives), but it's not the perfect maximum you'd get from a constant phase. The jumps
cause partial cancellation — some symbols point one way, others point another, and they don't all
reinforce perfectly.

That lost coherence didn't vanish — it got redistributed into neighboring bins.

Where the lost energy goes
Consider a bin slightly off −3.5 MHz, say −3.4 MHz. Its probe doesn't quite cancel the carrier — it
leaves a slow residual rotation of a few degrees per sample. For a pure carrier, this residual rotation
would scatter the terms and produce zero. But the data jumps happen to have frequency content at
that small offset. Some of the energy the jumps stripped from the center bin reappears here, because
the pattern of jumps partially matches this slightly-detuned probe.

The result: energy spreads from the exact center frequency into a band of bins around it. The faster
and more abruptly the data changes, the more energy spreads, and the wider the band.

The clean way to see it: modulation is multiplication
There's a precise principle underneath this. The transmitted carrier is:

$$\text{carrier 5}[n] = \underbrace{\text{data envelope}[n]}{\text{slowly varying, the symbols}} \
times \underbrace{e^{j2\pi(-3.5\text{MHz})n/f_s}}{\text{pure carrier spike}}$$

Multiplication in the time domain is convolution in the frequency domain. So the spectrum of
carrier 5 is:

$$\text{Spectrum} = \underbrace{\text{Data spectrum}}{\text{a band, ±675 kHz wide}} * \
underbrace{\delta(f + 3.5\text{MHz})}{\text{the pure carrier spike}}$$

Convolving anything with a delta function (the spike) just shifts it to the spike's location. So the
data's baseband spectrum — which spans ±675 kHz because of the RRC pulse shaping — gets
picked up and moved to sit centered on −3.5 MHz:

Data spectrum (at baseband):        Carrier 5 in the transponder:

      ┌──────┐                              ┌──────┐
      │ ±675 │                              │ ±675 │
──────┘  kHz └──────         →      ────────┘ kHz  └────────
      −0.675  +0.675                    −4.175  −2.825
         centered at 0                   centered at −3.5 MHz

The single carrier spike, convolved with the data's 1.35 MHz-wide baseband spectrum, becomes a
1.35 MHz-wide band centered at −3.5 MHz. That band is the smeared peak.

Why the band has the flat-top shape
The shape of the smear is exactly the data's baseband spectrum, which is |P(f)|² — the RRC
pulse's power spectrum. That's why each carrier on your plot has the characteristic flat top and
cosine roll-off:

The smear shape = the RRC spectrum:

  flat top    ← data phase changes don't spread energy beyond the RRC band
  roll-offs   ← the RRC pulse shaping confines the spread to ±675 kHz
  hard edges  ← RRC forces zero energy beyond (1+α)/2T from center

This is the through-line of everything we've discussed. The RRC filter controls how the data
modulation smears the carrier. Without RRC (abrupt phase jumps), the smear would have wide sinc
sidelobes leaking far beyond the slot — that's the ISI problem. With RRC, the smear is confined to
a clean 1.35 MHz band with sharp edges. The smear width is the symbol rate times (1+α), and the
smear shape is the RRC power spectrum.

Connecting back to your detector
So the peak width your detector measures isn't a measurement artifact — it's the actual occupied
bandwidth of the carrier, set by its symbol rate and RRC roll-off:

Carrier 5  (1 MSym/s):   smear width = 1.0 × 1.35 = 1.35 MHz
Carrier 4  (4 MSym/s):   smear width = 4.0 × 1.35 = 5.40 MHz
CUC        (0.3 MSym/s): smear width = 0.3 × 1.35 = 0.405 MHz

The CUC's narrow data rate produces a narrow smear (0.405 MHz), which is why it appears as a
thin bump sitting inside carrier 13's wide 4.05 MHz smear. Your wavelet edge detector finds the
boundaries of each smear (the RRC roll-offs), and the CUC's narrow smear is the anomaly you're
hunting — a small band of extra energy where carrier 13's flat top should be smooth.


