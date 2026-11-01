Public documentation of commercial denoisers gives us several important clues.
Component	What is publicly established
Noise profiling	Builds a statistical profile of the noise characteristics of the source footage
Temporal denoising	Compares information across consecutive frames
Spatial denoising	Analyses pixels within individual frames
Frequency separation	Separates image information into five spatial-frequency bands
Colour separation	Processes luminance (Y) and chrominance (Cr/Cb) independently
Adaptive filtration	Adjusts filtering to accommodate changing noise characteristics
Artifact removal	Separate processing for compression-related artifacts

2. Which actual algorithms should we investigate?
I would not assume a commercial denoiser uses any one of these. Instead, I'd investigate them as candidate implementations and compare their results.
Technique	What it does	Priority
Wavelet decomposition	Separates image detail at different scales	Very high
Non-local means (NLM)	Finds similar image patches and averages them intelligently	Very high
BM3D	Groups similar patches and filters them in a transform domain	Very high
Motion-compensated temporal filtering	Uses neighbouring frames while accounting for movement	Very high
Wiener filtering	Uses statistical estimates to suppress noise while preserving signal	High
Robust statistical estimation	Distinguishes outliers and noise from actual image structures	High
Frequency-domain filtering	Analyses and suppresses noise according to its spectral characteristics	High

A few particularly relevant references:
- BM3D: One of the most important classical denoising techniques. Its video extension, VBM3D, is particularly interesting. 
  arXiv
- Non-local means: Particularly interesting for retaining repetitive textures and fine detail. 
  Springer Nature Link
- VBM4D: A sophisticated approach exploiting spatial and temporal redundancy. 
  PubMed
My suspicion is that a combination of multiscale spatial filtering, statistical noise modelling and sophisticated temporal processing is a sensible direction to investigate. We should not assume a commercial denoiser uses BM3D or VBM3D, though.
3. The noise profile is probably the most important part
This is where I'd spend considerable research effort.

A simple denoiser assumes noise is relatively uniform. Real VFX footage is much more complicated.

For example:
- Sensor noise changes with exposure and ISO.
- Chroma noise behaves differently from luminance noise.
- Film grain has spatial and temporal correlations.
- Compression introduces structured noise.
- Demosaicing can introduce correlated patterns.
- Noise characteristics can vary across image intensity.
I'd want our denoiser to estimate a noise model rather than simply apply a generic noise reduction filter.

One possible mathematical representation is:
\[
I(x,y,t)=S(x,y,t)+N(x,y,t)
\]

Where:
- \(I\) is the observed image.
- \(S\) is the underlying signal.
- \(N\) is the noise.
We want to estimate the statistical properties of \(N\), including its variance, frequency distribution, colour correlations and temporal correlations.

Commercial profiling systems are explicitly designed around this principle.