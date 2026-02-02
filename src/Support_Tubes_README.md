# Support Tubes: Design and Rationale

### What is a support tube?

A support tube is a population-level geometric envelope that summarizes where the bulk of trajectories lie in embedding space as a function of time. Rather than plotting thousands of individual sample paths, the support tube provides a compact, interpretable representation of population structure through time.

Visually, a support tube answers the question:

“At each point in time, where does this population mostly live in embedding space?”

The tube is rendered as a continuous 3D surface in
(X, Y, Time) space, where: X, Y are embedding coordinates; Time is plotted along a log-scaled axis.

The construction proceeds in four conceptual phases:

⸻

### Phase A — Time slicing and density estimation

Time is transformed as:

$u = \log_{10}(\max(t, \varepsilon))$

and discretized into a fixed number of u-slices.
For each slice:

	1.	Sample points belonging to the population are projected into XY space.
	2.	A 2D histogram is computed on a fixed global grid.
	3.	The histogram is smoothed using a Gaussian kernel (sigma_xy) to reduce sampling noise.

This yields a smooth estimate of spatial density for each time slice.

⸻

### Phase B — Mass-fraction support masks

Instead of using a fixed density threshold, each slice is converted into a support mask using a mass-fraction criterion:

The support mask is the smallest set of grid cells whose total density contains a specified fraction of the population mass (e.g. 90–95%).

This approach:

	•	adapts naturally to changes in population spread,
	•	avoids arbitrary density cutoffs,
	•	ensures the tube consistently represents the “core” of the population.

The result is a boolean mask per time slice indicating supported regions in XY.

⸻

### Phase C — Temporal continuity and smoothing

Raw per-slice masks can be sparse or discontinuous due to:

	•	uneven sampling across time,
	•	abrupt demographic events (e.g. migration pulses).

To address this:

	•	u-gap filling (fill_u_gaps) ensures that the support is present for the full time range by copying the nearest non-empty slice into empty slices.
	•	Optional Gaussian smoothing along the u-axis (sigma_u) regularizes small fluctuations.
	•	Optional morphological closing can be applied to improve spatial coherence.

At this stage, we have a continuous 3D scalar field representing population support across time.

⸻

### Phase D — Surface extraction (support tube)

The final support tube is extracted by:

	1.	Thresholding the scalar field (tau) to obtain a 3D boolean volume.
	2.	Running marching cubes to extract a triangulated surface.
	3.	Mapping voxel indices back to real coordinates:
	-	X/Y via grid bin centers,
	- Time via t = 10^u.

The resulting mesh is rendered as a semi-transparent 3D surface — the support tube.

⸻

Interpretation notes

	-	The tube is not a convex hull of trajectories.
	-	It is not guaranteed to be smooth or continuous under sudden demographic shifts.
	-	Abrupt jumps in the tube reflect genuine structural discontinuities in the data (e.g. instantaneous migration events).

Current work focuses on improving how such discontinuities are bridged in a visually meaningful but conservative way.

⸻

Running the support tube visualization

Below are example commands using the current sandbox setup.

Basic run with support tubes and samples

```
python sandbox/convex_hull_sandbox.py \
  --outputs_dir results/simulations/outputs/split \
  --out test.html \
  --show_samples \
  --show_tube \
  --grid_res_xy 100 \
  --grid_res_u 50 \
  --mass_frac 0.95 \
  --sigma_xy 3
```

⸻

Notes on key parameters

	-	grid_res_xy: XY grid resolution (lower values often look cleaner).
	-	grid_res_u: number of time slices (increase for complex histories).
	-	mass_frac: fraction of population mass captured by the tube.
	-	sigma_xy: spatial smoothing (larger → smoother tubes).
	-	sigma_u: temporal smoothing along time.
	-	tau: final surface threshold (usually leave at 0.5).
