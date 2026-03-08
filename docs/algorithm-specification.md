# Algorithm Specification

## Purpose

This document specifies the reference image-processing pipeline implemented in [reference/tescan_digital_resolution.py](../reference/tescan_digital_resolution.py).

Its purpose is to preserve the observable behavior of the original Python prototype while the production pipeline is reimplemented in C++ and CUDA.

## Pipeline Overview

The reference pipeline processes a single grayscale image using four stages:

1. Non-local means denoising
2. Unsharp masking
3. Richardson-Lucy deconvolution
4. Histogram stretch

The Python entrypoint is:

```python
def tescan_digital_resolution_filter(img):
    img = non_local_means(img)
    img = unsharp_masking(img)
    img = richardson_lucy_deconvolution(img)
    img = histogram_stretch(img)
    return img
```

The C++ pipeline must expose the same logical stage ordering and the same stable stage identifiers used for validation dumps.

## Data Contract

- Input images are grayscale only.
- The production C++ pipeline uses `ImageF32` with normalized values in the range `[0, 1]`.
- The pipeline supports uploading source `uint8` or `uint16` grayscale pixels and normalizing them on the GPU.
- The normalized `ImageF32` overload is available for tests and in-memory callers, but the CLI path should prefer raw grayscale input.
- Intermediate validation dumps should use these stable stage names:

```text
00_input_normalized
10_non_local_means
20_unsharp_mask
30_richardson_lucy
40_histogram_stretch
90_output
```

- The `name` field is the stable identifier. Numeric prefixes exist only to keep dump files sorted.

## Stage 0: Input Normalization

### Python behavior

The Python prototype reads images with OpenCV grayscale loading and then performs stage-specific conversions internally.

- `non_local_means` converts the input with `img_as_float`, producing floating-point values in `[0, 1]`.
- `richardson_lucy_deconvolution` expects a 16-bit image-like representation and renormalizes by dividing by `65535.0`.
- `histogram_stretch` expects integer input, either `uint8` or `uint16`.

### C++/CUDA contract

The C++ pipeline should normalize the source image once at the pipeline boundary and keep the working representation in normalized floating point where practical. When a stage requires a different numeric convention internally, that conversion should be local to the stage implementation.

The current implementation accepts raw grayscale images at the pipeline boundary, uploads them once, expands borders on the GPU, and keeps device-resident working buffers until stage capture or final output download.

## Stage 1: Non-local Means

### Python implementation

```python
def non_local_means(img):
    img_float = img_as_float(img)

    h = 7
    templ = 3
    search = 3

    return denoise_nl_means(
        img_float,
        h=h / 255.0,
        patch_size=templ,
        patch_distance=search // 2,
        fast_mode=True,
        channel_axis=None
    )
```

### Parameters

- Filtering strength `h = 7 / 255`
- Patch size `3`
- Search window radius `1` because `3 // 2 == 1`
- Fast mode enabled
- Single-channel grayscale processing

### Required behavior

- Reduce high-frequency noise while preserving edges.
- Accept normalized grayscale input.
- Produce normalized floating-point output.

## Stage 2: Unsharp Mask

### Python implementation

```python
def unsharp_masking(img):
    amount = 0.6
    return filters.unsharp_mask(img, radius=5, amount=amount)
```

### Parameters

- Blur radius `5`
- Sharpening amount `0.6`

### Required behavior

- Enhance local contrast after denoising.
- Preserve image dimensions.
- Continue using floating-point grayscale output.

## Stage 3: Richardson-Lucy Deconvolution

### Python implementation summary

The Python implementation pads the image with replicated borders, optionally applies an edge taper window, builds a Gaussian point spread function, runs two Richardson-Lucy iterations in the frequency domain using TensorFlow FFT primitives, clamps negative values with ReLU, rescales the result to the full integer output range, and crops the replicated border.

### Parameters and sub-steps

- Iterations: `2`
- Border size: `max(0.2 * height, 0.2 * width)`
- Border mode: replicate
- Edge taper: enabled by default with `alpha = 0.2`
- PSF: centered Gaussian
- Gaussian sigma: `(2.5, 2.5)`
- PSF size: `(15, 15)`
- PSF normalization: peak normalized
- Numerical epsilon: `1e-7`
- Negative values clamped to `0`

### Detailed reference behavior

1. Expand the image using replicated borders.
2. Convert padded input to `float32` and normalize to `[0, 1]` by dividing by `65535.0`.
3. Optionally apply `edgetaper2d`.
4. Build a full-image PSF by embedding a `15 x 15` Gaussian kernel into a zero image of the padded size.
5. Apply `ifftshift` so the PSF is aligned for FFT convolution.
6. Compute the OTF with a 2D FFT.
7. Initialize the Richardson-Lucy estimate with the observed image.
8. Repeat twice:
   - Convolve current estimate with the OTF.
   - Compute `ratio = observed / (conv + eps)`.
   - Correlate that ratio with the conjugate OTF.
   - Multiply the estimate by the correlation result.
   - Clamp negative values to zero.
9. Rescale the final estimate to the full dynamic range using the image min and max.
10. Crop away the replicated border.

### Output convention

The Python function returns integer output, defaulting to `uint16`, after min-max rescaling. In the C++ pipeline, the stage should conceptually represent the Richardson-Lucy result before final file encoding. Any temporary integer conversion required to match the reference should stay internal to the stage implementation.

## Stage 4: Histogram Stretch

### Python implementation

```python
def histogram_stretch(img, sat_percent=0.5):
    low_val = np.percentile(img_float, sat_percent)
    high_val = np.percentile(img_float, 100 - sat_percent)
    stretched = np.clip((img_float - low_val) * max_val / (high_val - low_val), 0, max_val)
    return stretched.astype(out_dtype)
```

### Parameters

- Saturation percent `0.5`
- Low percentile `0.5`
- High percentile `99.5`

### Required behavior

- Stretch contrast after deconvolution.
- Clip values outside the retained percentile range.
- Match the reference percentile thresholds.
- Preserve full output dimensions.

## I/O and Validation Requirements

- The public pipeline entrypoint should remain a single orchestrator function.
- GPU kernels should implement stage computation.
- Stage logging and image dumps should stay on the host side.
- When intermediate capture is disabled, the pipeline should avoid unnecessary device-to-host copies.
- When intermediate capture is enabled, the pipeline should copy selected stage outputs back to the host using the stable stage identifiers above.

## C++/CUDA Structure

The intended production structure is:

1. Normalize input on the host and upload it once.
2. Execute each algorithmic stage on the GPU.
3. Keep the fast path fully device-resident until final output download.
4. Perform optional stage capture as explicit host-side debug copies.
5. Return the final image and any requested captured stages.

This separation keeps the algorithm single-sourced while preserving a fast non-debug execution path.

## Known Gaps Between Prototype and Production

- The Python prototype mixes floating-point and integer conventions between stages.
- The production C++ pipeline should use a cleaner internal numeric model while preserving externally visible behavior.
- The current C++ implementation may keep some stages as placeholders until their CUDA kernels are implemented.
- Exact numerical parity with scikit-image and TensorFlow may require tolerance-based validation rather than bit-exact comparison.