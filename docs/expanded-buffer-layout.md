# Expanded Buffer Layout

## Purpose

The CUDA pipeline stores working images in expanded device buffers with a replicated border around the visible image.

## Core Idea

If the visible image has dimensions:

- `width`
- `height`

and the pipeline border width is:

- `border`

then the expanded working buffer dimensions are:

```text
expanded_width = width + 2 * border
expanded_height = height + 2 * border
```

The expanded buffer uses row-major storage.

That means pixel `(x, y)` is stored at:

```text
index = y * expanded_width + x
```

with `x` increasing across columns and `y` increasing across rows.

## How Borders Are Filled

Every expanded output pixel `(x, y)` maps back to a source pixel by reflection:

```text
period = 2 * (width - 1)       // for horizontal
modulo = abs(x - border) % period
src_x = modulo < width ? modulo : period - modulo

period = 2 * (height - 1)      // for vertical
modulo = abs(y - border) % period
src_y = modulo < height ? modulo : period - modulo
```

This creates symmetric (mirrored) borders where edge pixels are reflected back into the image domain.

Example with `width=4`, `border=2`:
- Expanded indices: `-2, -1, 0, 1, 2, 3, 4, 5, ...`
- Maps to source:   `2,  1, 0, 1, 2, 3, 2, 1, ...`

## Why Reflection

Reflection padding (symmetric mirroring) is chosen over clamping for several reasons:

1. **Algorithm Compatibility**: Works well with Richardson-Lucy deconvolution and edge-sensitive algorithms
2. **Artifact Reduction**: Avoids sharp discontinuities at boundaries that clamping introduces
3. **Perceptual Quality**: Produces smoother transitions at image edges during processing
4. **Mathematical Properties**: Maintains important symmetries needed for iterative algorithms like Richardson-Lucy

## Access Rules In Kernels

### Pixel in the expanded buffer

```cpp
std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(expanded_width) + static_cast<std::size_t>(x);
```

### Visible image pixel `(x, y)` inside an expanded buffer

```cpp
std::size_t index = static_cast<std::size_t>(border + y) * static_cast<std::size_t>(expanded_width) + static_cast<std::size_t>(border + x);
```

## Stage Buffers

2D float images with:

- width `expanded_width`
- height `expanded_height`
- row stride `expanded_width`

Visible source pixel `(x, y)`:

```text
(border + y) * expanded_width + (border + x)
```

Expanded-buffer pixel `(x, y)`:

```text
y * expanded_width + x
```