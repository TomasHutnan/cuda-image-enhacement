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

Every expanded output pixel `(x, y)` maps back to a source pixel by clamping:

```text
src_x = clamp(x - border, 0, width - 1)
src_y = clamp(y - border, 0, height - 1)
```

This creates replicated borders.

## Why Replication

Replication matches the Python reference and is a good fit for the Richardson-Lucy stage.

It avoids the hard dark edge introduced by zero padding, avoids the wraparound artifact implied by periodic repetition, and does not invent mirrored structures outside the image.

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