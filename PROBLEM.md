# VAE Decoder Crash: AttentionBlock receives 0D tensor

## Crash Location

`TuberiaCatalog/Decoders/SDXLVAEModel.swift:124`

```
Fatal error: Array subscript out of range
```

Stack (abbreviated):
```
AttentionBlock.callAsFunction(_:)
  let shape = x.shape  // returns [] — 0D tensor
  let b = shape[0]     // CRASH
```

Full frame context:
```swift
func callAsFunction(_ x: MLXArray) -> MLXArray {
    let shape = x.shape  // [B, H, W, C] expected, but [] (0D) arrives
    let b = shape[0]     // line 124 — fatal: index out of bounds
    let hw = shape[1] * shape[2]
    let c = shape[3]
    ...
}
```

## What Has Already Been Fixed (SwiftTuberia 0.2.6)

`ResnetBlock2D.callAsFunction` had three `MLXNN.silu(h)` calls replaced with `h * MLX.sigmoid(h)`:

```swift
// BEFORE (broken):
h = norm1(h); h = MLXNN.silu(h); h = conv1(h)
h = norm2(h); h = MLXNN.silu(h); h = conv2(h)

// AFTER (fixed in 0.2.6):
h = norm1(h); h = h * MLX.sigmoid(h); h = conv1(h)
h = norm2(h); h = h * MLX.sigmoid(h); h = conv2(h)
```

`SDXLVAEDecoderModel.callAsFunction` also had `h = MLXNN.silu(h)` → `h = h * MLX.sigmoid(h)`.

## Why the Crash Still Happens

Despite the silu fix, the crash persists at the same line (confirmed via new binary with local SwiftVinetas — different frame addresses, same crash site). This means a **different upstream operation** is producing a 0D tensor that flows into `AttentionBlock`.

## Root Cause: `MLX.compile(shapeless: true)`

`MLXNN.silu` delegates to `compiledSilu`, which calls `MLX.compile(shapeless: true)`. Under memory pressure, `compile(shapeless: true)` can return an empty `[MLXArray]` from `compileState.call([a])`, and subscripting `[0]` into that crashes.

**Key insight**: `MLX.compile(shapeless: true)` is used by many MLX activations, not just silu. Any activation using `compile(shapeless: true)` in the path to `AttentionBlock` is a suspect.

## Files to Investigate

### `SDXLVAEModel.swift`

The call graph from `SDXLVAEDecoderModel` down to `AttentionBlock`:
1. `SDXLVAEDecoderModel.callAsFunction` — calls `midBlock` and upsample blocks
2. `UNetMidBlock2D.callAsFunction` — calls `ResnetBlock2D` and `AttentionBlock` alternately
3. `AttentionBlock.callAsFunction` — **crash site**

Callers of `AttentionBlock` pass tensors that have gone through:
- `GroupNorm` (with affine scale/shift) — calls `pytorchGroupNorm` → `MLXFast.layerNorm` (SAFE, C call, no compile)
- `Conv2d` — confirmed NOT using `compile()` in standard mlx-swift
- `ResnetBlock2D` — FIXED in 0.2.6

### What to look for

1. **Any remaining `MLXNN.silu` calls** in `SDXLVAEModel.swift` — search the entire file
2. **Any other activation functions using `compile(shapeless: true)`** in the path (gelu, mish, etc.)
3. **`UNetMidBlock2D.callAsFunction`** — check its full implementation; does it call any other compiled activations between ResnetBlock and AttentionBlock?
4. **The tensor just before `AttentionBlock`** — what operation produces `x` that is passed to `AttentionBlock.callAsFunction`?

## Hypothesis

There is at least one more `MLXNN.*` activation call (or another MLX compiled function) in the path between `SDXLVAEDecoderModel` and `AttentionBlock` that was not replaced in the 0.2.6 fix. This call returns `[]` under memory pressure, and that empty-shaped tensor flows into `AttentionBlock`.

## Approach to Fix

1. Read all of `SDXLVAEModel.swift` end-to-end
2. Find every call to any `MLXNN.*` function (silu, gelu, mish, etc.) — these all use `compile(shapeless: true)` in mlx-swift
3. Replace each with the direct math equivalent:
   - `silu(x)` → `x * MLX.sigmoid(x)`
   - `gelu(x)` → `x * 0.5 * (1 + MLX.tanh(...))`
   - etc.
4. Alternatively: override any remaining compiled activations in the VAE decoder to use un-compiled equivalents

## State of Local Dependencies

- **SwiftTuberia**: local at `../SwiftTuberia`, on `development` branch
- **SwiftVinetas**: local at `../SwiftVinetas`, on `development` branch, already references `../SwiftTuberia` in Package.swift
- **Vinetas.xcodeproj**: already references `../SwiftVinetas` as local package

## To Start Working

```bash
cd /Users/stovak/Projects/SwiftTuberia
git checkout development
# Open SDXLVAEModel.swift and search for all MLXNN.* calls
# Also check UNetMidBlock2D and any other blocks in the VAE decoder path
```
