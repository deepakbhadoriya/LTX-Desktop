# Switch MLX Video Library: Blaizzy/mlx-video to dgrauet/ltx-2-mlx

## Problem

The current MLX inference library (`Blaizzy/mlx-video`) does not support quantized model weights for LTX models. The bf16 checkpoint from `dgrauet/ltx-2.3-mlx` is ~56 GB, which leaves only ~8 GB on a 64 GB Mac — not enough for text encoder, OS, and inference overhead.

The `dgrauet/ltx-2-mlx` library has built-in Q8/Q4 quantization support and was designed specifically for the `dgrauet/ltx-2.3-mlx-q8` model weights (~37 GB working memory for distilled, fitting comfortably on 64 GB).

## Solution

Replace `Blaizzy/mlx-video` with `dgrauet/ltx-2-mlx` (`ltx-core-mlx` + `ltx-pipelines-mlx` packages). Each MLX pipeline class becomes a thin adapter wrapping the corresponding dgrauet pipeline class. Pipelines persist across generate calls with `low_memory=True` (the library loads/frees components per inference stage internally).

## Files to Change

### 1. `backend/pyproject.toml`

Remove:
```
"mlx-video; sys_platform == 'darwin'",
```
```
mlx-video = { git = "https://github.com/Blaizzy/mlx-video.git" }
```

Add:
```
"ltx-core-mlx; sys_platform == 'darwin'",
"ltx-pipelines-mlx; sys_platform == 'darwin'",
```
```
ltx-core-mlx = { git = "https://github.com/dgrauet/ltx-2-mlx.git", subdirectory = "packages/ltx-core-mlx" }
ltx-pipelines-mlx = { git = "https://github.com/dgrauet/ltx-2-mlx.git", subdirectory = "packages/ltx-pipelines-mlx" }
```

Also bump `mlx>=0.31.0` (dgrauet requires 0.31+).

### 2. `backend/runtime_config/model_download_specs.py`

Update MLX checkpoint spec:
- `repo_id`: `dgrauet/ltx-2.3-mlx` -> `dgrauet/ltx-2.3-mlx-q8`
- `expected_size_bytes`: 56 GB -> 38 GB (excluding dev transformer)
- `ignore_patterns`: add `transformer-dev.safetensors` (20.6 GB, not needed for distilled mode)
- Update upsampler `expected_size_bytes` to match Q8 repo (996 MB, same)

### 3. `backend/services/fast_video_pipeline/mlx_video_pipeline.py`

Rewrite to wrap dgrauet pipelines:

```python
class MLXVideoPipeline:
    def __init__(self, checkpoint_path, gemma_root, upsampler_path):
        self._model_dir = <resolve to checkpoint directory>
        self._gemma_repo = <resolve text encoder path or default>
        self._t2v_pipeline = None  # lazy
        self._i2v_pipeline = None  # lazy

    def generate(self, prompt, seed, height, width, num_frames, frame_rate, images, output_path):
        if images:
            pipeline = self._get_i2v_pipeline()
            pipeline.generate_and_save(prompt, output_path, image=images[0].path, ...)
        else:
            pipeline = self._get_t2v_pipeline()
            pipeline.generate_and_save(prompt, output_path, ...)
```

- Lazy-load pipeline on first use via `_get_t2v_pipeline()` / `_get_i2v_pipeline()`
- Pipeline `.load()` called once, persists across calls
- `low_memory=True` so the library manages component lifecycle internally

### 4. `backend/services/a2v_pipeline/mlx_a2v_pipeline.py`

Rewrite to wrap `AudioToVideoPipeline`:

```python
class MLXA2VPipeline:
    def __init__(self, checkpoint_path, gemma_root, upsampler_path):
        self._pipeline = None  # lazy

    def generate(self, prompt, ..., audio_path, output_path):
        pipeline = self._get_pipeline()
        pipeline.generate_and_save(prompt, output_path, audio_path=audio_path, ...)
```

### 5. `backend/services/retake_pipeline/mlx_retake_pipeline.py`

Rewrite to wrap dgrauet's `RetakePipeline`:

```python
class MLXRetakePipeline:
    def __init__(self, checkpoint_path, gemma_root):
        self._pipeline = None  # lazy

    def generate(self, *, video_path, prompt, start_time, end_time, seed, output_path, ...):
        pipeline = self._get_pipeline()
        # Convert time range to frame range
        start_frame = int(start_time * 24)
        end_frame = int(end_time * 24)
        pipeline.retake_from_video(prompt, video_path, start_frame, end_frame, seed=seed, ...)
```

### 6. `backend/services/ic_lora_pipeline/mlx_ic_lora_pipeline.py`

Rewrite to wrap dgrauet's `ICLoraPipeline`:

```python
class MLXIcLoraPipeline:
    def __init__(self, checkpoint_path, gemma_root, upsampler_path, lora_path):
        self._lora_path = lora_path
        self._pipeline = None  # lazy

    def generate(self, prompt, seed, height, width, num_frames, frame_rate, images, video_conditioning, output_path):
        pipeline = self._get_pipeline()
        # Pass lora_paths and video_conditioning to dgrauet's IC-LoRA pipeline
```

### 7. No changes needed

- `backend/services/text_encoder/mlx_text_encoder.py` — remains a no-op stub (dgrauet handles text encoding internally)
- `backend/app_handler.py` — no changes, same class names imported
- `backend/services/patches/safetensors_metadata_fix.py` — only used on non-Darwin (CUDA path), unaffected
- Protocol files — no changes, adapters conform to existing Protocols

## Model Memory Budget (64 GB Mac)

| Component | Q8 Size |
|-----------|---------|
| Transformer (distilled) | 20.6 GB |
| Connector | 6.34 GB |
| LoRA | 7.61 GB |
| VAE decoder | 0.8 GB |
| VAE encoder | 0.6 GB |
| Upsampler | 1.0 GB |
| Text encoder (Gemma Q4) | ~7 GB |
| **Peak loaded** | **~37 GB** |
| OS + app overhead | ~5 GB |
| **Available for inference** | **~22 GB** |

With `low_memory=True`, text encoder is freed before transformer loads, so actual peak is lower than the sum.

## Testing

Existing tests use fake services via `ServiceBundle` — they don't import the real MLX pipelines. No test changes needed. The fakes already conform to the Protocol interfaces.

Run `pnpm backend:test` and `pnpm typecheck:py` to verify nothing breaks.

## Future Work

- Download `transformer-dev.safetensors` to enable CFG-based retake/two-stage pipelines
- Add progress callbacks (dgrauet uses tqdm internally, no callback hooks yet)
- Upgrade to official quantized LTX 2.3 MLX models when available from mlx-community/Blaizzy
