# Download Pause/Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pause/resume button to model downloads, allowing users to interrupt and continue downloads without losing progress.

**Architecture:** Pause sets a `threading.Event` on the download session. The progress callback (which fires on every HF chunk) checks this event and raises `DownloadPausedError` to abort the blocking HF call. The `.incomplete` files are preserved. Resume re-calls the existing start endpoint which auto-detects partial files.

**Tech Stack:** Python (FastAPI, threading, huggingface_hub), TypeScript (React)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `backend/state/app_state_types.py` | Modify | Add `cancel_event` to DownloadingSession, add DownloadSessionPaused |
| `backend/api_types.py` | Modify | Add DownloadProgressPausedResponse, add PauseDownloadResponse |
| `backend/handlers/download_handler.py` | Modify | Add DownloadPausedError, pause check in callback, request_pause(), pause_download() |
| `backend/_routes/models.py` | Modify | Add POST /api/models/download/pause endpoint |
| `backend/tests/test_models.py` | Modify | Add pause/resume tests |
| `frontend/lib/api-client.ts` | Modify | Add pauseModelDownload() method |
| `frontend/components/FirstRunSetup.tsx` | Modify | Add pause/resume button to download UI |

---

### Task 1: Add State Types for Pause

**Files:**
- Modify: `backend/state/app_state_types.py:62-77`
- Modify: `backend/api_types.py:132-156`

- [ ] **Step 1: Add cancel_event and DownloadSessionPaused to state types**

In `backend/state/app_state_types.py`, add `import threading` at the top (after the existing imports), then add `from dataclasses import field` to the dataclasses import. Then modify `DownloadingSession` and add `DownloadSessionPaused`:

```python
import threading
```

Add `field` to the existing dataclasses import:
```python
from dataclasses import dataclass, field
```

Replace the `DownloadingSession` dataclass (lines 70-76):

```python
@dataclass
class DownloadingSession:
    id: DownloadSessionId
    current_running_file: FileDownloadRunning | None
    files_to_download: set[ModelFileType]
    completed_files: set[ModelFileType]
    completed_bytes: int
    cancel_event: threading.Event = field(default_factory=threading.Event)
```

Add after `DownloadSessionError` (after line 52):

```python
@dataclass(frozen=True)
class DownloadSessionPaused:
    files_to_download: frozenset[ModelFileType]
    completed_files: frozenset[ModelFileType]
    completed_bytes: int
    status: str = "paused"
```

Update `DownloadSessionResult` (line 55):

```python
DownloadSessionResult = DownloadSessionComplete | DownloadSessionError | DownloadSessionPaused
```

- [ ] **Step 2: Add DownloadProgressPausedResponse and PauseDownloadResponse to api_types**

In `backend/api_types.py`, add after `DownloadProgressErrorResponse` (after line 151):

```python
class DownloadProgressPausedResponse(BaseModel):
    status: Literal["paused"]
```

Update `DownloadProgressResponse` (lines 154-156):

```python
DownloadProgressResponse: TypeAlias = (
    DownloadProgressRunningResponse
    | DownloadProgressCompleteResponse
    | DownloadProgressErrorResponse
    | DownloadProgressPausedResponse
)
```

Add after `ClearPartialDownloadsResponse` (around line 160):

```python
class PauseDownloadResponse(BaseModel):
    status: Literal["pausing"] = "pausing"
```

- [ ] **Step 3: Run tests to verify nothing breaks**

Run:
```bash
cd /Users/deepakbhadoriya/Desktop/mlx-studio/backend && uv run --extra test pytest tests/test_models.py -v -x
```

Expected: all existing tests pass (DownloadingSession gains a default field, backward compatible)

- [ ] **Step 4: Commit**

```bash
git add backend/state/app_state_types.py backend/api_types.py
git commit -m "feat: add pause state types for download sessions

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add Pause Logic to Download Handler

**Files:**
- Modify: `backend/handlers/download_handler.py`

- [ ] **Step 1: Add DownloadPausedError and import DownloadSessionPaused**

At the top of `backend/handlers/download_handler.py`, add `DownloadSessionPaused` to the import from `state.app_state_types` (line 29-37):

```python
from state.app_state_types import (
    AppState,
    DownloadSessionComplete,
    DownloadSessionError,
    DownloadSessionId,
    DownloadSessionPaused,
    DownloadingSession,
    FileDownloadRunning,
    ModelFileType,
)
```

Add `DownloadProgressPausedResponse` and `PauseDownloadResponse` to the import from `api_types` (lines 13-18):

```python
from api_types import (
    DownloadProgressCompleteResponse,
    DownloadProgressErrorResponse,
    DownloadProgressPausedResponse,
    DownloadProgressResponse,
    DownloadProgressRunningResponse,
    PauseDownloadResponse,
)
```

Add after the `DownloadInProgressError` class (after line 47):

```python
class DownloadPausedError(Exception):
    """Raised from progress callback when user requests pause."""
```

- [ ] **Step 2: Add pause check to progress callback**

Modify `_make_progress_callback` (line 125-151). Add a pause check at the start of `on_progress`:

```python
    def _make_progress_callback(
        self, file_type: ModelFileType, initial_bytes: int = 0
    ) -> Callable[[int], None]:
        last_sample_time = time.monotonic()
        last_sample_bytes = initial_bytes
        smoothed_speed = 0.0

        def on_progress(downloaded: int) -> None:
            nonlocal last_sample_time, last_sample_bytes, smoothed_speed

            session = self.state.downloading_session
            if session is not None and session.cancel_event.is_set():
                raise DownloadPausedError("Download paused by user")

            now = time.monotonic()
            elapsed = now - last_sample_time
            if elapsed >= 1.0:
                instant_speed = (downloaded - last_sample_bytes) / elapsed
                if smoothed_speed == 0.0:
                    smoothed_speed = instant_speed
                else:
                    smoothed_speed = 0.3 * instant_speed + 0.7 * smoothed_speed
                last_sample_time = now
                last_sample_bytes = downloaded
            self.update_file_progress(file_type, downloaded, smoothed_speed)

        return on_progress
```

- [ ] **Step 3: Add request_pause() and pause_download() methods**

Add after `fail_download` (after line 123):

```python
    @with_state_lock
    def request_pause(self) -> bool:
        """Signal the download worker to pause. Returns False if no download is running."""
        session = self.state.downloading_session
        if session is None:
            return False
        session.cancel_event.set()
        return True

    @with_state_lock
    def pause_download(self) -> None:
        """Transition session from running to paused. Called by worker after catching DownloadPausedError."""
        session = self.state.downloading_session
        if session is None:
            return
        self.state.completed_download_sessions[session.id] = DownloadSessionPaused(
            files_to_download=frozenset(session.files_to_download),
            completed_files=frozenset(session.completed_files),
            completed_bytes=session.completed_bytes,
        )
        self.state.downloading_session = None
```

- [ ] **Step 4: Add DownloadPausedError handling to worker**

Modify `_download_models_worker` (lines 278-317). Wrap the download call in a try/except:

```python
    def _download_models_worker(self, files_to_download: dict[ModelFileType, str]) -> None:
        if not files_to_download:
            self.finish_download()
            return

        for file_type, target_name in files_to_download.items():
            spec = self.config.spec_for(file_type)
            logger.info("Downloading %s from %s", target_name, spec.repo_id)

            initial_bytes = self._resumable_bytes_for(file_type)
            if initial_bytes > 0:
                logger.info("Resuming %s from %d bytes", target_name, initial_bytes)
            self.start_file(file_type, target_name, initial_bytes=initial_bytes)
            progress_cb = self._make_progress_callback(file_type, initial_bytes=initial_bytes)

            resolve_downloading_dir(self.models_dir).mkdir(parents=True, exist_ok=True)

            try:
                if spec.is_folder:
                    self._model_downloader.download_snapshot(
                        repo_id=spec.repo_id,
                        local_dir=str(resolve_downloading_path(self.models_dir, self.config.model_download_specs, file_type)),
                        on_progress=progress_cb,
                        ignore_patterns=list(spec.ignore_patterns) if spec.ignore_patterns else None,
                        initial_bytes=initial_bytes,
                    )
                else:
                    self._model_downloader.download_file(
                        repo_id=spec.repo_id,
                        filename=spec.name,
                        local_dir=str(resolve_downloading_path(self.models_dir, self.config.model_download_specs, file_type)),
                        on_progress=progress_cb,
                        initial_bytes=initial_bytes,
                    )
            except DownloadPausedError:
                logger.info("Download paused by user during %s", target_name)
                self.pause_download()
                return

            self._move_to_final(file_type)

        self.finish_download()
        self._models_handler.refresh_available_files()
        shutil.rmtree(resolve_downloading_dir(self.models_dir), ignore_errors=True)
```

- [ ] **Step 5: Handle paused status in get_download_progress**

Modify `get_download_progress` (lines 192-200). Add a case for `DownloadSessionPaused` in the match block:

```python
        result = self.state.completed_download_sessions.get(typed_session_id)
        if result is not None:
            match result:
                case DownloadSessionComplete():
                    return DownloadProgressCompleteResponse(status="complete")
                case DownloadSessionError(error_message=error_message):
                    return DownloadProgressErrorResponse(status="error", error=error_message)
                case DownloadSessionPaused():
                    return DownloadProgressPausedResponse(status="paused")

        raise ValueError(f"Unknown download session: {session_id}")
```

- [ ] **Step 6: Handle DownloadPausedError in _on_background_download_error**

Modify `_on_background_download_error` (line 153-154). The `DownloadPausedError` is caught inside the worker, but if it somehow propagates to the error handler, handle it gracefully:

```python
    def _on_background_download_error(self, exc: Exception) -> None:
        if isinstance(exc, DownloadPausedError):
            self.pause_download()
            return
        self.fail_download(str(exc))
```

- [ ] **Step 7: Run tests**

Run:
```bash
cd /Users/deepakbhadoriya/Desktop/mlx-studio/backend && uv run --extra test pytest tests/test_models.py -v -x
```

Expected: all existing tests pass

- [ ] **Step 8: Commit**

```bash
git add backend/handlers/download_handler.py
git commit -m "feat: add pause/resume logic to download handler

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add Pause API Endpoint

**Files:**
- Modify: `backend/_routes/models.py`

- [ ] **Step 1: Add pause route**

In `backend/_routes/models.py`, add `PauseDownloadResponse` to the import from `api_types` (line 9-20):

```python
from api_types import (
    ClearPartialDownloadsResponse,
    DownloadProgressResponse,
    ModelDownloadRequest,
    ModelDownloadStartResponse,
    ModelInfo,
    ModelsStatusResponse,
    PauseDownloadResponse,
    RequiredModelsResponse,
    TextEncoderAlreadyDownloadedResponse,
    TextEncoderDownloadStartedResponse,
    TextEncoderDownloadResponse,
)
```

Add after the `route_clear_partial_downloads` function (after line 95):

```python
@router.post("/models/download/pause", response_model=PauseDownloadResponse)
def route_pause_download(
    handler: AppHandler = Depends(get_state_service),
) -> PauseDownloadResponse:
    """Request the running download to pause.

    The pause is asynchronous — the download worker will stop after the
    current chunk completes. Returns 409 if no download is running.
    """
    if not handler.downloads.request_pause():
        raise HTTPError(409, "No download in progress")
    return PauseDownloadResponse()
```

- [ ] **Step 2: Run tests**

Run:
```bash
cd /Users/deepakbhadoriya/Desktop/mlx-studio/backend && uv run --extra test pytest tests/test_models.py -v -x
```

Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add backend/_routes/models.py
git commit -m "feat: add POST /api/models/download/pause endpoint

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add Backend Tests for Pause

**Files:**
- Modify: `backend/tests/test_models.py`

- [ ] **Step 1: Add test for request_pause when no download running**

Add to the download test class in `backend/tests/test_models.py`:

```python
    def test_pause_returns_409_when_no_download(self, client):
        resp = client.post("/api/models/download/pause")
        assert resp.status_code == 409

    def test_pause_returns_200_when_download_running(self, client, test_state):
        test_state.downloads.start_download({"checkpoint"})
        resp = client.post("/api/models/download/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pausing"

    def test_paused_download_reports_paused_status(self, client, test_state, fake_services):
        from state.app_state_types import DownloadSessionPaused

        session_id = test_state.downloads.start_download({"checkpoint"})
        # Simulate pause completion
        test_state.downloads.pause_download()

        resp = client.get(f"/api/models/download/progress?sessionId={session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"
```

- [ ] **Step 2: Add test for progress callback raising DownloadPausedError**

```python
    def test_pause_event_interrupts_progress_callback(self, test_state):
        from handlers.download_handler import DownloadPausedError

        session_id = test_state.downloads.start_download({"checkpoint"})
        test_state.downloads.start_file("checkpoint", "test-model", initial_bytes=0)
        cb = test_state.downloads._make_progress_callback("checkpoint")

        # Before pause: callback should work
        cb(1024)

        # Set pause event
        test_state.downloads.request_pause()

        # After pause: callback should raise
        with pytest.raises(DownloadPausedError):
            cb(2048)
```

- [ ] **Step 3: Run tests**

Run:
```bash
cd /Users/deepakbhadoriya/Desktop/mlx-studio/backend && uv run --extra test pytest tests/test_models.py -v -x
```

Expected: all tests pass including new ones

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_models.py
git commit -m "test: add pause/resume download tests

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Add Frontend Pause/Resume Button

**Files:**
- Modify: `frontend/lib/api-client.ts`
- Modify: `frontend/components/FirstRunSetup.tsx`

- [ ] **Step 1: Add pauseModelDownload to API client**

In `frontend/lib/api-client.ts`, add after `clearPartialDownloads` method (after line 153):

```typescript
  static pauseModelDownload(): Promise<{ status: string }> {
    return this.requestJson('/api/models/download/pause' as any, 'post')
  }
```

- [ ] **Step 2: Add pause state and handler to FirstRunSetup**

In `frontend/components/FirstRunSetup.tsx`, add a `isPaused` state variable near the other state declarations (around line 36-38):

```typescript
  const [isPaused, setIsPaused] = useState(false)
```

Add a pause handler after `retryInstallation` (after line 198):

```typescript
  const pauseInstallation = async () => {
    try {
      await ApiClient.pauseModelDownload()
    } catch (e) {
      logger.error(`Pause error: ${e}`)
    }
  }

  const resumeInstallation = () => {
    setIsPaused(false)
    startInstallation()
  }
```

In the polling effect (around line 144), handle "paused" status:

```typescript
        if (progress.status === 'error') {
          setDownloadError(progress.error || 'Download failed.')
        } else if (progress.status === 'complete') {
```

Add before the error check:

```typescript
        if (progress.status === 'paused') {
          setIsPaused(true)
        } else if (progress.status === 'error') {
          setDownloadError(progress.error || 'Download failed.')
        } else if (progress.status === 'complete') {
```

- [ ] **Step 3: Add Pause/Resume button to download UI**

In `frontend/components/FirstRunSetup.tsx`, in the progress section (around line 670-675), add a pause/resume button next to the status text. Replace the header row:

```tsx
                {/* Header row with status, percentage, and pause button */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 8
                }}>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>
                    {isPaused ? 'Paused' : totalProgress > 85 ? 'Installing...' : 'Downloading...'}
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <span style={{ fontSize: 13, color: '#A98BD9', fontWeight: 600 }}>
                      {Math.round(totalProgress)}%
                    </span>
                    <button
                      onClick={isPaused ? resumeInstallation : pauseInstallation}
                      style={{
                        background: 'transparent',
                        border: '1px solid #444',
                        borderRadius: 4,
                        color: '#a0a0a0',
                        fontSize: 12,
                        padding: '2px 10px',
                        cursor: 'pointer',
                      }}
                    >
                      {isPaused ? 'Resume' : 'Pause'}
                    </button>
                  </div>
                </div>
```

- [ ] **Step 4: Commit**

```bash
git add frontend/lib/api-client.ts frontend/components/FirstRunSetup.tsx
git commit -m "feat: add pause/resume button to download UI

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Run Full Verification

**Files:** None (verification only)

- [ ] **Step 1: Run full backend test suite**

Run:
```bash
cd /Users/deepakbhadoriya/Desktop/mlx-studio/backend && uv run --extra test pytest tests/ -v -k 'not test_pyright'
```

Expected: all tests pass including new pause tests

- [ ] **Step 2: Run TypeScript typecheck**

Run:
```bash
cd /Users/deepakbhadoriya/Desktop/mlx-studio && pnpm typecheck:ts
```

Expected: no new errors from the frontend changes

- [ ] **Step 3: Verify no stale references**

Run:
```bash
grep -r "DownloadPausedError\|DownloadSessionPaused\|request_pause\|pause_download\|pauseModelDownload" /Users/deepakbhadoriya/Desktop/mlx-studio/backend /Users/deepakbhadoriya/Desktop/mlx-studio/frontend --include="*.py" --include="*.ts" --include="*.tsx" -l
```

Expected: files from our changes only — no orphaned references

- [ ] **Step 4: Commit if any fixes needed**

```bash
git add -A
git commit -m "chore: verify pause/resume implementation

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
