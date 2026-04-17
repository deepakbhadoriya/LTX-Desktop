# Download Pause/Resume Button

## Problem

Downloads cannot be paused while running. The only way to stop a download is to kill the app. The existing auto-resume infrastructure (`.incomplete` files) works on restart, but there's no user-facing control.

## Solution

Add a pause/resume button to the download UI. Pause works by raising an exception from inside the HuggingFace tqdm progress callback, which aborts the blocking `hf_hub_download()` / `snapshot_download()` call. The `.incomplete` files are preserved automatically. Resume re-calls the existing start endpoint, which detects partial files and resumes.

16 parallel connections for snapshot downloads are already configured (`DOWNLOAD_MAX_WORKERS = 16` in `hugging_face_downloader.py:27`).

## Files to Change

### Backend

**`backend/state/app_state_types.py`** — Add `cancel_event` to `DownloadingSession`:
```python
@dataclass
class DownloadingSession:
    id: DownloadSessionId
    current_running_file: FileDownloadRunning | None
    files_to_download: set[ModelFileType]
    completed_files: set[ModelFileType]
    completed_bytes: int
    cancel_event: threading.Event = field(default_factory=threading.Event)  # NEW
```

**`backend/handlers/download_handler.py`** — Three changes:

1. Add `DownloadPausedError` exception class (module level):
```python
class DownloadPausedError(Exception):
    """Raised from progress callback when user requests pause."""
```

2. Modify `_make_progress_callback` to check `cancel_event` on every chunk:
```python
def _make_progress_callback(self, file_type, initial_bytes=0):
    ...
    def on_progress(downloaded):
        # Check pause FIRST, before any state update
        session = self.state.downloading_session
        if session is not None and session.cancel_event.is_set():
            raise DownloadPausedError("Download paused by user")
        ...existing speed/progress logic...
    return on_progress
```

3. Modify `_download_models_worker` to catch `DownloadPausedError`:
```python
def _download_models_worker(self, files_to_download):
    ...
    for file_type, target_name in files_to_download.items():
        ...
        try:
            if spec.is_folder:
                self._model_downloader.download_snapshot(...)
            else:
                self._model_downloader.download_file(...)
        except DownloadPausedError:
            self.pause_download()  # NEW method — sets session to paused state
            return  # Exit worker, .incomplete files preserved
        
        self._move_to_final(file_type)
    
    self.finish_download()
    ...
```

4. Add `pause_download()` method:
```python
@with_state_lock
def pause_download(self) -> None:
    session = self.state.downloading_session
    if session is None:
        return
    self.state.completed_download_sessions[session.id] = DownloadSessionPaused(
        files_to_download=session.files_to_download,
        completed_files=session.completed_files,
        completed_bytes=session.completed_bytes,
    )
    self.state.downloading_session = None
```

5. Add `request_pause()` method (called by route):
```python
@with_state_lock
def request_pause(self) -> bool:
    session = self.state.downloading_session
    if session is None:
        return False
    session.cancel_event.set()
    return True
```

**`backend/state/app_state_types.py`** — Add `DownloadSessionPaused`:
```python
@dataclass(frozen=True)
class DownloadSessionPaused:
    files_to_download: set[ModelFileType]
    completed_files: set[ModelFileType]
    completed_bytes: int
    status: str = "paused"

DownloadSessionResult = DownloadSessionComplete | DownloadSessionError | DownloadSessionPaused
```

**`backend/api_types.py`** — Add paused response type:
```python
class DownloadProgressPausedResponse(BaseModel):
    status: Literal["paused"]

DownloadProgressResponse: TypeAlias = (
    DownloadProgressRunningResponse
    | DownloadProgressCompleteResponse
    | DownloadProgressErrorResponse
    | DownloadProgressPausedResponse
)
```

**`backend/_routes/models.py`** — Add pause endpoint:
```python
@router.post("/models/download/pause")
def route_pause_download(handler=Depends(get_state_service)):
    if not handler.downloads.request_pause():
        raise HTTPError(409, "No download in progress")
    return {"status": "pausing"}
```

**`backend/handlers/download_handler.py`** — Modify `get_download_progress()` to return paused status:
In the section that checks `completed_download_sessions`, handle `DownloadSessionPaused`:
```python
if isinstance(result, DownloadSessionPaused):
    return DownloadProgressPausedResponse(status="paused")
```

### Frontend

**`frontend/lib/api-client.ts`** — Add pause method:
```typescript
static pauseModelDownload(): Promise<...> {
    return this.requestJson('/api/models/download/pause', 'post')
}
```

**`frontend/generated/backend-openapi.json`** and **`frontend/generated/backend-openapi.ts`** — Add pause endpoint schema (regenerated from backend).

**`frontend/components/FirstRunSetup.tsx`** — Add pause/resume button:

During download (in the progress section around line 670-740):
- Add a "Pause" button next to the progress bar
- When clicked, calls `ApiClient.pauseModelDownload()`
- When polling returns `status="paused"`, show "Resume" button instead
- Resume button calls the existing `startModelDownload()` flow (which auto-detects `.incomplete` files)

## Resume Flow

Resume uses the existing download start flow unchanged:
1. User clicks "Resume"
2. Frontend calls `POST /api/models/download` with the same model types
3. `start_model_download()` runs `_discover_files_to_download()` — already-completed files are skipped (they were moved to final location before pause)
4. `_download_models_worker()` calls `_resumable_bytes_for()` — detects `.incomplete` files
5. Download continues from where it paused

## What Stays the Same

- 16 parallel connections (`DOWNLOAD_MAX_WORKERS = 16`) — already configured
- Auto-resume on app restart — unchanged
- Clear partials endpoint — unchanged
- Progress polling interval (500ms) — unchanged
- EWMA speed calculation — unchanged

## Testing

- Existing tests use `FakeModelDownloader` which doesn't support cancellation — no test changes needed for existing tests
- Add a new test: start download, set cancel_event, verify `DownloadPausedError` is caught and session transitions to paused state
- Add a new test: verify `request_pause()` returns False when no download is running
- Add a new test: verify progress endpoint returns "paused" after pause
