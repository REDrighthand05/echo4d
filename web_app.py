"""ECHO Web Application facade for backward compatibility and PyInstaller bootstrap."""

import sys
from pathlib import Path

# Legacy and public interface re-exports
from echo_desktop import (
    Bridge,
    install_resize_hook,
    launch_desktop,
    maximize_work_area,
    polish_and_hook,
    polish_window,
    resize_hook_status,
)
from echo_desktop.win32_drop import (
    drop_files_to_paths,
    handle_drop_files,
    install_drop_accept,
)
from echo_desktop.win32_hook import real_win32_facade
from echo_desktop.win32_types import (
    _HTCLIENT,
    hit_test_resize,
    signed_short,
    window_frame_rects,
)
from echo_desktop.win32_wndproc import (
    ensure_maximize_styles,
    resize_wndproc,
)
from echo_engine import (
    PLAIN_STATUS,
    STAGE_COUNT,
    STAGE_ORDER,
    job_finished,
    list_previous_jobs,
    play_file,
    replay_available,
    replay_file,
    run_depth_video,
    run_pipeline,
    scale_status_line,
    scene_file,
)
from echo_service import (
    JobManager,
    ThrottledCache,
    check_video,
    export_voxel_model,
    is_png_complete,
)

# Aliases preserving legacy internal naming conventions
_polish_and_hook = polish_and_hook
_polish_window = polish_window
_install_resize_hook = install_resize_hook
_install_drop_accept = install_drop_accept
_resize_wndproc = resize_wndproc
_resize_hook_status = resize_hook_status
_hit_test_resize = hit_test_resize
_signed_short = signed_short
_window_frame_rects = window_frame_rects
_ensure_maximize_styles = ensure_maximize_styles
_real_win32_facade = real_win32_facade
_png_complete = is_png_complete
_drop_files_to_paths = drop_files_to_paths
_handle_drop_files = handle_drop_files


def main():
    """Bootstrap the desktop application window via PyWebview."""
    launch_desktop()


if __name__ == "__main__":
    main()
