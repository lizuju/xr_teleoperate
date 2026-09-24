# Deployed WebXR client

`vuer-xr-session-fix.tar.gz` contains the deployed client assets for Vuer 0.0.60, including the Safari XR session and page-loading fixes. Source: https://github.com/vuer-ai/vuer (license in `LICENSE.vuer`). Deployment backups are excluded.

After installing the Python dependencies, run the following with the teleoperation Python environment before starting the Safari client:

```sh
python tools/install_webxr_client.py
```

This copies the versioned assets into the installed Vuer client directory. The native VisionProTeleop input does not use this browser client.
