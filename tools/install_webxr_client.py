from importlib.metadata import distribution
from pathlib import Path
import tarfile


if __name__ == "__main__":
    bundle = Path(__file__).resolve().parents[1] / "vendor/vuer-xr-session-fix.tar.gz"
    target = Path(distribution("vuer").locate_file("vuer/client_build/assets"))
    with tarfile.open(bundle) as archive:
        archive.extractall(target, filter="data")
    print(f"Installed WebXR client: {target / 'xr-session-fix'}")
