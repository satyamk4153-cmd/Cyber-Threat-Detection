"""
Packaging script to generate the final verified ZIP archive for academic submission.
Archive name: Intelligent_Cyber_Threat_Detection_NPTEL.zip
"""

import os
import zipfile
from pathlib import Path

ARCHIVE_NAME = "Intelligent_Cyber_Threat_Detection_NPTEL.zip"
PROJECT_ROOT = Path(__file__).resolve().parent

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv", "scratch", ".ipynb_checkpoints"}
EXCLUDED_EXTS = {".pyc", ".pyo", ".pyd", ".log"}


def should_include(rel_path: Path) -> bool:
    # Exclude system and temporary folders
    for part in rel_path.parts:
        if part in EXCLUDED_DIRS:
            return False

    # Exclude temporary extensions
    if rel_path.suffix in EXCLUDED_EXTS:
        return False

    # Exclude self archive
    if rel_path.name == ARCHIVE_NAME:
        return False

    # Exclude raw huge downloads from zip (keep sample fixture and gitkeeps)
    if "data" in rel_path.parts and "raw" in rel_path.parts:
        if rel_path.suffix.lower() == ".csv":
            return False

    return True


def package_project():
    zip_path = PROJECT_ROOT / ARCHIVE_NAME
    if zip_path.exists():
        zip_path.unlink()

    included_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded dirs in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(PROJECT_ROOT)

                if should_include(rel_path):
                    # Store with Cyber_Threat_Detection/ prefix for clean unzipping
                    archive_arcname = Path("Cyber_Threat_Detection") / rel_path
                    zf.write(full_path, arcname=str(archive_arcname))
                    included_count += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Archive successfully generated: {zip_path.name}")
    print(f"Total files packaged: {included_count}")
    print(f"Archive Size: {size_mb:.2f} MB")
    return zip_path


if __name__ == "__main__":
    package_project()

