"""Image utility functions for IrisAI.

Pure functions with no Chainlit dependency — fully testable.
Handles image path detection, saving to context-based directories,
and listing images in directories.
"""
import os
import re
import shutil
import datetime
from pathlib import Path
from typing import List, Optional


# Supported image extensions
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".bmp", ".webp", ".tiff", ".tif",
}

# Regex to find absolute file paths ending in image extensions (Unix-style)
# Matches paths like /data1/user/output/plot.png or /home/user/image.jpg
_IMAGE_PATH_PATTERN = re.compile(
    r'(?:^|\s|["\'(])'
    r'(/[\w./-]+\.(?:' + '|'.join(ext.lstrip('.') for ext in IMAGE_EXTENSIONS) + r'))'
    r'(?:\s|["\')],|$)',
    re.IGNORECASE | re.MULTILINE
)


def extract_image_paths(text: str) -> List[str]:
    """Extract image file paths from text content.

    Scans for absolute Unix file paths ending in image extensions.
    Only returns paths that actually exist on disk.

    Args:
        text: The text to scan for image paths

    Returns:
        List of existing image file paths (deduplicated, order preserved)
    """
    if not text:
        return []

    matches = _IMAGE_PATH_PATTERN.findall(text)

    # Deduplicate while preserving order
    seen = set()
    valid_paths = []
    for path in matches:
        if path not in seen and os.path.isfile(path):
            seen.add(path)
            valid_paths.append(path)

    return valid_paths


def get_context_image_dir(work_dir: str, context: str = "general") -> str:
    """Get or create a context-based image directory.

    Creates: work_dir/images/<context>/
    For example: work_dir/images/alphafold_BRCA2/
                 work_dir/images/plots_20260226/

    Args:
        work_dir: User's work directory
        context: Context name for the subdirectory (e.g. "alphafold_BRCA2", "analysis")

    Returns:
        Absolute path to the context image directory (created if needed)
    """
    # Sanitize context name — replace spaces/special chars with underscores
    safe_context = re.sub(r'[^\w\-.]', '_', context.strip())
    if not safe_context:
        safe_context = "general"

    image_dir = os.path.join(work_dir, "images", safe_context)
    os.makedirs(image_dir, exist_ok=True)
    return image_dir


def save_image_to_context_dir(
    source_path: str,
    work_dir: str,
    context: str = "general",
    new_name: Optional[str] = None,
) -> dict:
    """Save/copy an image file to a context-based directory.

    Copies the image from source_path to work_dir/images/<context>/<filename>.
    Optionally renames the file.

    Args:
        source_path: Path to the source image file
        work_dir: User's work directory
        context: Context name for organizing (e.g. "protein_analysis")
        new_name: Optional new filename (keeps original extension if not provided)

    Returns:
        Dict with success status, saved path, and metadata
    """
    try:
        source = Path(source_path)
        if not source.exists():
            return {"success": False, "error": f"Source file not found: {source_path}"}

        if not source.is_file():
            return {"success": False, "error": f"Not a file: {source_path}"}

        ext = source.suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            return {"success": False, "error": f"Not a supported image format: {ext}"}

        # Create context directory
        dest_dir = get_context_image_dir(work_dir, context)

        # Determine filename
        if new_name:
            # Ensure the new name has the right extension
            if not any(new_name.lower().endswith(e) for e in IMAGE_EXTENSIONS):
                new_name = new_name + ext
            dest_path = os.path.join(dest_dir, new_name)
        else:
            dest_path = os.path.join(dest_dir, source.name)

        # Avoid overwriting — add timestamp suffix if file exists
        if os.path.exists(dest_path):
            stem = Path(dest_path).stem
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            dest_path = os.path.join(dest_dir, f"{stem}_{timestamp}{ext}")

        shutil.copy2(source_path, dest_path)

        return {
            "success": True,
            "saved_path": dest_path,
            "context": context,
            "original_name": source.name,
            "size_bytes": os.path.getsize(dest_path),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def list_images_in_directory(directory: str, recursive: bool = False) -> dict:
    """List all image files in a directory.

    Args:
        directory: Path to search for images
        recursive: If True, search subdirectories too

    Returns:
        Dict with list of image files and metadata
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}

        if not dir_path.is_dir():
            return {"success": False, "error": f"Not a directory: {directory}"}

        images = []
        if recursive:
            for root, _dirs, files in os.walk(directory):
                for f in sorted(files):
                    if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                        full_path = os.path.join(root, f)
                        images.append({
                            "name": f,
                            "path": full_path,
                            "size_bytes": os.path.getsize(full_path),
                            "modified": datetime.datetime.fromtimestamp(
                                os.path.getmtime(full_path)
                            ).isoformat(),
                        })
        else:
            for f in sorted(os.listdir(directory)):
                full_path = os.path.join(directory, f)
                if os.path.isfile(full_path) and Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                    images.append({
                        "name": f,
                        "path": full_path,
                        "size_bytes": os.path.getsize(full_path),
                        "modified": datetime.datetime.fromtimestamp(
                            os.path.getmtime(full_path)
                        ).isoformat(),
                    })

        return {
            "success": True,
            "directory": directory,
            "image_count": len(images),
            "images": images,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def list_image_contexts(work_dir: str) -> dict:
    """List all image context directories under work_dir/images/.

    Returns:
        Dict with list of context directories and image counts
    """
    try:
        images_root = os.path.join(work_dir, "images")
        if not os.path.isdir(images_root):
            return {
                "success": True,
                "images_root": images_root,
                "contexts": [],
                "message": "No images directory yet. Images will be saved here when generated.",
            }

        contexts = []
        for entry in sorted(os.listdir(images_root)):
            ctx_path = os.path.join(images_root, entry)
            if os.path.isdir(ctx_path):
                # Count images in this context
                count = sum(
                    1 for f in os.listdir(ctx_path)
                    if os.path.isfile(os.path.join(ctx_path, f))
                    and Path(f).suffix.lower() in IMAGE_EXTENSIONS
                )
                contexts.append({
                    "context": entry,
                    "path": ctx_path,
                    "image_count": count,
                })

        return {
            "success": True,
            "images_root": images_root,
            "total_contexts": len(contexts),
            "contexts": contexts,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
