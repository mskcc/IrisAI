"""IrisAI Chainlit-dependent tools.

All tools that use Chainlit session context (cl.AskFileMessage, cl.Image,
cl.CustomElement, cl.Message, cl.step) are grouped here so the single agent
can register them all from one place.

Migrated from:
  - agents/search.py   → upload_file_tool, render_image_inline_tool,
                          extract_image_paths, IMAGE_EXTENSIONS
  - agents/alphafold.py → render_pdb_tool, render_cif_tool,
                           upload_weights_tool, _persist_weights_path_to_settings

These tools CANNOT be unit-tested without a running Chainlit session.
Tests verify file structure, exports, and non-Chainlit helpers only.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import chainlit as cl
from pathlib import Path
import shutil
import subprocess
import datetime
import logging
import json
import os
import pwd
import re
import stat
import time

logger = logging.getLogger("core.chainlit_tools")
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════════
# Image utilities (from agents/search.py)
# ═══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".bmp", ".webp", ".tiff", ".tif",
}

# Regex to find absolute file paths ending in image extensions
_IMAGE_PATH_PATTERN = re.compile(
    r'(?:^|\s|["\'\`]|[\(])'
    r'(/[\w./-]+\.(?:' + '|'.join(ext.lstrip('.') for ext in IMAGE_EXTENSIONS) + r'))'
    r'(?:\s|["\'\`]|[\)]|,|$)',
    re.IGNORECASE | re.MULTILINE
)


def extract_image_paths(text: str) -> list:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Render image inline tool (from agents/search.py)
# ═══════════════════════════════════════════════════════════════════════════════

@cl.step(name="Render Image Inline")
async def render_image_inline(image_paths: list[str], context_name: str = "") -> dict:
    """Render one or more images inline in the Chainlit chat interface.

    Takes a list of image file paths and displays each one inline in the chat.
    Also automatically saves images to work_dir/images/<context>/ for future reference.

    Args:
        image_paths: List of absolute paths to image files to render.
        context_name: Optional context name for organizing saved copies (e.g. 'protein_analysis').
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    rendered = []
    errors = []

    for img_path in image_paths:
        try:
            p = Path(img_path).resolve()
            if not p.exists() or not p.is_file():
                errors.append(f"File not found: {img_path}")
                continue

            ext = p.suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                errors.append(f"Not a recognized image: {img_path}")
                continue

            filename = p.name

            # Create Chainlit Image element and send inline
            image_element = cl.Image(
                path=str(p),
                name=filename,
                display="inline",
                size="large",
            )
            await cl.Message(
                content=f"🖼️ **{filename}**",
                elements=[image_element],
            ).send()

            rendered.append(str(p))
            logger.info(f"[IMAGE] Rendered inline: {p}")

        except Exception as e:
            errors.append(f"Failed to render {img_path}: {str(e)}")
            logger.error(f"[IMAGE_ERROR] {img_path}: {e}")

    result = {
        "success": len(rendered) > 0,
        "rendered": rendered,
        "rendered_count": len(rendered),
    }
    if errors:
        result["errors"] = errors
    if rendered:
        result["message"] = f"Rendered {len(rendered)} image(s) inline."
    else:
        result["message"] = "No images were rendered."

    return result


class RenderImageInlineInput(BaseModel):
    image_paths: list[str] = Field(
        ..., description="List of absolute paths to image files to render inline in chat"
    )
    context_name: str = Field(
        "", description="Optional context name for organizing saved copies (e.g. 'protein_analysis')"
    )


render_image_inline_tool = StructuredTool.from_function(
    func=render_image_inline,
    name="render_image_inline",
    description=(
        "Render image files inline in the chat interface. "
        "Takes a list of image file paths (.png, .jpg, .gif, .svg, etc.) "
        "and displays them directly in the conversation. "
        "Use this when the user asks to see/show/display/render an image, "
        "or when you have generated a plot/chart/visualization "
        "(NOT for protein structures — use render_pdb_from_paths or "
        "render_cif_from_paths for .pdb/.cif files). "
        "Also use this after save_image or list_saved_images to show the user their images."
    ),
    args_schema=RenderImageInlineInput,
    coroutine=render_image_inline,
    return_direct=False,
)


# ═══════════════════════════════════════════════════════════════════════════════
# General file upload tool (from agents/search.py)
# ═══════════════════════════════════════════════════════════════════════════════

@cl.step(name="Upload File")
async def upload_file(work_dir: str, project_dir: str = "") -> dict:
    """
    Upload one or more files to the project or work directory uploads folder.
    When project_dir is set, uploads go to project_dir/uploads/.
    Otherwise falls back to work_dir/uploads/.
    Accepts any file type up to 50MB each, max 10 files.
    Returns the saved file paths so any agent can reference them.

    IMPORTANT: The return dict includes 'file_paths' — a list of full absolute
    paths to every saved file. Always use these paths directly; never re-read
    file content to pass to downstream tools.
    """
    debug_log = []

    try:
        if not work_dir:
            return {
                "success": False,
                "message": "No work directory set. Please set one first with set_user_work_directory.",
                "debug_log": debug_log,
            }

        if project_dir:
            uploads_dir = Path(project_dir) / "uploads"
        else:
            uploads_dir = Path(work_dir) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        debug_log.append(f"Uploads directory: {uploads_dir}")

        # Prompt user to upload file(s)
        uploaded_files = await cl.AskFileMessage(
            content=(
                f"📂 **Upload location:** `{uploads_dir}`\n\n"
                "Please upload your file(s). Any file type is accepted (max 50MB each).\n"
                "You can upload multiple files at once."
            ),
            accept=["*"],
            max_files=10,
            max_size_mb=50,
            timeout=600,
        ).send()

        if not uploaded_files:
            debug_log.append("Upload cancelled or timed out")
            return {
                "success": False,
                "message": "File upload cancelled or timed out.",
                "debug_log": debug_log,
            }

        saved_files = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        for file in uploaded_files:
            await cl.Message(content=f"📥 `{file.name}` received!").send()
            debug_log.append(
                f"File received: name={file.name}, path={file.path}, size={file.size}"
            )

            source_path = Path(file.path)
            if not source_path.exists():
                debug_log.append(f"Source temp path does not exist: {source_path}")
                continue

            # Security: reject symlinks to prevent symlink attacks
            if source_path.is_symlink():
                debug_log.append(f"Rejected symlink source: {source_path}")
                continue

            # Save with timestamp prefix for uniqueness and easy sorting
            save_name = f"{timestamp}_{file.name}"
            save_path = uploads_dir / save_name
            try:
                shutil.copy(source_path, save_path)
                debug_log.append(f"Saved to: {save_path}")

                file_size = file.size if file.size else save_path.stat().st_size

                saved_files.append({
                    "name": file.name,
                    "path": str(save_path),
                    "size_bytes": file_size,
                })

                # ── CRITICAL: announce the FULL path explicitly so it is visible
                # in chat history for any downstream agent (e.g. AlphaFold agent).
                # The agent prompt reinforces this, but the message here is the
                # ground-truth record that survives cross-agent context.
                await cl.Message(
                    content=(
                        f"✅ `{file.name}` saved successfully.\n"
                        f"📁 **Full path:** `{save_path}`"
                    )
                ).send()

            except Exception as e:
                debug_log.append(f"Failed to save {file.name}: {str(e)}")
                await cl.Message(
                    content=f"❌ Failed to save `{file.name}`: {str(e)}"
                ).send()
                continue

        if not saved_files:
            return {
                "success": False,
                "message": "No files were saved successfully.",
                "debug_log": debug_log,
            }

        # Build a flat list of paths for easy consumption by other agents
        file_paths = [f["path"] for f in saved_files]

        return {
            "success": True,
            "saved_files": saved_files,
            # ── Convenience field: flat list of absolute paths ──────────────
            # Any agent receiving this result can use file_paths[0] directly
            # as fasta_path without reading saved_files[0]["path"].
            "file_paths": file_paths,
            "uploads_dir": str(uploads_dir),
            "debug_log": debug_log,
            "message": (
                f"Saved {len(saved_files)} file(s) to {uploads_dir}. "
                f"File path(s): {file_paths}"
            ),
        }

    except Exception as e:
        debug_log.append(f"Error: {str(e)}")
        return {"success": False, "error": str(e), "debug_log": debug_log}


class UploadFileInput(BaseModel):
    work_dir: str = Field(
        ..., description="The user's current work directory (absolute path)"
    )
    project_dir: str = Field(
        "", description="Active project directory (if set, uploads go here instead of work_dir/uploads/)"
    )


upload_file_tool = StructuredTool.from_function(
    func=upload_file,
    name="upload_file",
    description=(
        "Upload file(s) from the user's computer. "
        "When project_dir is provided, saves to project_dir/uploads/<timestamp>_<filename>. "
        "Otherwise saves to work_dir/uploads/<timestamp>_<filename>. "
        "Accepts any file type up to 50MB. "
        "Returns 'file_paths' (list of full absolute paths) and 'saved_files' (list of dicts with name/path/size). "
        "IMMEDIATELY call this tool when the user wants to upload a file — "
        "do NOT describe the upload process. "
        "Pass project_dir from the 'Project directory' in your USER ENVIRONMENT context."
    ),
    args_schema=UploadFileInput,
    coroutine=upload_file,
    return_direct=False,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Settings persistence helper (from agents/alphafold.py)
# ═══════════════════════════════════════════════════════════════════════════════

IRISAI_APP_NAME = os.environ.get("IRISAI_APP_NAME", "IrisAIdev")


def _persist_weights_path_to_settings(weights_path: str, debug_log: list) -> None:
    """Persist weights_path to the user's usersettings.json.

    This ensures the weights path survives across sessions and the LLM
    does not need to recompute it from work_dir every time.
    Uses the same file/directory layout as mcp_servers/file_ops.py.
    """
    try:
        uid = os.getuid()
        username = pwd.getpwuid(uid).pw_name
        base_dir = Path(f"/home/{username}/{IRISAI_APP_NAME}")
        base_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(base_dir, stat.S_IRWXU)  # 700: user only
        settings_path = base_dir / "usersettings.json"

        settings = {}
        if settings_path.exists():
            try:
                with open(settings_path, "r") as f:
                    settings = json.load(f)
            except Exception:
                pass  # Start fresh if corrupt

        settings["weights_path"] = str(weights_path)
        settings["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        os.chmod(settings_path, 0o600)

        debug_log.append(f"Persisted weights_path to user settings: {weights_path}")
    except Exception as e:
        debug_log.append(f"Warning: Could not persist weights_path to settings: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PDB rendering tool (from agents/alphafold.py)
# ═══════════════════════════════════════════════════════════════════════════════

@cl.step(name="Render PDB Structures")
async def render_pdb_from_paths(paths: list[str], job_id: str = ""):
    """
    Render interactive 3D viewers for PDB files at the given paths.
    Paths can be a single string or list of strings.
    """
    if isinstance(paths, str):
        paths = [paths]

    pdb_files = []
    for path_str in paths:
        path = Path(path_str).resolve()
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            pdb_files.append({
                "content": content,
                "name": path.name
            })
        except Exception as e:
            logger.error(f"Failed to read PDB {path}: {e}")

    if not pdb_files:
        await cl.Message(content="No valid PDB files found at the provided paths.").send()
        return "No PDBs rendered"

    await cl.Message(content=f"Rendering {len(pdb_files)} PDB structure(s) for job **{job_id}**:").send()

    for pdb_info in pdb_files:
        viewer = cl.CustomElement(
            name="ThreeDmolViewer",
            props={
                "pdbContent": pdb_info["content"],
                "width": "800px",
                "height": "600px",
                "backgroundColor": "white"
            }
        )
        await cl.Message(
            content=f"**{pdb_info['name']}** (interactive 3D viewer below)",
            elements=[viewer]
        ).send()

    return f"Rendered {len(pdb_files)} PDB viewers successfully"


# ═══════════════════════════════════════════════════════════════════════════════
# CIF rendering tool (from agents/alphafold.py)
# ═══════════════════════════════════════════════════════════════════════════════

@cl.step(name="Render CIF Structures")
async def render_cif_from_paths(paths: list[str], job_id: str = ""):
    """
    Render interactive 3D viewers for CIF (mmCIF/PDBx) files at the given paths.
    Paths can be a single string or list of strings. Uses 3Dmol for CIF support.
    """
    if isinstance(paths, str):
        paths = [paths]

    cif_files = []
    for path_str in paths:
        path = Path(path_str).resolve()
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            cif_files.append({
                "content": content,
                "name": path.name
            })
        except Exception as e:
            logger.error(f"Failed to read CIF {path}: {e}")

    if not cif_files:
        await cl.Message(content="No valid CIF files found at the provided paths.").send()
        return "No CIFs rendered"

    await cl.Message(content=f"Rendering {len(cif_files)} CIF structure(s) for job **{job_id}**:").send()

    for cif_info in cif_files:
        viewer = cl.CustomElement(
            name="ThreeDmolViewer",
            props={
                "cifContent": cif_info["content"],  # 3Dmol prop for CIF/mmCIF
                "width": "800px",
                "height": "600px",
                "backgroundColor": "white"
            }
        )
        await cl.Message(
            content=f"**{cif_info['name']}** (interactive 3D viewer below)",
            elements=[viewer]
        ).send()

    return f"Rendered {len(cif_files)} CIF viewers successfully"


# ═══════════════════════════════════════════════════════════════════════════════
# AlphaFold3 weights upload tool (from agents/alphafold.py)
# ═══════════════════════════════════════════════════════════════════════════════

@cl.step(name="Upload alphafold weights")
async def upload_weights_to_fixed_location(work_dir: str) -> dict:
    """
    Check the fixed relative weights location: work_dir/alphafold3/weights/
    If missing or empty → prompt user to upload open weights directly there.
    Saves files, decompresses .zst if present, returns final path.

    IMPORTANT: The return dict ALWAYS contains 'final_weights_path' on success.
    The caller MUST read and use this path directly — do NOT re-check or recompute.
    Also persists weights_path to usersettings.json so it survives across sessions.
    """
    debug_log = []

    try:
        if not work_dir:
            return {
                "success": False,
                "weights_confirmed": False,
                "message": "No work directory provided. Cannot locate weights.",
                "debug_log": debug_log,
            }

        weights_relative_path = Path(work_dir) / "alphafold3" / "weights"
        debug_log.append(f"Relative weights path: {weights_relative_path}")

        base_dir = weights_relative_path.resolve()
        debug_log.append(f"Resolved weights path: {base_dir}")

        # Check if directory exists and has files
        weights_exist = base_dir.exists() and base_dir.is_dir() and any(base_dir.iterdir())
        debug_log.append(f"Weights directory exists and has files: {weights_exist}")

        if weights_exist:
            debug_log.append("Weights found → using existing directory")
            _persist_weights_path_to_settings(str(base_dir), debug_log)
            cl.user_session.set("weights_path", str(base_dir))
            return {
                "success": True,
                "weights_confirmed": True,
                "final_weights_path": str(base_dir),
                "debug_log": debug_log,
                "message": f"✅ Weights CONFIRMED at {base_dir}. Use this path as weights_path for submit_alphafold3_job. Do NOT re-check."
            }

        # Weights missing → ask user how to proceed
        debug_log.append("Weights missing → asking user for action")

        action_res = await cl.AskActionMessage(
            content=f"AlphaFold3 requires open weights. I couldn't find them at the standard location:\n`{base_dir}`\n\nHow would you like to provide them?",
            actions=[
                cl.Action(name="weights_action", payload={"choice": "upload"}, label="Upload weights now"),
                cl.Action(name="weights_action", payload={"choice": "path"}, label="Specify path on cluster"),
                cl.Action(name="weights_action", payload={"choice": "skip"}, label="Skip for now"),
            ],
            timeout=300,
        ).send()

        # Parse action choice
        _choice = "skip"
        if action_res is not None:
            if isinstance(action_res, dict):
                _payload = action_res.get("payload", action_res)
                _choice = _payload.get("choice", "skip") if isinstance(_payload, dict) else "skip"
            elif hasattr(action_res, "payload"):
                _choice = getattr(action_res, "payload", {}).get("choice", "skip")

        if _choice == "skip":
            debug_log.append("User chose to skip weights setup")
            return {
                "success": False,
                "weights_confirmed": False,
                "message": "Weights setup skipped. You can provide weights later by telling me where they are or uploading them.",
                "debug_log": debug_log,
            }

        if _choice == "path":
            debug_log.append("User chose to specify path on cluster")
            path_res = await cl.AskUserMessage(
                content="Please provide the absolute path to your AlphaFold3 weights directory on the cluster (e.g. `/scratch/shared/af3_weights/`):",
                timeout=120,
            ).send()
            if path_res:
                _raw_path = path_res.get("output", "") if isinstance(path_res, dict) else str(
                    path_res.content if hasattr(path_res, "content") else path_res
                )
                _raw_path = _raw_path.strip().strip("`'\"")
                _user_path = Path(_raw_path)
                if _user_path.exists() and _user_path.is_dir() and any(_user_path.iterdir()):
                    debug_log.append(f"User-specified path validated: {_user_path}")
                    _persist_weights_path_to_settings(str(_user_path), debug_log)
                    cl.user_session.set("weights_path", str(_user_path))
                    return {
                        "success": True,
                        "weights_confirmed": True,
                        "final_weights_path": str(_user_path),
                        "debug_log": debug_log,
                        "message": f"✅ Weights CONFIRMED at {_user_path}. Use this path as weights_path for submit_alphafold3_job. Do NOT re-check."
                    }
                else:
                    debug_log.append(f"User-specified path invalid or empty: {_raw_path}")
                    return {
                        "success": False,
                        "weights_confirmed": False,
                        "message": f"Path '{_raw_path}' does not exist, is not a directory, or is empty. Please check and try again.",
                        "debug_log": debug_log,
                    }
            else:
                debug_log.append("User did not provide a path (timed out)")
                return {
                    "success": False,
                    "weights_confirmed": False,
                    "message": "No path provided. You can try again later.",
                    "debug_log": debug_log,
                }

        # _choice == "upload" → show file upload dialog
        debug_log.append("User chose to upload weights")

        uploaded_files = await cl.AskFileMessage(
            content=f"Please upload your open weights file(s) or archive now (e.g. af3.bin.zst). They will be saved to:\n`{base_dir}`",
            accept=["*"],
            max_files=10,
            max_size_mb=2000,
            timeout=9000
        ).send()

        if not uploaded_files:
            debug_log.append("Upload cancelled or timed out")
            return {
                "success": False,
                "weights_confirmed": False,
                "message": "Weights upload cancelled or timed out. You can try again later.",
                "debug_log": debug_log,
            }

        # Create directory if missing
        base_dir.mkdir(parents=True, exist_ok=True)
        debug_log.append(f"Directory ready: {base_dir}")

        file_infos = []
        final_weights_path = base_dir

        for file in uploaded_files:
            # Let the user know that the system is ready
            await cl.Message(
                content=f"`{file.name}` uploaded!"
            ).send()

            debug_log.append(f"File uploaded with these attributes: name={file.name}, path={file.path}, size={file.size}")

            # Use file.path (temporary server location) as source
            source_path = Path(file.path)
            if not source_path.exists():
                debug_log.append(f"Source temp path does not exist: {source_path}")
                continue

            save_path = base_dir / file.name
            try:
                # Copy the file (preserves binary integrity)
                shutil.copy(source_path, save_path)
                debug_log.append(f"Copied to destination: {save_path}")

                file_infos.append({
                    "name": file.name,
                    "server_path": str(save_path),
                    "size_bytes": file.size if file.size else save_path.stat().st_size,
                })
            except Exception as e:
                debug_log.append(f"Failed to copy {file.name}: {str(e)}")
                continue

            # Decompress .zst with timeout to prevent server hangs
            if save_path.suffix == ".zst":
                debug_log.append(f"Decompressing .zst: {save_path}")
                uncompressed_path = save_path.with_suffix("")
                try:
                    subprocess.run(
                        ["/usr/bin/zstd", "-d", str(save_path), "-o", str(uncompressed_path)],
                        check=True,
                        timeout=300,
                    )
                    debug_log.append(f"Uncompressed to: {uncompressed_path}")
                    os.remove(str(save_path))
                    final_weights_path = uncompressed_path.parent
                except subprocess.TimeoutExpired:
                    debug_log.append(f"Decompress timed out after 300s: {save_path}")
                    logger.error(f"[WEIGHTS] zstd decompress timed out for {save_path}")
                except Exception as e:
                    debug_log.append(f"Decompress failed: {str(e)}")

        _persist_weights_path_to_settings(str(final_weights_path), debug_log)
        cl.user_session.set("weights_path", str(final_weights_path))

        return {
            "success": True,
            "weights_confirmed": True,
            "uploaded_files": file_infos,
            "final_weights_path": str(final_weights_path),
            "debug_log": debug_log,
            "message": f"✅ Weights CONFIRMED at {final_weights_path}. Use this path as weights_path for submit_alphafold3_job. Do NOT re-check."
        }

    except Exception as e:
        debug_log.append(f"Error: {str(e)}")
        return {
            "success": False,
            "weights_confirmed": False,
            "error": str(e),
            "debug_log": debug_log,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas for tool inputs
# ═══════════════════════════════════════════════════════════════════════════════

class RenderPDBInput(BaseModel):
    paths: list[str] = Field(..., description="List of PDB file paths")
    job_id: str = Field("", description="Optional job ID for display")


class RenderCIFInput(BaseModel):
    paths: list[str] = Field(..., description="List of CIF/mmCIF file paths")
    job_id: str = Field("", description="Optional job ID for display")


class UploadWeightsInput(BaseModel):
    work_dir: str = Field(
        ..., description="The user's current work directory (absolute path)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# StructuredTool wrappers
# ═══════════════════════════════════════════════════════════════════════════════

render_pdb_tool = StructuredTool.from_function(
    func=render_pdb_from_paths,
    name="render_pdb_from_paths",
    description="Render interactive 3D viewers for one or more PDB files",
    args_schema=RenderPDBInput,
    coroutine=render_pdb_from_paths,  # important for async
    return_direct=False
)

render_cif_tool = StructuredTool.from_function(
    func=render_cif_from_paths,
    name="render_cif_from_paths",
    description="Render interactive 3D viewers for one or more CIF/mmCIF files",
    args_schema=RenderCIFInput,
    coroutine=render_cif_from_paths,  # important for async
    return_direct=False
)

upload_weights_tool = StructuredTool.from_function(
    func=upload_weights_to_fixed_location,
    name="upload_weights_to_fixed_location",
    description=(
        "Check for AlphaFold3 open weights and prompt the user to upload them if missing or empty. "
        "Returns dict with 'weights_confirmed' (bool) and 'final_weights_path' (str). "
        "IMPORTANT: After this tool returns successfully, use 'final_weights_path' directly as "
        "weights_path for submit_alphafold3_job. Do NOT re-check or recompute the weights path. "
        "The tool also persists weights_path to usersettings.json so it survives across sessions."
    ),
    args_schema=UploadWeightsInput,
    coroutine=upload_weights_to_fixed_location,
    return_direct=False
)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience list for registration in the single agent
# ═══════════════════════════════════════════════════════════════════════════════

CHAINLIT_TOOLS = [
    upload_file_tool,
    render_image_inline_tool,
    render_pdb_tool,
    render_cif_tool,
    upload_weights_tool,
]
