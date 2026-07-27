---
name: file-operations
description: Find, read, write, create, explore files and directories — upload handling,
  image display, file metadata, directory creation, text search across files
allowed_tools:
  - find_files
  - list_directory
  - read_text_file
  - write_text_file
  - edit_file
  - grep_file
  - get_file_info
  - make_directory
  - render_image_inline
  - save_image
  - analyze_files
  - review_codebase_section
  - summarize_command_output
  - batch
  - batch_readonly
  - upload_file
  - list_recent_uploads
  - read_file_head
  - read_file_tail
  - read_file_lines
  - get_file_overview
  - remove_file
  - list_saved_images
model: null
max_iterations: 30
guardrails:
- When user wants to upload a file, IMMEDIATELY call upload_file — NEVER tell user to click a button or describe an upload process
- For large files (>50KB), check size with get_file_info first, then use read_file_head/read_file_tail/read_file_lines/grep_file
- ALWAYS use find_files before claiming a file doesn't exist (check multiple patterns)
- NEVER read binary files (h5ad, BAM, .sif, .pkl) with read_text_file — use appropriate specialized tools
- When displaying images, ALWAYS use render_image_inline — user cannot see files on disk
- For file writes, confirm path with user if it would overwrite existing content
- For destructive actions (remove_file) ALWAYS ask user confirmation FIRST
---

# File Operations

Find, read, write, and manage files and directories. Handles file discovery,
content reading, text search, directory creation, image display, and file
metadata queries.

## When to Use This Skill

**Triggers:**
- "Find file X" / "Where is Y?" / "Search for files"
- "Read this file" / "Show me the contents"
- "Write to file" / "Save this as" / "Create file"
- "Show me this image" / "Display the figure"
- "What's in this directory?" / "List files"
- "Create a directory" / "Make folder"
- "How big is this file?" / "File info"
- "Search for 'pattern' in files"
- "What files were uploaded?"
- "Upload a file" / "I have a file to upload" / "I want to upload"

**NOT for (route elsewhere):**
- "Run this script" → code-execution
- "Analyze this h5ad" → bioinformatics-analysis
- "How much disk space?" → storage-management
- "Download from URL" → data-transfer

## Complete Workflow

### Step 1: Context

- work_dir from system context = user's primary workspace
- project_dir = active project directory
- Use these as starting points for file discovery

### Step 2: Execute Based on Request Type

**Finding files:**
```
find_files(pattern="*.csv", directory="{search_dir}")
```
Supports glob patterns: *.py, **/*.fastq.gz, data_*.txt

**Reading files:**
```
read_text_file(path="{file_path}")
```
For large files: specify start_line/end_line

**Writing files:**
```
write_text_file(path="{file_path}", content="{content}")
```

**Searching content:**
```
grep_file(pattern="{regex}", path="{file_or_dir}")
```

**Displaying images:**
```
render_image_inline(image_path="{path}")
```

**Directory operations:**
```
list_directory(path="{dir}")
make_directory(path="{new_dir}")
```

## Key Recipes

### Recipe: Upload File

1. Call `upload_file(work_dir=..., project_dir=...)` — this programmatically prompts the user
2. Tool returns file paths — use these directly in downstream tools
3. NEVER tell user to "click the upload button" or "use the attachment icon" — just call the tool

### Recipe: Find Files by Pattern

1. find_files(pattern, directory) → list matches
2. If no results: try broader pattern or parent directory
3. Report found files with paths

### Recipe: Read and Summarize Large File

1. get_file_info(path) → check size
2. If small (<50KB): read_text_file entirely
3. If large: read first/last N lines, or grep for specific content
4. Summarize key contents for user

### Recipe: Display Image/Figure

1. find_files(pattern="*.png", directory=figures_dir)
2. render_image_inline(image_path=path) — MANDATORY for user to see it
3. Describe what the image shows

### Recipe: Search Across Multiple Files

1. grep_file(pattern, directory) → find occurrences
2. read_text_file for context around matches
3. Report: file, line number, matching content

## Best Practices

- Check file exists before reading (find_files first if uncertain)
- For binary files: use specialized tools (extract_h5ad_summary, inspect_vcf_summary)
- Always render_image_inline for images — user cannot see disk files
- Use get_file_info for metadata (size, modification time) before reading large files
- Prefer grep_file over reading entire files when searching for patterns

## Tools

- `find_files` — Search for files by glob pattern
- `list_directory` — List directory contents
- `read_text_file` — Read file contents
- `write_text_file` — Write/create files
- `edit_file` — Modify existing files
- `grep_file` — Search file contents with regex
- `get_file_info` — File metadata (size, type, modification time)
- `make_directory` — Create directories
- `render_image_inline` — Display images in chat
- `save_image` — Save images to specific path
- `analyze_files` — AI-powered analysis of multiple files
- `review_codebase_section` — Understand code structure
- `summarize_command_output` — Summarize verbose content
- `upload_file` — Programmatically prompt user to upload files (returns saved paths)
- `list_recent_uploads` — List recently uploaded files
- `read_file_head` — Read first N lines of a file
- `read_file_tail` — Read last N lines of a file
- `read_file_lines` — Read specific line range from a file
- `get_file_overview` — Quick structural summary of a file
- `remove_file` — Delete a file (ALWAYS confirm with user first)
- `list_saved_images` — List previously saved images
