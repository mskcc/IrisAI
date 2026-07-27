# Copyright 2026 Lohit Valleru and contributors at
# Memorial Sloan Kettering Cancer Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mcp_servers/bio_processing.py
from fastmcp import FastMCP
from shared_auth import StaticBearerProvider
import base64
import shlex
import tempfile
import os
from pathlib import Path
import json
import subprocess
import pwd
import re
import io
import time
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from collections import Counter
import datetime

mcp = FastMCP("Bioinformatics Processing Server", auth=StaticBearerProvider())

@mcp.tool
def mutate_fasta(fasta_path: str, mutation: str) -> dict:
    """Apply a point mutation to a single-sequence FASTA file and save the result. Call this when the user wants to introduce a specific amino acid substitution (e.g. R175H, p.V600E). Takes a FILE PATH — reads the file directly to prevent sequence corruption. DO NOT pass file content as a string. DO NOT read the FASTA file with read_text_file first — pass the path directly here. Returns both the mutated content and the path to the saved mutated file.

        Args:
            fasta_path: Absolute path to the FASTA file on disk. Must start with /. Must contain exactly one sequence.
            mutation: Mutation in format like R175H, p.V600E, M1A. The reference amino acid at the position must match.
        Returns dict with mutated_fasta content, mutated_fasta_path (saved to disk), and success status."""
    # Validate file exists
    if not os.path.isfile(fasta_path):
        return {"error": f"FASTA file not found: {fasta_path}. "
                "Use list_recent_uploads() to find uploaded files, or provide the correct path."}
    
    try:
        with open(fasta_path, "r") as f:
            fasta_content = f.read()
    except Exception as e:
        return {"error": f"Failed to read FASTA file {fasta_path}: {str(e)}"}
    
    if not fasta_content.strip():
        return {"error": f"FASTA file is empty: {fasta_path}"}
    
    records = list(SeqIO.parse(io.StringIO(fasta_content), "fasta"))
    if len(records) != 1:
        return {"error": "FASTA must contain exactly one sequence"}
    rec = records[0]
    seq = str(rec.seq)
    m = re.match(r"[A-Za-z]?[.:]?([A-Za-z])(\d+)([A-Za-z\*\?])", mutation.strip())
    if not m:
        return {"error": "Mutation format invalid. Use e.g. R175H"}
    old_aa, pos, new_aa = m.groups()
    pos = int(pos)
    if seq[pos-1].upper() != old_aa.upper():
        return {"error": f"Position {pos} is {seq[pos-1]}, expected {old_aa}"}
    new_seq = seq[:pos-1] + new_aa.upper() + seq[pos:]
    new_rec = SeqRecord(Seq(new_seq), id=f"{rec.id}_{mutation}", description=f"mutated {mutation}")
    output = io.StringIO()
    SeqIO.write(new_rec, output, "fasta")
    mutated_content = output.getvalue()
    
    # Save mutated FASTA to disk next to original
    mutated_path = fasta_path.rsplit(".", 1)[0] + f"_{mutation.replace('.', '')}.fasta"
    try:
        with open(mutated_path, "w") as f:
            f.write(mutated_content)
    except Exception as e:
        return {"error": f"Failed to save mutated FASTA: {str(e)}"}
    
    return {"success": True, "mutated_fasta": mutated_content, "mutated_fasta_path": mutated_path}

# NOTE: generate_af3_input_json was removed — it was a duplicate of
# prepare_af3_json_from_sequences.  Having two identical tools confused the LLM
# (it sometimes called neither and wrote JSON manually).  All prompts now point
# to prepare_af3_json_from_sequences as the single canonical tool.

@mcp.tool
def submit_alphafold3_job(
    work_dir: str,
    weights_path: str,
    input_json_path: str,
    job_name: str = "af3_job",
    output_subdir: str = "output",
    run_data_pipeline: bool = True,
    run_inference: bool = True,
    project_dir: str = None,
    partition: str = "gpu",
) -> dict:
    """Submit an AlphaFold3 structure prediction job to the Slurm cluster.

    Call this ONCE after prepare_af3_json_from_sequences has created the input
    JSON and upload_weights_to_fixed_location has confirmed weights exist.

    REQUIRED parameters (all three must be provided):
      - work_dir: the user's work directory (e.g. /data1/.../irisai_workdir)
      - weights_path: path to AlphaFold3 weights (e.g. .../alphafold3/weights)
      - input_json_path: path to fold_input.json from prepare_af3_json_from_sequences

    DO NOT call if a job is already running for the same input.

        Args:
            work_dir: REQUIRED. Absolute path to user's work directory. Must exist.
            weights_path: REQUIRED. Absolute path to AlphaFold3 weights directory containing .bin files.
            input_json_path: REQUIRED. Absolute path to the fold_input.json file created by prepare_af3_json_from_sequences. This is the ONLY parameter for the input file — do not use any other name.
            job_name: Short alphanumeric name for Slurm (default: af3_job). No spaces.
            output_subdir: Output subdirectory name (default: output).
            run_data_pipeline: Run MSA/template pipeline step (default: true).
            run_inference: Run structure inference step (default: true).
            project_dir: If provided, organizes job under project_dir/jobs/<job_name>_<date>/.
            partition: Slurm partition to submit to (default: gpu). Use 'gpu' for general access or 'gpu_project' if user has access to idle H200 nodes.
        Returns dict with job_id, job_dir, output_dir on success."""
    resolved_input_json = input_json_path
    if not resolved_input_json:
        return {"error": "Missing required parameter: input_json_path. "
                "This must be the absolute path to the fold_input.json file "
                "created by prepare_af3_json_from_sequences."}

    # ── Validate work_dir exists on disk ───────────────────────────────────
    if not os.path.isdir(work_dir):
        return {"error": f"work_dir does not exist on disk: {work_dir}. "
                "Please call get_user_settings() to get the correct work directory, "
                "or call set_user_work_directory() to set one."}

    # ── Validate weights_path exists AND contains weight files ─────────────
    if not os.path.isdir(weights_path):
        return {"error": f"Weights directory does not exist: {weights_path}. "
                "AlphaFold3 requires model weights (af3.bin). "
                "Please call upload_weights_to_fixed_location(work_dir=...) to upload "
                "the weights file (af3.bin.zst) you received from Google DeepMind."}

    # Check for actual weight files (.bin)
    weight_files = [f for f in os.listdir(weights_path)
                    if f.endswith('.bin') or f.endswith('.npz') or f.endswith('.pkl')]
    if not weight_files:
        return {"error": f"Weights directory is EMPTY — no weight files found in: {weights_path}. "
                "Expected af3.bin or similar weight files. "
                "Please call upload_weights_to_fixed_location(work_dir=...) to upload "
                "the weights file (af3.bin.zst) you received from Google DeepMind."}

    # ── Validate input_json file exists on disk ────────────────────────────
    if not os.path.isfile(resolved_input_json):
        return {"error": f"Input JSON file does not exist: {resolved_input_json}. "
                "Please use prepare_af3_json_from_sequences() to create the input JSON first, "
                "or provide the correct path to an existing fold_input.json file."}

    input_json_resolved = resolved_input_json

    # ── Validate project_dir if provided ──────────────────────────────────
    _missing_project_dir_warning = None
    if project_dir:
        if not os.path.isabs(project_dir):
            return {"error": f"project_dir must be an absolute path, got: {project_dir}"}
        Path(project_dir).mkdir(parents=True, exist_ok=True)
    else:
        _projects_path = Path(work_dir) / "projects"
        if _projects_path.exists() and any(_projects_path.iterdir()):
            _missing_project_dir_warning = (
                "WARNING: project_dir not provided — job placed in work_dir root. "
                "Pass project_dir from 'Project directory' in your context for proper organization."
            )

    try:
        if project_dir:
            # Organized project directory structure with timestamp-based uniqueness
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_job_name = re.sub(r'[^\w\-]', '_', job_name)
            job_dir_name = f"{safe_job_name}_{ts_str}"
            jobs_base = Path(project_dir) / "jobs"
            jobs_base.mkdir(parents=True, exist_ok=True)
            job_dir = jobs_base / job_dir_name
            if job_dir.exists():
                counter = 1
                while (jobs_base / f"{job_dir_name}_{counter}").exists():
                    counter += 1
                job_dir = jobs_base / f"{job_dir_name}_{counter}"
            job_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Legacy behavior: flat job directories in work_dir
            base_work_dir = Path(work_dir)
            job_dir = base_work_dir / f"job_{int(time.time())}_{os.getpid()}"
            job_dir.mkdir(parents=True, exist_ok=True)
        
        final_output = job_dir / output_subdir
        final_output.mkdir(exist_ok=True)
        
        username = pwd.getpwuid(os.getuid()).pw_name
        user_email = f"{username}@{os.environ.get(chr(39)+chr(69)+chr(77)+chr(65)+chr(73)+chr(76)+chr(95)+chr(68)+chr(79)+chr(77)+chr(65)+chr(73)+chr(78)+chr(39), chr(101)+chr(120)+chr(97)+chr(109)+chr(112)+chr(108)+chr(101)+chr(46)+chr(111)+chr(114)+chr(103))}"
        
        # Build the Slurm script content.
        # IMPORTANT: We avoid f-string backslash escaping issues by building
        # the singularity command separately and joining with real backslash
        # line continuations.
        bslash = '\\'
        newline = '\n'
        
        singularity_cmd_parts = [
            f"{os.environ.get('SINGULARITY_BIN', 'singularity')} exec --nv",
            f'    -B /data1:/data1',
            f'    -B /admin:/admin',
            f'    -B "${{ALPHAFOLD_DB}}":/data',
            f'    -B "${{WORK_DIR}}":/output',
            f'    -B "${{MODEL_DIR}}":/root/models',
            f'    "${{SIF}}"',
            f'    /opt/alphafold3_venv/bin/python /app/alphafold/run_alphafold.py',
            f'    --json_path="${{INPUT_JSON}}"',
            f'    --model_dir=/root/models',
            f'    --db_dir=/data',
            f'    --output_dir=/output/{output_subdir}',
            f"    --run_data_pipeline={'true' if run_data_pipeline else 'false'}",
            f"    --run_inference={'true' if run_inference else 'false'}",
        ]
        # Join parts with backslash-newline for bash line continuation
        singularity_cmd = (f' {bslash}{newline}').join(singularity_cmd_parts)
        
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name={shlex.quote(job_name)}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=ALL

set -e
trap 'echo "Error at line $LINENO. Exit $?" >&2' ERR

module purge
module load cuda

export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:/usr/local/cuda-12.0/lib:/usr/local/cuda-12.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export APPTAINERENV_LD_LIBRARY_PATH="/usr/local/cuda:$LD_LIBRARY_PATH"

# Clear inherited bind mounts from MCP container (prevents /external mount failure)
export SINGULARITY_BIND=''
export APPTAINER_BIND=''

WORK_DIR="{job_dir}"
INPUT_JSON="{input_json_resolved}"
OUTPUT_DIR="{final_output}"
ALPHAFOLD_DB="/admin/shared_resources/alphafold3-db"
SIF="${ALPHAFOLD3_SIF:-/path/to/containers/alphafold3_v1.sif}"
MODEL_DIR="{weights_path}"

cd "${{WORK_DIR}}"

{singularity_cmd}

if [ $? -eq 0 ]; then
    PDB_FILES=$(find "${{OUTPUT_DIR}}" -name "*.pdb" | tr '\\n' ' ')
    echo '{{ "success": true, "pdb_files": "'"$PDB_FILES"'", "output_dir": "'"$OUTPUT_DIR"'" }}' > "${{WORK_DIR}}/results.json"
else
    echo '{{ "success": false }}' > "${{WORK_DIR}}/results.json"
fi
"""
        script_path = job_dir / "run_af3.slurm"
        script_path.write_text(slurm_content)
        os.chmod(script_path, 0o755)
        
        res = subprocess.run([
            "sbatch", 
            "--output", str(job_dir / "slurm-%j.out"),
            "--error", str(job_dir / "slurm-%j.err"), 
            str(script_path)
            ], capture_output=True, text=True, check=True)
        job_id = res.stdout.strip().split()[-1]

        result = {
            "success": True,
            "job_id": job_id,
            "job_dir": str(job_dir),
            "output_dir": str(final_output),
        }
        if _missing_project_dir_warning:
            result["warning"] = _missing_project_dir_warning
        return result
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
def extract_h5ad_summary(h5ad_path: str, max_cells: int = 1000, max_genes: int = 100) -> dict:
    """Get a basic summary of a single-cell .h5ad file — cell count, gene count, and available metadata columns. Call this as the first step when exploring a new .h5ad dataset. DO NOT call this on non-.h5ad files — it will fail. For detailed column analysis, follow up with list_obs_columns or get_unique_values.

        Args:
            h5ad_path: Absolute path to the .h5ad file. Must start with /.
            max_cells: Maximum cells to report (default: 1000). Does not limit reading.
            max_genes: Maximum genes to report (default: 100). Does not limit reading.
        Returns dict with num_cells, num_genes, obs_keys (metadata columns), and var_keys (gene annotations)."""
    try:
        with h5py.File(h5ad_path, 'r') as f:
            shape = (0, 0)
            if 'X' in f:
                if isinstance(f['X'], h5py.Dataset):
                    shape = f['X'].shape
                elif isinstance(f['X'], h5py.Group):
                    shape = (f['obs'].shape[0] if 'obs' in f else 0,
                             f['var'].shape[0] if 'var' in f else 0)
            
            obs_keys = list(f['obs'].keys()) if 'obs' in f else []
            var_keys = list(f['var'].keys()) if 'var' in f else []
            
        return {
            "success": True,
            "num_cells": shape[0],
            "num_genes": shape[1],
            "obs_keys": obs_keys[:20],
            "var_keys": var_keys[:20]
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
def inspect_vcf_summary(vcf_path: str, min_qual: float = 20.0) -> dict:
    """Get a quick overview of a VCF file — variant count, affected genes, and variant types. Call this as the first step when the user provides a VCF file. DO NOT call this on non-VCF files. Follow up with extract_coding_variants for detailed variant analysis.

        Args:
            vcf_path: Absolute path to the VCF file. Must start with /.
            min_qual: Minimum quality score to include a variant (default: 20.0).
        Returns dict with summary text, list of unique genes, and a suggestion for next steps."""
    try:
        from collections import Counter
        import re
        
        genes = set()
        variant_types = Counter()
        total_variants = 0
        coding_variants = 0
        
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 8:
                    continue
                chrom, pos, _, ref, alt, qual, filt, info = fields[:8]
                
                if float(qual or 0) < min_qual:
                    continue
                
                total_variants += 1
                
                # Very basic gene extraction from INFO (real VCFs often have GENE= or ANN=)
                gene_match = re.search(r'(?:GENE|Gene|gene)=([^;]+)', info, re.IGNORECASE)
                gene = gene_match.group(1).strip() if gene_match else "UNKNOWN"
                
                # Rough variant type (missense if SNV, etc.)
                if len(ref) == 1 and len(alt) == 1:
                    var_type = "SNV (potential missense)"
                elif len(ref) > 1 or len(alt) > 1:
                    var_type = "Indel"
                else:
                    var_type = "Other"
                
                variant_types[var_type] += 1
                
                # Count as coding if gene present (very approximate)
                if gene != "UNKNOWN":
                    genes.add(gene)
                    coding_variants += 1
        
        summary = f"""
        VCF Summary:
        - Total passing variants: {total_variants}
        - Variants with gene annotation: {coding_variants}
        - Unique genes affected: {len(genes)}
        - Variant types: {dict(variant_types)}
        """
        
        gene_list = sorted(genes)[:20]  # top 20 to avoid huge lists
        
        return {
            "success": True,
            "summary_text": summary.strip(),
            "unique_genes": gene_list,
            "total_genes_found": len(genes),
            "question_suggestion": "Which gene(s) would you like to focus on? (e.g. 'BRAF' or 'TP53, KRAS')"
        }
    
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
def extract_coding_variants(
    vcf_path: str,
    gene_filter: str = None,
    variant_types: list = ["missense", "nonsense"],
    max_return: int = 10
) -> dict:
    """Extract coding variants (missense, nonsense) from a VCF file with optional gene filtering. Call this AFTER inspect_vcf_summary — do NOT call this as the first step on a VCF file. Returns HGVS-like protein change annotations.

        Args:
            vcf_path: Absolute path to the VCF file. Must start with /.
            gene_filter: Optional gene symbol to filter by (e.g. 'BRAF'). If not provided, returns all genes.
            variant_types: List of variant types to include (default: ['missense', 'nonsense']).
            max_return: Maximum variants to return (default: 10).
        Returns dict with variants table, HGVS protein changes, and a suggestion for which to model."""
    try:
        import re
        from collections import defaultdict
        
        candidates = []
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 8:
                    continue
                chrom, pos, _, ref, alt, qual, filt, info = fields[:8]
                
                # Gene filter
                gene_match = re.search(r'(?:GENE|Gene|gene)=([^;]+)', info, re.IGNORECASE)
                gene = gene_match.group(1).strip() if gene_match else None
                if gene_filter and gene != gene_filter:
                    continue
                
                # Very basic type detection (improve with real ANN/CSQ parsing later)
                if len(ref) == 1 and len(alt) == 1:
                    var_type = "missense"  # assume SNV = missense for demo
                else:
                    continue
                
                if var_type not in variant_types:
                    continue
                
                # Rough HGVS.p approximation (real would use VEP/hgvs lib)
                aa_pos = int(pos) // 3 + 1  # naive
                hgvs_p_approx = f"p.X{aa_pos}{alt}"
                
                candidates.append({
                    "chrom_pos": f"{chrom}:{pos}",
                    "ref_alt": f"{ref}>{alt}",
                    "gene": gene or "UNKNOWN",
                    "type": var_type,
                    "approx_hgvs_p": hgvs_p_approx
                })
                
                if len(candidates) >= max_return:
                    break
        
        if not candidates:
            return {"error": "No coding variants found matching criteria"}
        
        table_str = "Variant | Gene | Type | Approx HGVS.p\n---|---|---|---\n"
        for c in candidates:
            table_str += f"{c['chrom_pos']} | {c['gene']} | {c['type']} | {c['approx_hgvs_p']}\n"
        
        return {
            "success": True,
            "variants_table_md": table_str,
            "variants_list": candidates,
            "question_suggestion": "Which variant(s) do you want to model? (e.g. reply with the approx HGVS.p or position)"
        }
    
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
def get_wildtype_protein_sequence(
    gene_symbol: str,
    organism: str = "Homo sapiens",
    use_canonical: bool = True
) -> dict:
    """Fetch the wild-type protein sequence from UniProt by gene symbol and save it as a FASTA file. Call this when the user needs a reference sequence for mutation modeling. Returns the file PATH — pass this fasta_path directly to apply_protein_variants or prepare_af3_json_from_sequences. DO NOT read the returned FASTA file with read_text_file — always pass the path, never the content.

        Args:
            gene_symbol: Gene symbol like TP53, BRAF, EGFR. Case-insensitive but uppercase preferred.
            organism: Species name (default: 'Homo sapiens').
            use_canonical: Whether to prefer the canonical isoform (default: true).
        Returns dict with fasta_path (saved file), fasta_content, uniprot_id, gene, and sequence length."""
    try:
        import requests

        # Simple UniProt REST query for human gene
        query = f"gene_exact:{gene_symbol} AND organism:\"{organism}\" AND reviewed:true"
        if use_canonical:
            query += " AND keyword:canonical"

        url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&format=fasta&size=1"
        response = requests.get(url)
        response.raise_for_status()

        fasta = response.text.strip()
        if not fasta:
            return {"error": f"No reviewed canonical protein found for {gene_symbol}"}

        # Parse basic info
        header, sequence = fasta.split('\n', 1)
        uniprot_id = header.split('|')[1]
        description = header.split(' ', 1)[1] if ' ' in header else ""

        # Save FASTA to a temp file so the agent can pass the PATH (not content)
        # to downstream tools like prepare_af3_json_from_sequences
        fasta_dir = tempfile.mkdtemp(prefix=f"wildtype_{gene_symbol}_")
        fasta_path = os.path.join(fasta_dir, f"{gene_symbol}_{uniprot_id}.fasta")
        with open(fasta_path, "w") as fasta_file:
            fasta_file.write(fasta)
        
        return {
            "success": True,
            "fasta": fasta,
            "fasta_path": fasta_path,
            "uniprot_id": uniprot_id,
            "gene_symbol": gene_symbol,
            "sequence_length": len(sequence.replace('\n', '')),
            "description": description,
            "question_suggestion": f"Using canonical protein {uniprot_id} ({len(sequence)} aa) for {gene_symbol}. FASTA saved to {fasta_path}. Is this the correct version/transcript?"
        }

    except Exception as e:
        return {"error": f"Failed to fetch sequence: {str(e)}"}

@mcp.tool
def apply_protein_variants(
    wildtype_fasta_path: str,
    variants_hgvsp: list[str]  # e.g. ["p.Val600Glu", "p.Arg175His"]
) -> dict:
    """Apply one or more HGVS protein changes to a wild-type FASTA file and save mutated FASTAs. Call this after get_wildtype_protein_sequence to create mutant sequences for structure prediction. Takes a FILE PATH — reads the file directly to prevent sequence corruption. DO NOT pass file content. DO NOT read the FASTA file with read_text_file first — pass the path directly here.

        Args:
            wildtype_fasta_path: Absolute path to the wild-type FASTA file. Must start with /. Must contain exactly one sequence.
            variants_hgvsp: List of HGVS protein changes, e.g. ['p.V600E', 'p.R175H']. Only simple substitutions (p.X123Y) supported.
        Returns dict with applied_mutations (each with mutated_fasta_path), failed mutations, and success status."""
    try:
        from Bio import SeqIO
        from Bio.Seq import Seq
        from io import StringIO
        
        # Read FASTA from file path
        if not os.path.isfile(wildtype_fasta_path):
            return {"error": f"Wild-type FASTA file not found: {wildtype_fasta_path}. "
                    "Use list_recent_uploads() or get_wildtype_protein_sequence() to get the file."}
        
        with open(wildtype_fasta_path, "r") as f:
            wildtype_fasta = f.read()
        
        if not wildtype_fasta.strip():
            return {"error": f"Wild-type FASTA file is empty: {wildtype_fasta_path}"}

        records = list(SeqIO.parse(StringIO(wildtype_fasta), "fasta"))
        if len(records) != 1:
            return {"error": "Wild-type FASTA must have exactly one sequence"}
        wt_rec = records[0]
        wt_seq = str(wt_rec.seq)

        results = []
        for hgvs in variants_hgvsp:
            # Parse simple p.X123Y format (extend later with full hgvs lib)
            match = re.match(r"p\.([A-Z])(\d+)([A-Z*])", hgvs)
            if not match:
                results.append({"variant": hgvs, "error": "Unsupported HGVS format (only p.X123Y supported now)"})
                continue

            old_aa, pos_str, new_aa = match.groups()
            pos = int(pos_str)

            if pos < 1 or pos > len(wt_seq):
                results.append({"variant": hgvs, "error": f"Position {pos} out of range (seq len {len(wt_seq)})"})
                continue

            if wt_seq[pos-1] != old_aa:
                results.append({"variant": hgvs, "warning": f"Reference AA mismatch: expected {old_aa}, found {wt_seq[pos-1]}"})

            mutated_seq = wt_seq[:pos-1] + new_aa + wt_seq[pos:]
            new_id = f"{wt_rec.id}_{hgvs.replace('p.', '')}"
            new_rec = wt_rec.__class__(seq=Seq(mutated_seq), id=new_id, description=f"mutated {hgvs}")

            output_fasta = new_rec.format("fasta")
            results.append({
                "variant": hgvs,
                "mutated_fasta": output_fasta,
                "original_aa": wt_seq[pos-1],
                "new_aa": new_aa,
                "position": pos
            })

        # Save mutated FASTAs to files
        for r in results:
            if "error" not in r and "mutated_fasta" in r:
                mutated_path = wildtype_fasta_path.rsplit(".", 1)[0] + f"_{r['variant'].replace('p.', '').replace('.', '')}.fasta"
                try:
                    with open(mutated_path, "w") as mf:
                        mf.write(r["mutated_fasta"])
                    r["mutated_fasta_path"] = mutated_path
                except Exception:
                    pass  # non-fatal — content is still in response
        
        return {
            "success": len([r for r in results if "error" not in r]) > 0,
            "applied_mutations": [r for r in results if "error" not in r],
            "failed": [r for r in results if "error" in r],
            "note": "Only simple substitutions supported. Use full HGVS library for complex variants."
        }

    except Exception as e:
        return {"error": str(e)}

# ── Valid amino acid, DNA, RNA character sets ─────────────────────────────────
_VALID_PROTEIN_CHARS = set("ACDEFGHIKLMNPQRSTVWYX")
_VALID_DNA_CHARS = set("ATCGN")
_VALID_RNA_CHARS = set("AUCGN")
# Valid SMILES characters (broad set — allows most organic chemistry notation)
_VALID_SMILES_CHARS = set(r"ABCDEFGHIKLMNOPRSTUVWXYZabcdefghiklmnoprstuvwxyz0123456789@+-=#$/\()[]%.:")


def _parse_fasta_file(fasta_path: str) -> dict:
    """Parse a FASTA file and return sequences with auto-detected types.
    
    Returns dict with 'success', 'sequences' list, and 'warnings'.
    Each sequence has: id, sequence, type (protein/dna/rna), header.
    """
    if not os.path.isfile(fasta_path):
        return {"error": f"FASTA file not found: {fasta_path}. "
                "Use list_recent_uploads() to find uploaded files."}
    
    try:
        with open(fasta_path, "r") as f:
            content = f.read()
    except Exception as e:
        return {"error": f"Failed to read FASTA file: {str(e)}"}
    
    if not content.strip():
        return {"error": f"FASTA file is empty: {fasta_path}"}
    
    # Parse FASTA
    records = list(SeqIO.parse(io.StringIO(content), "fasta"))
    if not records:
        return {"error": f"No valid FASTA sequences found in: {fasta_path}"}
    
    chain_ids = [chr(ord("A") + i) for i in range(26)]  # A-Z
    sequences = []
    warnings = []
    
    for i, rec in enumerate(records):
        seq = str(rec.seq).upper().strip()
        chain_id = chain_ids[i] if i < 26 else f"chain_{i}"
        
        # Auto-detect sequence type
        seq_chars = set(seq)
        if seq_chars <= _VALID_DNA_CHARS and len(seq) > 0:
            seq_type = "dna"
        elif "U" in seq_chars and seq_chars <= _VALID_RNA_CHARS:
            seq_type = "rna"
        else:
            seq_type = "protein"
            # Validate protein characters
            invalid = seq_chars - _VALID_PROTEIN_CHARS
            if invalid:
                warnings.append(f"Chain {chain_id}: found unusual characters {invalid} in protein sequence")
        
        sequences.append({
            "id": chain_id,
            "sequence": seq,
            "type": seq_type,
            "header": rec.description,
        })
    
    return {"success": True, "sequences": sequences, "warnings": warnings}


def _validate_ligands(ligands_json: str) -> dict:
    """Validate and sanitize ligand entries from JSON string.
    
    Returns dict with 'success', 'ligands' list, and 'warnings'.
    """
    if not ligands_json or ligands_json.strip() == "[]":
        return {"success": True, "ligands": [], "warnings": []}
    
    try:
        ligands = json.loads(ligands_json) if isinstance(ligands_json, str) else ligands_json
    except json.JSONDecodeError as e:
        return {"error": f"Invalid ligands JSON: {str(e)}. "
                "Expected format: [{{\"id\": \"B\", \"ccdCodes\": [\"ATP\"]}}]"}
    
    if not isinstance(ligands, list):
        return {"error": "Ligands must be a JSON array/list"}
    
    validated = []
    warnings = []
    
    for i, lig in enumerate(ligands):
        if not isinstance(lig, dict):
            return {"error": f"Ligand entry {i} must be a dict, got {type(lig).__name__}"}
        
        entry = {"ligand": {}}
        
        # Validate CCD codes
        if "ccdCodes" in lig:
            codes = lig["ccdCodes"]
            if not isinstance(codes, list):
                return {"error": f"Ligand {i}: ccdCodes must be a list, got {type(codes).__name__}"}
            sanitized_codes = []
            for code in codes:
                code_str = str(code).strip().upper()
                if not code_str.isalnum() or len(code_str) < 1 or len(code_str) > 5:
                    return {"error": f"Ligand {i}: invalid CCD code '{code}'. "
                            "CCD codes must be 1-5 alphanumeric characters (e.g. ATP, ZN, MG)."}
                sanitized_codes.append(code_str)
            entry["ligand"]["ccdCodes"] = sanitized_codes
        
        # Validate SMILES
        if "smiles" in lig:
            smiles = str(lig["smiles"]).strip()
            if not smiles:
                return {"error": f"Ligand {i}: SMILES string is empty"}
            invalid_chars = set(smiles) - _VALID_SMILES_CHARS
            if invalid_chars:
                warnings.append(f"Ligand {i}: SMILES contains unusual characters: {invalid_chars}")
            entry["ligand"]["smiles"] = smiles
        
        # Must have either ccdCodes or smiles
        if "ccdCodes" not in entry["ligand"] and "smiles" not in entry["ligand"]:
            return {"error": f"Ligand {i}: must have either 'ccdCodes' or 'smiles'"}
        
        # ID
        if "id" in lig:
            entry["ligand"]["id"] = str(lig["id"])
        
        validated.append(entry)
    
    return {"success": True, "ligands": validated, "warnings": warnings}


@mcp.tool
def prepare_af3_json_from_sequences(
    fasta_path: str,
    project_dir: str,
    output_path: str = "",
    model_seeds: list[int] = None,
    name: str = "af3_prediction",
    ligands: str = "[]",
) -> dict:
    """Create the AlphaFold3 fold_input.json from a FASTA file.

    THIS IS THE ONLY TOOL FOR GENERATING ALPHAFOLD3 INPUT JSON.
    DO NOT use write_text_file to create fold_input.json manually.
    DO NOT read the FASTA file first — pass the path directly here.

    REQUIRED parameters:
      - fasta_path: path to the uploaded .fasta file
      - project_dir: the project directory (from 'Project directory' in USER ENVIRONMENT)

    OPTIONAL parameters:
      - output_path: where to save the JSON. If empty (default), auto-generated as
        project_dir/inputs/{name}_{timestamp}.json. If provided, MUST be under project_dir.

    Example call:
      prepare_af3_json_from_sequences(
        fasta_path="/data1/.../uploads/P01308.fasta",
        project_dir="/data1/.../projects/myproject",
        model_seeds=[1, 2, 3],
        name="P01308_monomer"
      )

        Args:
            fasta_path: REQUIRED. Absolute path to a FASTA file (not content). Must start with /. The tool reads it directly.
            project_dir: REQUIRED. Absolute path to the project directory (from 'Project directory' in your USER ENVIRONMENT context).
            output_path: Optional. If empty, auto-generated under project_dir/inputs/. If provided, must be under project_dir.
            model_seeds: List of integer random seeds for predictions (default: [1, 2, 3]). More seeds = more prediction models.
            name: Short alphanumeric name for the prediction (default: af3_prediction). Used in the JSON metadata.
            ligands: JSON string for ligands. Format: '[{{"id": "B", "ccdCodes": ["ATP"]}}]'. Default: '[]' (no ligands).
        Returns dict with json_path (where it was saved), json_preview, and any warnings."""
    # ── Validate project_dir ─────────────────────────────────────────────
    if project_dir:
        if not os.path.isabs(project_dir):
            return {"error": f"project_dir must be an absolute path, got: {project_dir}"}
        Path(project_dir).mkdir(parents=True, exist_ok=True)

    # ── Validate output_path consistency with project_dir ─────────────────
    if output_path and project_dir:
        try:
            _resolved_out = str(Path(output_path).resolve())
            _resolved_proj = str(Path(project_dir).resolve())
            if not _resolved_out.startswith(_resolved_proj):
                return {"error": f"output_path ({output_path}) is not under project_dir ({project_dir}). "
                        "Remove output_path to auto-generate it under the project directory."}
        except Exception:
            pass

    # ── Parse FASTA file directly (NO LLM middleman) ──────────────────────
    parsed = _parse_fasta_file(fasta_path)
    if "error" in parsed:
        return parsed

    if model_seeds is None:
        model_seeds = [1, 2, 3]

    data = {
        "name": name,
        "dialect": "alphafold3",
        "version": 1,
        "modelSeeds": model_seeds,
        "sequences": [],
    }

    # Track any sequence sanitization performed
    sanitization_warnings = list(parsed.get("warnings", []))

    for seq_entry in parsed["sequences"]:
        seq_type = seq_entry["type"]
        seq = seq_entry["sequence"]
        chain_id = seq_entry["id"]
        
        if seq_type == "protein":
            entry = {"protein": {"sequence": seq, "id": chain_id}}
            data["sequences"].append(entry)
        elif seq_type == "dna":
            entry = {"dna": {"sequence": seq, "id": chain_id}}
            data["sequences"].append(entry)
        elif seq_type == "rna":
            # ── AlphaFold3 RNA sanitization ────────────────────────────
            u_count = seq.count("U") + seq.count("u")
            sanitized_seq = seq.replace("U", "C").replace("u", "c")
            if u_count > 0:
                sanitization_warnings.append(
                    f"RNA chain {chain_id}: replaced {u_count} U→C "
                    f"(uracil→cytosine) for AlphaFold3 compatibility"
                )
            entry = {"rna": {"sequence": sanitized_seq, "id": chain_id}}
            data["sequences"].append(entry)
    
    # ── Validate and add ligands ──────────────────────────────────────────
    lig_result = _validate_ligands(ligands)
    if "error" in lig_result:
        return lig_result
    
    for lig_entry in lig_result.get("ligands", []):
        data["sequences"].append(lig_entry)
    
    sanitization_warnings.extend(lig_result.get("warnings", []))

    # Determine output path — use timestamp to avoid overwriting previous runs
    if not output_path and project_dir:
        inputs_dir = Path(project_dir) / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _safe_name = re.sub(r'[^\w\-]', '_', name) if name else "fold_input"
        output_path = str(inputs_dir / f"{_safe_name}_{_ts}.json")

    try:
        # Build the base result dict
        result = {"success": True}

        # Include sanitization warnings if any U→C replacements were made
        if sanitization_warnings:
            result["sanitization_warnings"] = sanitization_warnings

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            result["json_path"] = output_path
            result["json_preview"] = json.dumps(data, indent=2)[:500] + "..."
            result["question_suggestion"] = "Input JSON prepared. Do you want to add any ligands, DNA/RNA partners, or model as monomer/multimer?"
        else:
            result["json_content"] = json.dumps(data, indent=2)
            result["question_suggestion"] = "JSON ready. Shall I submit the AlphaFold3 job now?"

        return result

    except Exception as e:
        return {"error": str(e)}

# ── Single-cell / AnnData helpers ─────────────────────────────────────────────

import h5py
import numpy as np
import pandas as pd
from collections import Counter

@mcp.tool
def list_obs_columns(h5ad_path: str) -> dict:
    """List all metadata column names available in adata.obs of a .h5ad file. Call this after extract_h5ad_summary to discover what cell annotations are available before querying specific columns. DO NOT call this on non-.h5ad files.

        Args:
            h5ad_path: Absolute path to the .h5ad file. Must start with /.
        Returns dict with list of obs column names and count."""
    try:
        with h5py.File(h5ad_path, 'r') as f:
            if 'obs' not in f:
                return {"error": "No 'obs' group found in .h5ad file"}
            obs_keys = list(f['obs'].keys())
        return {
            "success": True,
            "obs_columns": obs_keys,
            "message": f"Found {len(obs_keys)} cell metadata columns"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
def get_unique_values(h5ad_path: str, column: str) -> dict:
    """Get all unique values and their counts from a specific column in adata.obs. Call this after list_obs_columns to verify the column name exists. DO NOT guess column names — check list_obs_columns first.

        Args:
            h5ad_path: Absolute path to the .h5ad file. Must start with /.
            column: Column name as a string (e.g. 'cell_type', 'leiden'). Must be an exact match to an obs column name.
        Returns dict with unique values, their counts, and total cells."""
    if not isinstance(column, str):
        return {"error": f"Column name must be a string, got {type(column).__name__} instead. Example: 'cell_type'"}

    column = column.strip()
    if not column:
        return {"error": "Column name cannot be empty"}

    try:
        with h5py.File(h5ad_path, 'r') as f:
            if 'obs' not in f:
                return {"error": "No 'obs' group found in .h5ad file"}
            if column not in f['obs']:
                obs_keys = list(f['obs'].keys())
                return {
                    "error": f"Column '{column}' not found in obs",
                    "available_columns": obs_keys[:20],  # show first 20 for help
                    "total_columns": len(obs_keys)
                }
            
            dataset = f['obs'][column]
            data = dataset[:]
            
            # Handle different dtypes safely
            if data.dtype.kind in ('S', 'O'):
                values = np.array([x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x) for x in data])
            else:
                values = data.astype(str)
            
            counts = Counter(values)
            unique_list = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            
        return {
            "success": True,
            "column": column,
            "unique_values": [v for v, c in unique_list],
            "counts": {v: int(c) for v, c in unique_list},
            "total_cells": len(values),
            "message": f"Found {len(unique_list)} unique values in '{column}'"
        }
    except Exception as e:
        return {"error": f"Failed to read column '{column}': {str(e)}"}

@mcp.tool
def get_top_n_categories(h5ad_path: str, column: str, n: int = 10) -> dict:
    """Get the top N most frequent categories in an obs column. Call this after list_obs_columns to verify the column name exists. Provides a quick summary of the most common cell types or clusters without seeing all values.

        Args:
            h5ad_path: Absolute path to the .h5ad file. Must start with /.
            column: Column name as a string (e.g. 'cell_type').
            n: Number of top categories to return (default: 10).
        Returns dict with top categories, their counts, total unique values, and total cells."""
    result = get_unique_values(h5ad_path, column)
    if not result.get("success"):
        return result

    top = sorted(result["counts"].items(), key=lambda x: x[1], reverse=True)[:n]
    return {
        "success": True,
        "column": column,
        "top_n": dict(top),
        "total_unique": len(result["unique_values"]),
        "total_cells": result["total_cells"]
    }

@mcp.tool
def summarize_cell_types(h5ad_path: str) -> dict:
    """Auto-detect and summarize cell-type-related columns in a .h5ad file. Call this after extract_h5ad_summary to confirm the file is valid. Provides a quick overview of cell annotations — looks for columns containing 'cell_type', 'cluster', 'leiden', 'annotation', etc. Limits output to top 3 matching columns.

        Args:
            h5ad_path: Absolute path to the .h5ad file. Must start with /.
        Returns dict with detected cell-type columns and their value summaries."""
    try:
        with h5py.File(h5ad_path, 'r') as f:
            obs_keys = list(f['obs'].keys()) if 'obs' in f else []

        likely_cell_type_cols = [
            k for k in obs_keys
            if any(term in k.lower() for term in ['cell_type', 'celltypist', 'cluster', 'leiden', 'annotation', 'label', 'major', 'minor'])
        ]

        if not likely_cell_type_cols:
            return {"message": "No obvious cell-type columns found", "obs_columns": obs_keys}

        summaries = {}
        for col in likely_cell_type_cols[:3]:  # limit to top 3 to avoid huge output
            res = get_unique_values(h5ad_path, col)
            if res.get("success"):
                summaries[col] = {
                    "unique_count": len(res["unique_values"]),
                    "top_5": dict(sorted(res["counts"].items(), key=lambda x: x[1], reverse=True)[:5]),
                    "total_cells": res["total_cells"]
                }

        return {
            "success": True,
            "likely_columns": likely_cell_type_cols,
            "summaries": summaries,
            "message": f"Found {len(likely_cell_type_cols)} likely cell-type/annotation columns"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("MCP_BIO_PROCESS_PORT", 8004))
    mcp.run(transport="streamable-http", host="127.0.0.1", port=port, path="/")
