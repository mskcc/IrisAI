"""Configuration loading — extracted from app.py.

Pure functions for loading YAML configs.
No Chainlit or LLM dependencies.

Phase 1 update: Removed load_agent_config() and get_agent_names_and_descriptions()
which loaded from agents.yaml. These are replaced by SkillLoader which reads
skills/*.md files directly. Only load_mcp_server_config() remains.
"""
import json
import logging
import os
import yaml
from typing import List, Dict, Optional

logger = logging.getLogger("core.config")


def load_mcp_server_config(
    config_path: str,
    env_overrides: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """Load MCP server configuration from YAML.
    
    Args:
        config_path: Path to mcp_servers.yaml
        env_overrides: Optional dict of env var overrides (for testing)
    
    Returns:
        List of server dicts with name, url, description
    """
    if not os.path.exists(config_path):
        return []
    
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    servers = data.get("servers", [])
    result = []
    
    for s in servers:
        port_var = s.get("env_port_var")
        if env_overrides and port_var in env_overrides:
            port = int(env_overrides[port_var])
        else:
            port = int(os.environ.get(port_var, s.get("default_port", 8000)))
        
        url = f"http://127.0.0.1:{port}/"
        result.append({
            "name": s["name"],
            "url": url,
            "description": s.get("description", s["name"]),
        })
    
    return result


def load_user_extension_configs(extensions_dir: str) -> List[Dict[str, str]]:
    """Load user extension MCP server configs from .active_ports.json + manifests.

    At session start, script.sh.erb dynamically allocates ports for user extensions
    and writes them to {extensions_dir}/.active_ports.json. This function reads that
    file and combines it with manifest.yaml metadata to produce server dicts.

    Args:
        extensions_dir: Path to user's extensions directory
            (e.g. /home/{user}/IrisAI/extensions/)

    Returns:
        List of server dicts with name, url, description — same format as
        load_mcp_server_config() so they can be merged directly.
    """
    if not os.path.isdir(extensions_dir):
        return []

    ports_file = os.path.join(extensions_dir, ".active_ports.json")
    if not os.path.isfile(ports_file):
        return []

    try:
        with open(ports_file, encoding="utf-8") as f:
            active_ports = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read active ports file {ports_file}: {e}")
        return []

    if not isinstance(active_ports, dict):
        return []

    result = []
    for ext_name, port in active_ports.items():
        manifest_path = os.path.join(extensions_dir, ext_name, "manifest.yaml")
        if not os.path.isfile(manifest_path):
            logger.warning(f"Extension '{ext_name}' has port but no manifest, skipping")
            continue

        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            logger.warning(f"Failed to read manifest for extension '{ext_name}': {e}")
            continue

        if not isinstance(manifest, dict):
            continue

        url = f"http://127.0.0.1:{int(port)}/"
        result.append({
            "name": f"user_{ext_name}",
            "url": url,
            "description": manifest.get("description", ext_name),
        })
        logger.info(f"Discovered user extension: {ext_name} on port {port}")

    return result
