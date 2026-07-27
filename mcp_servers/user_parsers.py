"""Structured output parsers for user and group data.

These are pure-logic functions that convert raw user/group data
into clean, structured JSON-friendly dicts.
They contain NO subprocess calls and NO I/O — making them fully testable.

Used by mcp_servers/file_ops.py to structure get_current_user_info,
get_user_groups, and list_group_accessible_dirs output at the source
before returning results to the supervisor.
"""


# ---------------------------------------------------------------------------
# User/group data structuring
# ---------------------------------------------------------------------------

def structure_user_info(username: str, uid: int, home_dir: str,
                        primary_group: str, all_groups: list) -> dict:
    """Structure user info into a clean, deduplicated JSON dict.

    Args:
        username: OS username.
        uid: User ID.
        home_dir: Home directory path.
        primary_group: Primary group name.
        all_groups: List of all group names (may contain duplicates).

    Returns:
        Clean dict with deduplicated groups.
    """
    # Deduplicate and sort groups
    unique_groups = sorted(set(all_groups))

    return {
        "success": True,
        "username": username,
        "uid": uid,
        "home_dir": home_dir,
        "primary_group": primary_group,
        "all_groups": unique_groups,
        "groups_count": len(unique_groups),
    }


def structure_group_info(username: str, primary_group: str,
                         primary_gid: int, groups: list) -> dict:
    """Structure group info into a clean JSON dict without raw member lists.

    Instead of dumping all members of every group (which can be huge and
    is a privacy concern), we return only the member_count per group.

    Args:
        username: OS username.
        primary_group: Primary group name.
        primary_gid: Primary group GID.
        groups: List of dicts with 'group_name', 'gid', 'members' keys.

    Returns:
        Clean dict with member_count instead of raw member lists.
    """
    clean_groups = []
    for g in groups:
        clean_groups.append({
            "group_name": g["group_name"],
            "gid": g["gid"],
            "member_count": len(g.get("members", [])),
        })

    return {
        "success": True,
        "username": username,
        "primary_group": primary_group,
        "primary_gid": primary_gid,
        "groups": clean_groups,
        "group_count": len(clean_groups),
    }


def structure_accessible_dirs(username: str, user_groups: list,
                              accessible_dirs: list,
                              roots_checked: list) -> dict:
    """Structure accessible directory results without debug noise.

    Removes all_scanned_dirs and debug_log from the success response.

    Args:
        username: OS username.
        user_groups: List of group names the user belongs to.
        accessible_dirs: List of accessible directory dicts.
        roots_checked: List of root paths that were scanned.

    Returns:
        Clean dict without internal debugging data.
    """
    return {
        "success": True,
        "username": username,
        "your_groups": sorted(set(user_groups)),
        "accessible_level1_dirs": accessible_dirs,
        "count": len(accessible_dirs),
        "roots_checked": roots_checked,
    }


# ---------------------------------------------------------------------------
# HPC user/group directory lookup parsers
# ---------------------------------------------------------------------------

def parse_adquery_user(raw_output: str) -> dict:
    """Parse the output of 'adquery user <username> -A' into a structured dict.

    The output format is key:value pairs, one per line. Example:
        unixname:jsmith
        uid:163881175
        gid:163881175
        gecos:Valleru, Lohit
        home:/home/jsmith
        shell:/bin/bash
        dn:CN=jsmith,OU=Staff,OU=HPC,...
        canonicalName:example.org/Staff/Information Systems/jsmith

    Args:
        raw_output: Raw stdout from 'adquery user <username> -A'.

    Returns:
        Structured dict with parsed user fields.
    """
    fields = {}
    for line in raw_output.strip().splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        # Split on first colon only (values like DN contain colons)
        key, _, value = line.partition(':')
        fields[key.strip()] = value.strip()

    # Extract department from canonicalName
    # Format: example.org/Staff/Information Systems/jsmith
    department = ""
    canonical = fields.get("canonicalName", "")
    if canonical:
        parts = canonical.split("/")
        # Department is typically the second-to-last segment
        # e.g. .../Staff/Information Systems/jsmith → "Information Systems"
        if len(parts) >= 3:
            department = parts[-2]

    return {
        "success": True,
        "username": fields.get("unixname", ""),
        "full_name": fields.get("gecos", ""),
        "uid": int(fields["uid"]) if "uid" in fields else None,
        "gid": int(fields["gid"]) if "gid" in fields else None,
        "home_dir": fields.get("home", ""),
        "shell": fields.get("shell", ""),
        "department": department,
        "dn": fields.get("dn", ""),
    }


def parse_adquery_group(raw_output: str) -> dict:
    """Parse the output of 'adquery group <groupname> -A' into a structured dict.

    The output format is key:value pairs, one per line. Example:
        unixname:grp_hpc_users
        gid:164073933
        required:false
        dn:CN=GRP_HPC_USERS,OU=Groups,...
        canonicalName:example.org/Resources/Groups/GRP_HPC_USERS
        unixMembers:azadianz,brophye,chakradm,...

    Args:
        raw_output: Raw stdout from 'adquery group <groupname> -A'.

    Returns:
        Structured dict with parsed group fields.
    """
    fields = {}
    for line in raw_output.strip().splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        key, _, value = line.partition(':')
        fields[key.strip()] = value.strip()

    # Parse comma-separated member list
    members_raw = fields.get("unixMembers", "")
    members = [m.strip() for m in members_raw.split(",") if m.strip()] if members_raw else []

    return {
        "success": True,
        "group_name": fields.get("unixname", ""),
        "gid": int(fields["gid"]) if "gid" in fields else None,
        "dn": fields.get("dn", ""),
        "members": sorted(members),
        "member_count": len(members),
    }


def structure_hpc_user_lookup(adquery_parsed: dict, api_primary_group: str = "", investigator_info: dict = None) -> dict:
    """Combine adquery user data with API primary_group and investigators data into a final user profile.

    Args:
        adquery_parsed: Dict from parse_adquery_user().
        api_primary_group: Primary group from the MongoDB API (optional enrichment).
        investigator_info: Dict with 'program' and 'department' from the investigators API.
            When provided, overwrites the AD-derived department (which is inaccurate for
            research program mapping). This is the authoritative source.

    Returns:
        Final structured user profile dict.
    """
    if investigator_info is None:
        investigator_info = {}

    # Start with AD-derived department as fallback
    department = adquery_parsed.get("department", "")
    # Overwrite with investigators API if available (authoritative source)
    if investigator_info.get("department"):
        department = investigator_info["department"]

    result = {
        "success": True,
        "username": adquery_parsed.get("username", ""),
        "full_name": adquery_parsed.get("full_name", ""),
        "uid": adquery_parsed.get("uid"),
        "gid": adquery_parsed.get("gid"),
        "home_dir": adquery_parsed.get("home_dir", ""),
        "shell": adquery_parsed.get("shell", ""),
        "department": department,
    }
    # Add program from investigators API if available
    if investigator_info.get("program"):
        result["program"] = investigator_info["program"]
    # Use API primary_group if provided, otherwise leave it out
    if api_primary_group:
        result["primary_group"] = api_primary_group

    return result


def structure_hpc_group_lookup(adquery_parsed: dict, api_investigator: str = "") -> dict:
    """Combine adquery group data with API investigator into a final group profile.

    Args:
        adquery_parsed: Dict from parse_adquery_group().
        api_investigator: Investigator/PI from the MongoDB API (optional enrichment).

    Returns:
        Final structured group profile dict.
    """
    result = {
        "success": True,
        "group_name": adquery_parsed.get("group_name", ""),
        "gid": adquery_parsed.get("gid"),
        "members": adquery_parsed.get("members", []),
        "member_count": adquery_parsed.get("member_count", 0),
    }
    if api_investigator:
        result["investigator"] = api_investigator

    return result


def filter_hpc_users(users_list: list, query: str = "",
                     group_filter: str = "") -> dict:
    """Filter the HPC users list from the MongoDB API.

    Args:
        users_list: List of dicts with 'username' and 'primary_group' keys
                    (from the /users API endpoint).
        query: Optional substring to match against username (case-insensitive).
        group_filter: Optional exact group name to filter by.

    Returns:
        Filtered list of users with match count.
    """
    filtered = users_list

    if group_filter:
        filtered = [u for u in filtered
                    if u.get("primary_group", "") == group_filter]

    if query:
        q_lower = query.lower()
        filtered = [u for u in filtered
                    if q_lower in u.get("username", "").lower()]

    return {
        "success": True,
        "users": filtered,
        "total": len(filtered),
        "query": query or None,
        "group_filter": group_filter or None,
    }
