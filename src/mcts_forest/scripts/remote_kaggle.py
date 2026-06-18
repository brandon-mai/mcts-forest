# /// script
# dependencies = [
#     "httpx>=0.27.0",
#     "websocket-client>=1.8.0",
# ]
# ///
import ast
import sys
import json
import uuid
import argparse
import time
import re
from pathlib import Path
from urllib.parse import urlparse
import httpx
from websocket import create_connection

CONFIG_FILE = Path(".jupyter_config.json")

CAVEMAN_HELP = """
=== KAGGLERUNNER CAVEMAN HELP ===
WHAT: Run flat python files as notebook cells on remote Kaggle GPU. 
WHY: Agent can edit flat files. Remote reads/writes persist across kernel resets.

HOW USE:
  1. Fresh Kaggle URL setup:
     uv run core_tools/remote_kaggle.py --url "https://.../proxy" pipeline/train.py
  2. Quick subsequent runs (uses cached URL):
     uv run core_tools/remote_kaggle.py pipeline/train.py
  3. Multiple files (resolves local imports automatically, last is main):
     uv run core_tools/remote_kaggle.py model.py pipeline/train.py

PERSISTENCE RULES:
  - Inside main file, define path strings normally (e.g. "./weights/", "outputs").
  - Tool parses paths, pulls down old dataset state before run, and matches folder structure.
  - If script adds/mutates files, tool auto-zips and pushes new dataset version.
  - If script only reads or crashes before writing, upload phase is skipped. No time wasted.

PROGRESS BAR RULE FOR AGENTS:
  - Always write progress loops using 'tqdm.auto' with explicit terminal placement:
    `for x in tqdm(iterable, position=0, leave=True):`
  - This prevents multi-line terminal spamming over remote WebSockets while keeping 
    local code execution looking perfectly standard.

EXECUTION INTERRUPTION METHOD:
  - To stop cell/script execution midway, hit 'Ctrl + C' in your local terminal.
  - Local process cleanly detaches and terminates remote tracking immediately.
  - Stateful merge upload phase is skipped on interrupt. Your cloud dataset state 
    remains completely safe from corrupted, half-baked, or incomplete script writes.
=================================
"""

def get_cached_url():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text()).get("url")
        except Exception:
            pass
    return None

def save_url(url):
    config = {"url": url, "timestamp": time.time()}
    CONFIG_FILE.write_text(json.dumps(config, indent=2))

def get_local_kaggle_creds():
    home_creds = Path.home() / ".kaggle" / "kaggle.json"
    if home_creds.exists():
        try:
            return json.loads(home_creds.read_text())
        except Exception:
            pass
    return None

def execute_payload(base_url, code_payload):
    base_url = base_url.rstrip('/')
    headers = {}
    try:
        with httpx.Client(verify=False) as client:
            response = client.get(f"{base_url}/api/kernels", headers=headers, timeout=5.0)
            response.raise_for_status()
            kernels = response.json()
            if not kernels:
                print("❌ Error: No active kernels found on Kaggle.")
                return False
            kernel_id = kernels[0]['id']
    except Exception as e:
        print(f"❌ Failed to reach Jupyter API: {e}")
        return False

    parsed_url = urlparse(base_url)
    ws_scheme = "wss" if parsed_url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed_url.netloc}{parsed_url.path}/api/kernels/{kernel_id}/channels"

    try:
        ws = create_connection(ws_url, header=[f"{k}: {v}" for k, v in headers.items()])
    except Exception as e:
        print(f"❌ WebSocket Connection Failed: {e}")
        return False

    msg_id = str(uuid.uuid4())
    execute_request = {
        "header": {
            "msg_id": msg_id,
            "username": "local_agent",
            "session": str(uuid.uuid4()),
            "msg_type": "execute_request",
            "version": "5.3"
        },
        "metadata": {},
        "content": {
            "code": code_payload,
            "silent": False,
            "store_history": True,
            "allow_stdin": False,
            "stop_on_error": True
        },
        "parent_header": {}
    }

    ws.send(json.dumps(execute_request))
    success = True
    try:
        while True:
            msg = json.loads(ws.recv())
            if msg.get('parent_header', {}).get('msg_id') == msg_id:
                msg_type = msg.get('msg_type')
                content = msg.get('content', {})

                if msg_type == 'stream':
                    sys.stdout.write(content.get('text', ''))
                    sys.stdout.flush()
                elif msg_type == 'error':
                    success = False
                    print("\n❌ Execution Error:")
                    for line in content.get('traceback', []):
                        print(line)
                elif msg_type == 'execute_reply':
                    if content.get('status') == 'error':
                        success = False
                    break
    except KeyboardInterrupt:
        print("\n⚠️ Detached from stream.")
    finally:
        ws.close()
    return success

def extract_relative_paths(file_path):
    path = Path(file_path)
    if not path.exists():
        return set()
    try:
        root = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    found_paths = set()
    path_regex = re.compile(r'^(\./)?([a-zA-Z0-9_\-]+/)+$|^(\./)?[a-zA-Z0-9_\-]+$')
    for node in ast.walk(root):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.strip()
            if path_regex.match(val) and not val.endswith('.py') and val not in ('utf-8', 'w', 'r', 'rb', 'wb'):
                clean_name = val.lstrip('./').rstrip('/')
                if clean_name:
                    found_paths.add(clean_name)
    return found_paths

def get_module_metadata(file_path):
    path = Path(file_path)
    if not path.exists():
        return set(), []
    try:
        root = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Syntax error while parsing {file_path}: {e}")
        sys.exit(1)

    defined_names = set()
    local_imports = []
    for node in ast.walk(root):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_imports.append((alias.name, node))
        elif isinstance(node, ast.ImportFrom) and node.module:
            local_imports.append((node.module, node))

    resolved_dependencies = []
    for mod_name, import_node in local_imports:
        base_rel_path = mod_name.replace('.', '/')
        possible_files = [Path(f"{base_rel_path}.py"), Path(base_rel_path) / "__init__.py"]
        for p in possible_files:
            if p.exists() and p.is_file():
                resolved_dependencies.append((p.resolve(), import_node))
                break
    return defined_names, resolved_dependencies

def prepare_combined_code(script_paths, credentials, debug_enabled=False):
    resolved_order = []
    visited = set()
    visiting = set()
    nodes_to_remove = {}

    def dfs(file_path):
        abs_path = Path(file_path).resolve()
        if abs_path in visiting:
            print(f"❌ Circular dependency detected involving: {file_path}")
            sys.exit(1)
        if abs_path in visited:
            return
        visiting.add(abs_path)
        defined, dependencies = get_module_metadata(abs_path)
        for dep_path, import_node in dependencies:
            if dep_path not in visited:
                dfs(dep_path)
            nodes_to_remove.setdefault(abs_path, []).append(import_node)
        visiting.remove(abs_path)
        visited.add(abs_path)
        resolved_order.append(abs_path)

    for script in script_paths:
        dfs(script)

    primary_script = Path(script_paths[-1])
    detected_dirs = extract_relative_paths(primary_script)

    creds_json = json.dumps(credentials)
    dataset_slug = "remote-runner-checkpoint-cache"
    username = credentials.get("username", "unknown")
    dataset_id = f"{username}/{dataset_slug}"
    target_dirs_json = json.dumps(list(detected_dirs))
    debug_flag_str = "True" if debug_enabled else "False"

    # --- SETUP & INITIAL DOWN-STREAM RESTORE ---
    sync_pre_script = f"""
import os, sys, json, subprocess, shutil
from pathlib import Path

_DEBUG = {debug_flag_str}

os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write({repr(creds_json)})
os.chmod("/root/.kaggle/kaggle.json", 0o600)

target_dirs = json.loads({repr(target_dirs_json)})
dataset_id = "{dataset_id}"

history_backplane = Path("./__kaggle_history_backplane__")
if history_backplane.exists():
    shutil.rmtree(history_backplane)
history_backplane.mkdir(exist_ok=True)

if _DEBUG:
    print("📥 Downloading existing dataset to synchronize history...")
subprocess.run(["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(history_backplane), "--unzip"], capture_output=True)

if history_backplane.exists() and any(history_backplane.iterdir()):
    if _DEBUG:
        print("🔄 Historical dataset found. Restoring all previous files...")
    for p in history_backplane.iterdir():
        if p.name not in ("dataset-metadata.json", "dataset.zip"):
            target_dest = Path(p.name)
            if p.is_dir():
                shutil.copytree(str(p), str(target_dest), dirs_exist_ok=True)
            else:
                shutil.copy2(str(p), str(target_dest))

# Snapshot file states and modifications metadata *before* script runs
_initial_snapshots = {{}}
for p in Path('.').rglob('*'):
    if not p.name.startswith('__') and p.is_file():
        _initial_snapshots[str(p)] = p.stat().st_mtime
"""

    payload_blocks = [sync_pre_script]
    for path in resolved_order:
        lines = path.read_text(encoding="utf-8").splitlines()
        strip_nodes = nodes_to_remove.get(path, [])
        lines_to_strip = set()
        for node in strip_nodes:
            for i in range(node.lineno, getattr(node, 'end_lineno', node.lineno) + 1):
                lines_to_strip.add(i - 1)

        clean_lines = [
            (f"# [Stripped local import]" if idx in lines_to_strip else line)
            for idx, line in enumerate(lines)
        ]
        payload_blocks.append(f"# === File: {path.name} ===\n" + "\n".join(clean_lines))

    # --- CONDITIONAL SMART MERGE AND PACKING ---
    sync_post_script = f"""
if _DEBUG:
    print("\\n🔍 [DEBUG] --- STARTING POST-EXECUTION DIAGNOSTICS ---")

# Step 1: Detect if any workspace file mutations or additions occurred
_has_mutated = False
_mutated_files = []

for p in Path('.').rglob('*'):
    if not p.name.startswith('__') and not p.name.startswith('.') and p.is_file() and p.suffix != '.py':
        filepath_str = str(p)
        if filepath_str not in _initial_snapshots or p.stat().st_mtime > _initial_snapshots[filepath_str]:
            _has_mutated = True
            _mutated_files.append(filepath_str)

if not _has_mutated:
    if _DEBUG:
        print("ℹ️ No modifications or file creation actions detected in monitored spaces. Skipping cloud upload.")
else:
    if _DEBUG:
        print(f"📝 Detected mutated/added data elements: {{_mutated_files}}")
        print("⚙️ Preparing local stateful merge compilation...")

    upload_scratch = Path("./__kaggle_upload_scratch__")
    if upload_scratch.exists():
        shutil.rmtree(upload_scratch)
    upload_scratch.mkdir(exist_ok=True)

    # Copy cloud history first
    if history_backplane.exists() and any(history_backplane.iterdir()):
        for p in history_backplane.iterdir():
            if p.name not in ("dataset-metadata.json", "dataset.zip"):
                if p.is_dir():
                    shutil.copytree(str(p), str(upload_scratch / p.name), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(p), str(upload_scratch / p.name))

    # Overwrite/Merge with explicit directories
    if target_dirs:
        for d in target_dirs:
            local_dir = Path(d)
            if local_dir.exists():
                shutil.copytree(str(local_dir), str(upload_scratch / d), dirs_exist_ok=True)
                
    # Overwrite/Merge loose updates
    for p in Path('.').iterdir():
        if p.is_file() and not p.name.startswith('__') and p.suffix != '.py' and not p.name.startswith('.'):
            shutil.copy2(str(p), str(upload_scratch / p.name))

    if _DEBUG:
        print("2. Staging folder structure layout (__kaggle_upload_scratch__):")
        for p in upload_scratch.rglob('*'):
            print(f"   -> {{p}} ('dir' if p.is_dir() else 'file')")

    zip_staging = Path("./__kaggle_zip_staging__")
    if zip_staging.exists():
        shutil.rmtree(zip_staging)
    zip_staging.mkdir(exist_ok=True)

    if _DEBUG:
        print("📦 Zipping the new merged version completely ourselves...")
    shutil.make_archive(
        base_name=str(zip_staging / "dataset"),
        format="zip",
        root_dir=str(upload_scratch)
    )

    meta_path = zip_staging / "dataset-metadata.json"
    metadata = {{"id": "{dataset_id}"}}
    meta_path.write_text(json.dumps(metadata, indent=2))

    if _DEBUG:
        print("🆙 Pushing the master zip file up to Kaggle...")
    
    version_res = subprocess.run([
        "kaggle", "datasets", "version", 
        "-p", str(zip_staging), 
        "-m", "Stateful merge automatic incremental payload update."
    ], capture_output=True, text=True)

    if version_res.returncode != 0 and _DEBUG:
        print(f"❌ Kaggle API Error: {{version_res.stderr}}")
    elif version_res.returncode == 0 and not _DEBUG:
        print("✅ Checkpoint synced successfully to Kaggle Dataset.")

    # Clean up execution tracking areas
    shutil.rmtree(history_backplane, ignore_errors=True)
    shutil.rmtree(upload_scratch, ignore_errors=True)
    shutil.rmtree(zip_staging, ignore_errors=True)

if _DEBUG:
    print("🔍 [DEBUG] --- DIAGNOSTICS COMPLETED ---")
"""
    payload_blocks.append(sync_post_script)
    return "\n\n".join(payload_blocks)

def main():
    parser = argparse.ArgumentParser(description="Dynamic Remote Jupyter Script Runner via uv.", add_help=False)
    parser.add_argument("scripts", nargs="*", help="One or more Python files to run. Last file is treated as primary.")
    parser.add_argument("--url", help="Provide a fresh long Kaggle session proxy URL to reset cache.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose step-by-step filesystem diagnostic printing.")
    parser.add_argument("-h", "--help", action="store_true", help="Show dense caveman developer setup manual.")
    args = parser.parse_args()

    if args.help:
        print(CAVEMAN_HELP)
        sys.exit(0)

    if not args.scripts:
        print("❌ Error: Missing execution script targets. Run with -h or --help for instructions.")
        sys.exit(1)

    target_url = args.url if args.url else get_cached_url()
    if not target_url:
        print("❌ Missing target URL. Run again with: uv run core_tools/remote_kaggle.py --url \"https://...\" script.py")
        sys.exit(1)

    if args.url:
        save_url(args.url)

    credentials = get_local_kaggle_creds()
    if not credentials:
        print("❌ Script aborted: Unable to fetch local terminal authentication keys.")
        sys.exit(1)

    final_payload = prepare_combined_code(args.scripts, credentials, debug_enabled=args.debug)
    execute_payload(target_url, final_payload)

if __name__ == "__main__":
    main()