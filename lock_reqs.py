import sys
import subprocess

def get_installed_versions():
    output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    versions = {}
    for line in output.splitlines():
        if "==" in line:
            pkg, ver = line.split("==", 1)
            versions[pkg.lower()] = ver
        elif "@" in line:
            pkg = line.split("@")[0].strip()
            versions[pkg.lower()] = "latest"
    return versions

# Read current requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

installed = get_installed_versions()

new_lines = []
for line in lines:
    clean_line = line.strip()
    if not clean_line or clean_line.startswith("#"):
        new_lines.append(line)
        continue
    
    # parse package name
    pkg_name = clean_line.split(">=")[0].split("==")[0].split("<")[0].strip()
    
    # lookup exact version
    exact_ver = installed.get(pkg_name.lower())
    if exact_ver and exact_ver != "latest":
        new_lines.append(f"{pkg_name}=={exact_ver}\n")
    else:
        new_lines.append(line)

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("✅ Updated requirements.txt with exact versions.")
