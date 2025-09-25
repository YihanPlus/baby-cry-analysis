import os, sys

def init_notebook(marker_files=("src", ".git")):
    """
    Set current working directory to project root and add src/ to Python path.
    Looks for a folder containing src/ or .git as a marker.
    """
    cur = os.getcwd()
    while True:
        if any(os.path.exists(os.path.join(cur, m)) for m in marker_files):
            break
        parent = os.path.dirname(cur)
        if parent == cur:  # reached system root
            raise RuntimeError("Project root not found")
        cur = parent

    os.chdir(cur)
    sys.path.append(os.path.join(cur, "src"))
    print(f"✅ Notebook initialized. Project root: {cur}")
    return cur