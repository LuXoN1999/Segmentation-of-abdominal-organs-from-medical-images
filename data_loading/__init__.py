from pathlib import Path


def get_project_root() -> Path:
    current_file_path = Path(__file__).resolve()
    project_marker = "README.md"  # using README.md as marker for root level of project
    for parent in current_file_path.parents:
        if (parent / project_marker).exists():
            return parent.resolve()
    raise ValueError("Project root not found.")
