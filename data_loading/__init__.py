from pathlib import Path


def _get_project_path() -> str:
    current_file_path = Path(__file__).resolve()
    project_marker = 'README.md'  # using README.md as marker for base level of project
    for parent in current_file_path.parents:
        if (parent / project_marker).exists():
            return str(parent.resolve())
    raise ValueError("Project path not found.")
