from pathlib import Path


def export_log_txt(path: Path, lines: list[str]) -> None:
    # Save execution logs to a persistent text file.
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
