"""Remove loose figure copies only after verifying retained and recovery copies."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent.parent
base = (root / "papers/project_figures").resolve()
assert base.is_relative_to(root)
report_path = root / "audit/cleanup-2026-10-27.json"
report = json.loads(report_path.read_text())
retained = [p for p in (root / "papers/final").rglob("*") if p.is_file()]
removed = []
with zipfile.ZipFile(root / "backups/cleanup-2026-10-27.zip") as archive:
    for path in list(base.iterdir()):
        if not path.is_file():
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        matches = [p for p in retained if p.stat().st_size == path.stat().st_size and hashlib.sha256(p.read_bytes()).hexdigest() == sha]
        if not matches:
            continue
        original = next(x["from"] for x in report["moved_files"] if x["to"] == path.relative_to(root).as_posix())
        assert hashlib.sha256(archive.read(original)).hexdigest() == sha
        assert path.resolve().parent == base
        removed.append({"removed": path.relative_to(root).as_posix(), "identical_retained_file": matches[0].relative_to(root).as_posix(), "original_archive_member": original, "sha256": sha})
        path.unlink()
report["post_move_image_deduplication"] = removed
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(removed, indent=2))
