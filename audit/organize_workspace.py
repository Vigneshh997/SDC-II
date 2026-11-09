"""One-time, archive-first organization of project documents and paper artifacts."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
STAMP = "2026-10-27"
BACKUP = ROOT / "backups" / f"cleanup-{STAMP}.zip"
REPORT = ROOT / "audit" / f"cleanup-{STAMP}.json"


def checked(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError(f"Unsafe path: {path}")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(path):
    path = checked(ROOT / path)
    if not path.exists():
        raise FileNotFoundError(path)
    result = [path] if path.is_file() else list(path.rglob("*"))
    for item in result:
        if item.is_symlink():
            raise ValueError(f"Symlink requires manual review: {item}")
    return [p for p in result if p.is_file()]


moves = {}
removed = {}


def relocate(source, target, exclude_build=False):
    for old in files(source):
        relative = old.relative_to(ROOT / source) if old != ROOT / source else Path()
        new = checked(ROOT / target / relative) if relative != Path() else checked(ROOT / target)
        if exclude_build and (old.suffix in {".aux", ".log", ".out", ".blg", ".fls", ".fdb_latexmk"} or old.name.endswith(".synctex.gz")):
            removed[old] = "Regenerable LaTeX build output"
        else:
            if new.exists():
                raise FileExistsError(new)
            moves[old] = new


relocate("FINAL_PAPER_SPRINGER", "papers/final/springer", True)
relocate("MIKE_2026_REVISED", "papers/final/mike2026_reviewed", True)
# Keep the final conference source separate from the journal package.
springer = ROOT / "papers/final/springer"
conference = ROOT / "papers/final/mike2026"
conference_names = {"X_Fusion_MIKE2026_Final.tex", "X_Fusion_MIKE2026_Final.pdf", "X_Fusion_MIKE2026_Final.bbl", "architecture_diagram.tex", "optimization_traces.jpg", "llncs.cls", "splncs04.bst", "DescriptionTexts.txt"}
for old, new in list(moves.items()):
    if new.parent == springer and new.name in conference_names:
        moves[old] = conference / new.name
# Both final manuscripts need a local bibliography to remain self-contained.
copy_reference = ROOT / "FINAL_PAPER_SPRINGER/references.bib"

for name in ["PROJECT_COMPLETE_CONTEXT.md", "PROJECT_REVIEW_GUIDE.md", "project_review_parameters.md", "architecture_and_workflow.md"]:
    relocate(name, "docs/" + name)
for name in ["MIKE_2026_AUTHOR_NOTES.md", "MIKE_2026_COAUTHORS_PRIVATE.md"]:
    relocate(name, "papers/submission_records/" + name)
relocate("X_Fusion_Submission_Audit.md", "papers/review_records/X_Fusion_Submission_Audit.md")
for name in ["Conference Management Toolkit - Submission Summary.pdf", "cmt3.research.microsoft.com_MIKE2026_Track_1_Submission_Create.png"]:
    relocate(name, "papers/submission_records/" + name)
reference = "DAViT_A_Domain-Adapted_Vision_Transformer_for_Automated_Pneumonia_Detection_and_Explanation_Using_Chest_X-Ray_Images.pdf"
relocate(reference, "papers/references/" + reference)
relocate("X-Fusion_Project_Review_Presentation_V3_TEMPLATE_ORDER (2).pptx", "presentations/X-Fusion_Project_Review.pptx")
relocate("figures", "papers/project_figures")

obsolete = ["MIKE_2026_SUBMISSION", "MIKE_2026_EVIDENCE_UPDATED", "RESEARCH_PAPER", "tmp", "X_Fusion_MIKE2026_Reviewed.zip", "Download+the+journal+article+template+package+(December+2024+version).zip", "X_Fusion_IEEE_Camera_Ready.pdf", "X_Fusion_IEEE_Camera_Ready.tex", "X_Fusion_IEEE_Equation_Explained.pdf", "X_Fusion_IEEE_Equation_Explained.tex", "X_Fusion_IEEE_Equation_Explained.synctex.gz", "X_Fusion_MIKE2026_Anonymous.pdf", "X_Fusion_MIKE2026_Anonymous (1).pdf", "X_Fusion__Tri_Modal_Transformer_Fusion_forMulti_Label_Chest_X_ray_Analysis.pdf"]
for name in obsolete:
    for path in files(name):
        removed[path] = "Superseded paper, duplicate, original template package, or temporary paper work (retained in recovery ZIP)"

protected = {}
for directory in ["backend", "frontend/src", "frontend/public", "tests", ".github"]:
    base = ROOT / directory
    if base.exists():
        for p in base.rglob("*"):
            rel = p.relative_to(base)
            if p.is_file() and not any(part in {"__pycache__", "nih_cxr14", "checkpoints", "external", "paired"} for part in rel.parts):
                protected[p] = digest(p)
for name in ["config.yaml", "pyproject.toml", "requirements.txt", "requirements-dev.txt", "requirements-verified.txt", "HOW_TO_START.md", "test_all_cases.py", "frontend/package.json", "frontend/package-lock.json", "frontend/next.config.ts"]:
    protected[ROOT / name] = digest(ROOT / name)

originals = {p: digest(p) for p in set(moves) | set(removed) | {ROOT / "README.md", ROOT / ".gitignore"}}
plan = {"date": STAMP, "recovery_archive": str(BACKUP), "moved_files": [{"from": p.relative_to(ROOT).as_posix(), "to": new.relative_to(ROOT).as_posix(), "sha256": originals[p]} for p, new in sorted(moves.items())], "archived_and_removed_files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": originals[p], "reason": reason} for p, reason in sorted(removed.items())], "protected_file_count": len(protected)}
if "--apply" not in sys.argv:
    print(json.dumps({"move": len(moves), "archive_then_remove": len(removed), "protected": len(protected), "archive_bytes_before_compression": sum(p.stat().st_size for p in originals), "destinations": sorted({str(p.parent.relative_to(ROOT)) for p in moves.values()})}, indent=2))
    sys.exit()

if BACKUP.exists() or REPORT.exists():
    raise FileExistsError("Cleanup already run; inspect the existing recovery archive and report.")
BACKUP.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(BACKUP, "x", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(originals):
        z.write(p, p.relative_to(ROOT).as_posix())
    z.writestr("CLEANUP_MANIFEST.json", json.dumps(plan, indent=2))
with zipfile.ZipFile(BACKUP) as z:
    assert z.testzip() is None
    for p, sha in originals.items():
        assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest() == sha
print("Recovery ZIP verified against every original SHA256.", flush=True)
reference_bytes = copy_reference.read_bytes()
for old, new in moves.items():
    checked(old)
    checked(new)
    assert digest(old) == originals[old]
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old, new)
    assert digest(new) == originals[old]
(conference / "references.bib").write_bytes(reference_bytes)

# Repair existing relative Markdown links in relocated documents when the original target exists.
for old, new in moves.items():
    if new.suffix != ".md":
        continue
    content = new.read_text(encoding="utf-8")
    def fix_link(match):
        target = match.group(2)
        if "://" in target or target.startswith("#"):
            return match.group(0)
        raw, sep, anchor = target.partition("#")
        prior = (old.parent / raw).resolve()
        actual = moves.get(prior, prior)
        if not actual.exists() or prior in removed:
            return match.group(0)
        import os
        relative = Path(os.path.relpath(actual, new.parent)).as_posix()
        return match.group(1) + relative + (sep + anchor if sep else "") + ")"
    content = re.sub(r"(\]\()([^\)]+)\)", fix_link, content)
    new.write_text(content, encoding="utf-8")

for p in sorted(set(moves) | set(removed)):
    checked(p)
    assert digest(p) == originals[p], f"File changed during cleanup: {p}"
    p.unlink()
# Remove only now-empty directories beneath the explicitly selected old paths.
for name in ["FINAL_PAPER_SPRINGER", "MIKE_2026_REVISED", "figures"] + obsolete:
    base = checked(ROOT / name)
    if base.is_dir():
        for p in sorted(base.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                checked(p).rmdir()
        if not any(base.iterdir()):
            base.rmdir()

assert all(digest(p) == sha for p, sha in protected.items()), "Protected project file changed"
plan["verification"] = {"backup_crc_and_sha256": "passed", "copied_files_sha256": "passed", "protected_project_file_hashes": "passed", "shared_bibliography_copied_to_conference_package": True}
REPORT.write_text(json.dumps(plan, indent=2), encoding="utf-8")
print(json.dumps({"moved": len(moves), "archived_and_removed": len(removed), "protected_unchanged": len(protected), "backup": str(BACKUP), "report": str(REPORT)}, indent=2))
