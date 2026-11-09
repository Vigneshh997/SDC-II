# Workspace organization — 27 October 2026

Application source, datasets, checkpoints, Python environments, frontend dependencies, training outputs, tests and original audit evidence retain their locations. Existing uncommitted work was preserved; Git was not reset or committed.

| Location | Contents |
| --- | --- |
| `backend/`, `frontend/`, `tests/` | Application and tests, unchanged |
| `papers/final/mike2026/` | Final-named MIKE PDF, LaTeX and build assets |
| `papers/final/mike2026_reviewed/` | Reviewed anonymous PDF, LaTeX, figures and experimental evidence |
| `papers/final/springer/` | Springer journal PDF, LaTeX and build assets |
| `papers/references/` | External reference paper |
| `papers/project_figures/` | Preserved original project figures |
| `papers/submission_records/` | Submission records and private author notes |
| `papers/review_records/` | Submission audit |
| `docs/` | Project context, review guide, review parameters and architecture/workflow |
| `presentations/` | Project review PowerPoint |
| `audit/` | Existing evidence plus cleanup manifest, script and verification |
| `backups/cleanup-2026-10-27.zip` | Original paths and bytes of every moved or removed file, plus original root README and gitignore |

The cleanup moved 62 files and archived then removed 207 superseded, duplicate or temporary files from their former working locations. Older non-identical drafts were archived, not treated as identical duplicates. Local bibliography, style and class copies required by each retained paper remain intentionally. No dataset/model deduplication was attempted because it could break project references.

Two additional loose project figure copies were removed after confirming identical retained images in the Springer package and matching original bytes in the recovery ZIP. Their retained paths are recorded under `post_move_image_deduplication` in the manifest.

Before removal, the recovery ZIP passed CRC checking and every archived file was compared with the original SHA256. Copied files passed SHA256 checks before originals were removed. Sixty-four application, configuration and test files were checked before and after and were unchanged. Existing PDFs and LaTeX content were preserved.

See [the complete old-to-new mapping and removal manifest](audit/cleanup-2026-10-27.json). Historical audit manifests deliberately retain original paths and hashes as evidence of their original snapshot; use the cleanup mapping to find relocated files. Historical prose may also refer to old drafts now stored in the recovery archive.

For recovery, extract the ZIP into a separate empty folder first. Its members use the original project-relative paths. Consult `CLEANUP_MANIFEST.json` there before selectively copying files back; avoid extracting over current application work. Keep the archive until the cleaned layout has been reviewed.
