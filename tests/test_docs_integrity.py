"""Documentation integrity: references resolve, and the index is complete.

The docs/ tree was reorganised into guides/ reference/ plans/ history/, and
paths are cited from code comments, config files, LaTeX sources and the radar
config headers as well as from other documents. A moved file that leaves a
stale citation behind is silent: nothing fails, the reader just follows a dead
path. These tests are the guard.

Resolution is against **git-tracked** paths, not the working directory. A
developer's checkout also holds gitignored files (internal planning documents,
the LaTeX writeups and their PDFs), so an on-disk existence check passes
locally and fails in CI, which is worse than having no check at all. What
matters is what someone who clones the repository actually receives.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".py", ".md", ".tex", ".yaml", ".yml", ".sh", ".cfg", ".txt",
                 ".conf", ".service", ".toml"}
INDEX_FILES = ["README.md", "docs/README.md", "docs/history/README.md",
               "deploy/README.md"]

#: Documents that are cited but deliberately NOT distributed: internal
#: planning artefacts, gitignored by policy alongside PLAN.md and CLAUDE.md.
#: Citing one is allowed, but the citation must say it is internal so a reader
#: is not sent after a file they will never have -- see
#: `test_untracked_citations_are_marked_internal`.
#:
#: Adding a path here is a deliberate act. If you find yourself wanting to,
#: consider instead whether the document should simply be tracked.
ALLOWED_UNTRACKED = {"docs/lars_segmentation_plan.md"}

# A path citation in prose or code, e.g. `docs/guides/pi_setup.md`.
CITATION = re.compile(r"(?:docs|deploy)/[A-Za-z0-9_./-]+\.(?:md|sh|conf|csv)")
# A relative markdown link: [text](path)
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _tracked(*patterns):
    out = subprocess.run(["git", "ls-files", *patterns], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.split("\n") if line]


@pytest.fixture(scope="module")
def tracked_paths():
    """Every tracked file, plus every directory containing one, so a link to
    `CAD/` resolves as readily as a link to a file."""
    files = set(_tracked())
    dirs = {str(parent) for rel in files for parent in Path(rel).parents
            if str(parent) != "."}
    return files | dirs


#: This module is skipped when scanning for citations: it declares the policy
#: (ALLOWED_UNTRACKED, the regexes) rather than referring readers anywhere, so
#: auditing it would only ever flag its own definitions.
SELF = "tests/test_docs_integrity.py"


@pytest.fixture(scope="module")
def tracked_text_files():
    return [rel for rel in _tracked()
            if rel != SELF and Path(rel).suffix in TEXT_SUFFIXES
            and (REPO / rel).is_file()]


def _citations(text):
    """Every docs/ or deploy/ path cited in `text` (LaTeX underscores undone)."""
    return CITATION.findall(text.replace(r"\_", "_"))


def test_no_dangling_doc_citations(tracked_text_files, tracked_paths):
    """Every docs/ or deploy/ path cited anywhere in the tree is one that a
    fresh clone actually receives."""
    dangling = {}
    for rel in tracked_text_files:
        try:
            text = (REPO / rel).read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for cited in _citations(text):
            if cited not in tracked_paths and cited not in ALLOWED_UNTRACKED:
                dangling.setdefault(cited, []).append(rel)
    assert not dangling, "documentation paths cited but not in the repository:\n" + "\n".join(
        f"  {path}  (cited in {', '.join(sorted(set(where)))})"
        for path, where in sorted(dangling.items()))


def test_untracked_citations_are_marked_internal(tracked_text_files):
    """A citation of a deliberately-undistributed document must say so on the
    same line. Otherwise a new colleague follows the path, finds nothing, and
    cannot tell whether the file is missing or was never theirs to begin with.
    """
    unmarked = []
    for rel in tracked_text_files:
        try:
            lines = (REPO / rel).read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(lines, 1):
            for cited in _citations(line):
                if cited not in ALLOWED_UNTRACKED:
                    continue
                # The marker may sit on the line before or after when prose
                # wraps, so look at a small window.
                window = " ".join(lines[max(0, n - 2):n + 1]).lower()
                if "internal" not in window:
                    unmarked.append(f"{rel}:{n}  cites {cited}")
    assert not unmarked, (
        "citations of undistributed documents must be marked internal:\n  "
        + "\n  ".join(unmarked))


@pytest.mark.parametrize("index", INDEX_FILES)
def test_relative_links_resolve(index, tracked_paths):
    """Relative markdown links in the entry-point documents resolve to
    something a fresh clone receives."""
    path = REPO / index
    base = Path(index).parent
    broken = []
    for target in MD_LINK.findall(path.read_text()):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target = target.split("#", 1)[0].rstrip("/")
        if not target:
            continue
        candidates = {target, str((base / target)) if str(base) != "." else target}
        if not (candidates & tracked_paths):
            broken.append(target)
    assert not broken, f"{index} has broken relative links: {broken}"


def test_docs_index_covers_every_living_document():
    """docs/README.md is the entry point; a guide or spec that is not listed
    there is effectively invisible, which is how the flat docs/ tree became
    unnavigable in the first place."""
    index = (REPO / "docs" / "README.md").read_text()
    missing = []
    for section in ("guides", "reference", "plans"):
        for rel in _tracked(f"docs/{section}/*.md"):
            name = Path(rel).name
            if name != "README.md" and name not in index:
                missing.append(rel)
    assert not missing, f"not listed in docs/README.md: {missing}"


def test_history_index_covers_every_dated_log():
    index = (REPO / "docs" / "history" / "README.md").read_text()
    missing = [rel for rel in _tracked("docs/history/*.md")
               if Path(rel).name not in index and Path(rel).name != "README.md"]
    assert not missing, f"not listed in docs/history/README.md: {missing}"


def test_every_doc_directory_has_an_index():
    """guides/ reference/ plans/ are indexed from docs/README.md; history/
    carries its own index because it is append-only."""
    assert (REPO / "docs" / "README.md").is_file()
    assert (REPO / "docs" / "history" / "README.md").is_file()
