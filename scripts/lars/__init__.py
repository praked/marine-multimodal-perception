"""LaRS dataset tooling.

LaRS (Lakes, Rivers and Seas) is a maritime panoptic obstacle-detection
benchmark (Žust et al., ICCV 2023). We use only its 3-stuff *semantic*
labelling (obstacle / water / sky) to train an eWaSR water-segmentation
model that gives a robust water edge and per-object waterline-contact point
for monocular range (see docs/reference/segmentation.md, or the internal
docs/lars_segmentation_plan.md if you have it).

Submodules:
    prepare: stage LaRS into the MaSTr1325 layout eWaSR's loader expects.
    loader: pure-numpy reader for sanity checks / previews / tests.
    preview: render mask overlays (Agg) to confirm the class mapping.

LaRS semantic mask encoding (on disk): 0=obstacle, 1=water, 2=sky, 255=ignore.
"""

from __future__ import annotations

# Class ids in LaRS semantic masks (and the convention used throughout the
# segmentation code: scripts/utils/segmentation.py).
OBSTACLE = 0
WATER = 1
SKY = 2
LARS_IGNORE = 255

# eWaSR / MaSTr1325 use 4 for the ignore/unknown label. The eWaSR loader
# one-hots only {0,1,2}, so any other value is dropped from the loss either
# way; we remap to 4 purely to match the MaSTr convention its configs assume.
MASTR_IGNORE = 4
