"""Learned fusion-scoring model: training-pipeline stages.

Phase 0 (docs/plans/fusion_scoring_training.md): feature-table
export. Later phases add target building, the gated-mixture / GBT bake-off,
and evaluation. Nothing here changes runtime behaviour: the modules consume
the canonical ObstacleDetectionPipeline read-only.
"""
