"""Cross-sensor track identity: camera tracks <-> radar targets.

The association layer (scripts/utils/association.py) decides per FRAME
which radar returns project inside which camera detection; the target
tracker (target_motion.py) decides per SENSOR which radar clusters are the
same object over time. Nothing said "camera detection D and radar track T
are the same OBJECT across frames" -- the deliberate non-goal of the
2026-08-24 target-motion session, built here.

Identity is a persistent pairing between

  - a **camera track**: BBoxTracker (tracker.py, kept for exactly this)
    run over the per-frame blob detections (centre+diameter -> padded
    xyxy), and
  - a **radar target**: a TargetMotionTracker track id,

established by sustained per-frame association agreement and broken by
sustained disagreement. The per-frame agreement evidence is PHYSICAL, not
positional: detection D agrees with target T when the radar returns that
project inside D's pixel gate (DetectionMatch.point_indices, surfaced as
FisheyeResult.radar_point_indices) intersect the cluster points matched to
T this frame (TargetState.point_indices). Same index space -- both index
the frame's filtered cloud -- so the intersection is exact.

Hysteresis (all knobs in detection.yaml `motion.targets.identity`):

  - **form**: a candidate (camera, radar) pair confirms after
    `pair_min_hits` agreeing frames within the last `pair_window`
    co-observed frames (N-of-M; house 3-of-5 tracker numbers). Pairing is
    exclusive both ways; when several candidates confirm at once the one
    with the most agreement (then the largest current point overlap) wins.
  - **break**: `break_misses` CONSECUTIVE co-observed frames without
    overlap. Frames where either side is unobserved (camera missed the
    blob, radar dropped the cluster) count neither way: a radar dropout
    must not divorce the pair -- surviving it is the point.
  - **death**: the pair dies when its camera track is pruned, or when the
    radar target has not been reported for `stale_after` frames.

Each pair also learns the systematic radar-camera bearing offset (EMA of
target bearing minus detection bearing on agreeing frames; the 2026-07-09
association A/B measured a few degrees of residual bearing bias between
the models, so feeding raw camera bearings into the radar track would
step its rate fit). The offset-corrected camera bearing is what
`camera_obs` hands to TargetMotionTracker.apply_camera_observations when
`motion.targets.camera_bearing_rate` is on.

Everything degrades gracefully: no radar targets -> camera tracks keep
their ids, no pairs form or break; no camera detections -> targets stay
unpaired; association off -> no point indices, so pairing evidence never
accumulates (documented in detection.yaml: identity wants
fusion.association.enabled).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from scripts.sensor_processing.tracker import BBoxTracker


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class IdentityConfig:
    """Knobs for the identity layer (detection.yaml motion.targets.identity).

    Defaults mirror the YAML; each number's rationale lives there.
    """
    pair_min_hits: int = 3
    pair_window: int = 5
    break_misses: int = 5
    min_overlap_points: int = 1
    stale_after: int = 15
    bbox_min_hits: int = 3
    bbox_max_age: int = 5
    bbox_pad_px: float = 8.0
    offset_alpha: float = 0.3

    @classmethod
    def from_config(cls, cfg: dict | None) -> "IdentityConfig":
        cfg = cfg or {}
        return cls(
            pair_min_hits=int(cfg.get("pair_min_hits", 3)),
            pair_window=int(cfg.get("pair_window", 5)),
            break_misses=int(cfg.get("break_misses", 5)),
            min_overlap_points=int(cfg.get("min_overlap_points", 1)),
            stale_after=int(cfg.get("stale_after", 15)),
            bbox_min_hits=int(cfg.get("bbox_min_hits", 3)),
            bbox_max_age=int(cfg.get("bbox_max_age", 5)),
            bbox_pad_px=float(cfg.get("bbox_pad_px", 8.0)),
            offset_alpha=float(cfg.get("offset_alpha", 0.3)),
        )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class _Pair:
    camera_track_id: int
    radar_track_id: int
    # EMA of (radar target bearing - camera detection bearing) on agreeing
    # frames: the learned systematic offset between the two bearing models.
    bearing_offset_deg: float = 0.0
    frames_paired: int = 0            # agreeing frames since formation
    conflict_misses: int = 0          # consecutive co-observed disagreements
    radar_missing: int = 0            # frames since the radar target reported


@dataclass
class IdentityFrame:
    """Per-frame identity output (aligned with the update() inputs)."""
    # Per-detection camera track id (BBoxTracker; never None once a
    # detection spawns/matches a track).
    camera_track_ids: list[int | None] = field(default_factory=list)
    # Per-detection paired radar target id (None = detection's camera
    # track is not paired).
    radar_target_ids: list[int | None] = field(default_factory=list)
    # radar track_id -> camera track_id for every live pair.
    pairs: dict[int, int] = field(default_factory=dict)
    # radar track_id -> offset-corrected camera bearing (deg, boat frame)
    # for pairs whose camera track was observed THIS frame: the second
    # observation stream for apply_camera_observations.
    camera_obs: dict[int, float] = field(default_factory=dict)


class TrackIdentityManager:
    """Stateful per clip, like the trackers it joins.

    Call `update(...)` once per frame after the target tracker has run;
    it returns the frame's IdentityFrame and keeps honesty counters
    (`pairs_formed`, `pairs_broken`, `id_switches`, `frames_paired`).
    """

    def __init__(self, config: IdentityConfig | None = None):
        self.config = config or IdentityConfig()
        c = self.config
        self.bbox = BBoxTracker(min_hits=c.bbox_min_hits,
                                max_age=c.bbox_max_age)
        self.pairs: dict[int, _Pair] = {}          # radar id -> pair
        # (radar id, camera id) -> deque of agreement booleans over the
        # last pair_window CO-OBSERVED frames.
        self._candidates: dict[tuple[int, int], deque] = {}
        self._cand_touch: dict[tuple[int, int], int] = {}
        self._frame = 0
        # Honesty counters (reported by the eval, never consumed by nav).
        self.pairs_formed = 0
        self.pairs_broken = 0
        self.id_switches = 0
        self.frames_paired = 0
        self._last_partner: dict[int, int] = {}       # radar id -> camera id
        self._last_partner_cam: dict[int, int] = {}   # camera id -> radar id

    # -- internals ---------------------------------------------------------

    def _det_boxes(self, coords, sizes) -> list[tuple[float, float, float, float]]:
        """Blob centre+diameter -> padded xyxy for the IoU tracker.

        The pad stabilises frame-to-frame IoU on small blobs: at the 3 fps
        capture cadence a crossing subject moves ~7 px/frame on the
        fisheye, which drops the IoU of an unpadded ~20 px blob box near
        the 0.3 gate; +8 px per side keeps the overlap comfortably above.
        """
        pad = self.config.bbox_pad_px
        out = []
        for (u, v), s in zip(coords, sizes):
            h = s / 2.0 + pad
            out.append((float(u) - h, float(v) - h, float(u) + h, float(v) + h))
        return out

    def _record_pairing(self, rid: int, cid: int) -> None:
        if self._last_partner.get(rid, cid) != cid:
            self.id_switches += 1
        if self._last_partner_cam.get(cid, rid) != rid:
            self.id_switches += 1
        self._last_partner[rid] = cid
        self._last_partner_cam[cid] = rid

    # -- public ------------------------------------------------------------

    def update(
        self,
        coords: list[tuple[int, int]],
        sizes: list[float],
        det_point_indices: list[list[int]],
        det_bearings: list[float],
        targets: list,
        timestamp: str | None = None,
    ) -> IdentityFrame:
        """Step the identity layer with one frame's evidence.

        `coords`/`sizes`/`det_bearings` are the camera detections
        (FisheyeResult); `det_point_indices` the per-detection matched
        radar point indices (FisheyeResult.radar_point_indices; empty
        when association is off/silent); `targets` the frame's reported
        TargetStates (FrameResult.targets.targets).
        """
        c = self.config
        self.bbox.update(self._det_boxes(coords, sizes))
        det_cids = list(self.bbox.last_det_track_ids)
        det_of_cid = {cid: i for i, cid in enumerate(det_cids)
                      if cid is not None}

        # Physical evidence this frame: radar targets with cluster points
        # (camera-sustained snapshots have none and prove nothing).
        tgt_points = {t.track_id: set(t.point_indices)
                      for t in targets if t.point_indices}
        tgt_bearing = {t.track_id: float(t.bearing_deg) for t in targets}
        reported_rids = {t.track_id for t in targets}
        overlap: dict[tuple[int, int], int] = {}
        for i, cid in enumerate(det_cids):
            if cid is None:
                continue
            pts = (set(det_point_indices[i])
                   if i < len(det_point_indices) and det_point_indices[i]
                   else set())
            if not pts:
                continue
            for rid, q in tgt_points.items():
                n = len(pts & q)
                if n:
                    overlap[(rid, cid)] = max(overlap.get((rid, cid), 0), n)

        alive_cids = {tr.track_id for tr in self.bbox.tracks}

        # 1. Maintain existing pairs.
        for rid, pair in list(self.pairs.items()):
            cid = pair.camera_track_id
            pair.radar_missing = (0 if rid in reported_rids
                                  else pair.radar_missing + 1)
            if cid not in alive_cids or pair.radar_missing >= c.stale_after:
                del self.pairs[rid]
                self.pairs_broken += 1
                continue
            r_obs = rid in tgt_points
            c_obs = cid in det_of_cid
            if not (r_obs and c_obs):
                continue          # dropout on either side: hold the pair
            if overlap.get((rid, cid), 0) >= c.min_overlap_points:
                pair.conflict_misses = 0
                pair.frames_paired += 1
                self.frames_paired += 1
                db = float(det_bearings[det_of_cid[cid]])
                sample = tgt_bearing[rid] - db
                a = c.offset_alpha
                pair.bearing_offset_deg = (
                    a * sample + (1.0 - a) * pair.bearing_offset_deg)
            else:
                pair.conflict_misses += 1
                if pair.conflict_misses >= c.break_misses:
                    del self.pairs[rid]
                    self.pairs_broken += 1

        # 2. Accumulate candidates on co-observed unpaired (rid, cid).
        paired_rids = set(self.pairs)
        paired_cids = {p.camera_track_id for p in self.pairs.values()}
        observed_cids = set(det_of_cid)
        for rid in tgt_points:
            if rid in paired_rids:
                continue
            for cid in observed_cids:
                if cid in paired_cids:
                    continue
                key = (rid, cid)
                dq = self._candidates.setdefault(
                    key, deque(maxlen=c.pair_window))
                dq.append(overlap.get(key, 0) >= c.min_overlap_points)
                self._cand_touch[key] = self._frame
        # Prune candidates whose ids died, got paired away, or went stale
        # (no co-observation for stale_after frames: dead radar ids are
        # invisible to us, so age them out by silence).
        self._frame += 1
        self._candidates = {
            (rid, cid): dq for (rid, cid), dq in self._candidates.items()
            if cid in alive_cids and rid not in paired_rids
            and cid not in paired_cids
            and self._frame - self._cand_touch.get((rid, cid), 0)
            <= c.stale_after}
        self._cand_touch = {k: v for k, v in self._cand_touch.items()
                            if k in self._candidates}

        # 3. Confirm the strongest candidates (exclusive both ways).
        eligible = [(sum(dq), overlap.get(key, 0), key)
                    for key, dq in self._candidates.items()
                    if sum(dq) >= c.pair_min_hits]
        for _n_agree, _ov, (rid, cid) in sorted(eligible, reverse=True):
            if rid in paired_rids or cid in paired_cids:
                continue
            db_i = det_of_cid.get(cid)
            offset = (tgt_bearing[rid] - float(det_bearings[db_i])
                      if db_i is not None and rid in tgt_bearing
                      and overlap.get((rid, cid), 0) >= c.min_overlap_points
                      else 0.0)
            self.pairs[rid] = _Pair(camera_track_id=cid, radar_track_id=rid,
                                    bearing_offset_deg=offset)
            self.pairs_formed += 1
            self._record_pairing(rid, cid)
            paired_rids.add(rid)
            paired_cids.add(cid)
            self._candidates = {
                k: dq for k, dq in self._candidates.items()
                if k[0] != rid and k[1] != cid}

        # 4. Frame outputs.
        pairs_map = {rid: p.camera_track_id for rid, p in self.pairs.items()}
        cam_to_rad = {cid: rid for rid, cid in pairs_map.items()}
        camera_obs = {}
        for rid, p in self.pairs.items():
            i = det_of_cid.get(p.camera_track_id)
            if i is not None:
                camera_obs[rid] = (float(det_bearings[i])
                                   + p.bearing_offset_deg)
        return IdentityFrame(
            camera_track_ids=det_cids,
            radar_target_ids=[cam_to_rad.get(cid) for cid in det_cids],
            pairs=pairs_map,
            camera_obs=camera_obs,
        )

    def annotate(self, targets: list) -> None:
        """Stamp camera_track_id onto TargetStates (post-update/apply)."""
        for t in targets:
            t.camera_track_id = (
                self.pairs[t.track_id].camera_track_id
                if t.track_id in self.pairs else None)

    def stats(self) -> dict:
        """Honesty counters for eval/telemetry."""
        return {
            "pairs_live": len(self.pairs),
            "pairs_formed": self.pairs_formed,
            "pairs_broken": self.pairs_broken,
            "id_switches": self.id_switches,
            "frames_paired": self.frames_paired,
        }
