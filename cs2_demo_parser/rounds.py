from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

ROUND_START_EVENTS = {"round_start"}
ROUND_END_EVENTS = {"round_end", "round_end_verbose", "round_officially_ended"}
PRIMARY_ROUND_END_EVENTS = {"round_end", "round_end_verbose"}
FALLBACK_ROUND_START_EVENTS = {"round_freeze_end"}
FALLBACK_ROUND_END_EVENTS = {"cs_pre_restart", "cs_win_panel_match"}
WINNER_FIELDS = [
    "winner",
    "winner_team",
    "winnerteam",
    "winning_team",
    "team",
    "team_num",
    "teamnumber",
    "teamid",
    "winner_side",
    "side",
]


@dataclass
class RoundInfo:
    round_number: int
    tick_start: int
    tick_end: int | None = None
    winner_team: int | None = None
    end_event: str | None = None


def normalize_winner_team(event: dict[str, Any]) -> int | None:
    for field in WINNER_FIELDS:
        if field not in event or event[field] is None:
            continue
        value = event[field]
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"2", "t", "terrorist", "terrorists", "team_t", "tt"}:
                return 2
            if normalized in {"3", "ct", "counter-terrorist", "counter-terrorists", "team_ct"}:
                return 3
        try:
            numeric = int(value)
            if numeric in (2, 3):
                return numeric
        except (TypeError, ValueError):
            pass
    if "ct_wins" in event or "t_wins" in event:
        ct_wins = _as_bool(event.get("ct_wins"))
        t_wins = _as_bool(event.get("t_wins"))
        if ct_wins and not t_wins:
            return 3
        if t_wins and not ct_wins:
            return 2
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "t", "ct"}
    return False


def detect_rounds(events: list[dict[str, Any]], frames: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    sorted_events = sorted(
        [event for event in events if event.get("tick") is not None],
        key=lambda event: int(event["tick"]),
    )
    starts = _dedupe_events([event for event in sorted_events if event.get("event_name") in ROUND_START_EVENTS])
    ends = _dedupe_events([event for event in sorted_events if event.get("event_name") in PRIMARY_ROUND_END_EVENTS])
    if not ends:
        ends = _dedupe_events([event for event in sorted_events if event.get("event_name") == "round_officially_ended"])

    rounds = _rounds_from_starts_and_ends(starts, ends, frames)
    if not rounds:
        starts = _dedupe_events(
            [event for event in sorted_events if event.get("event_name") in FALLBACK_ROUND_START_EVENTS]
        )
        ends = _dedupe_events(
            [event for event in sorted_events if event.get("event_name") in FALLBACK_ROUND_END_EVENTS]
        )
        rounds = _rounds_from_starts_and_ends(starts, ends, frames)
    if not rounds:
        rounds = _rounds_from_starts_only(starts, frames)
    if not rounds and frames:
        ticks = [int(frame["tick"]) for frame in frames if frame.get("tick") is not None]
        if ticks:
            rounds.append(RoundInfo(round_number=1, tick_start=min(ticks), tick_end=max(ticks)))
    last_tick = _last_frame_tick(frames)
    for idx, rnd in enumerate(rounds):
        if rnd.tick_end is None:
            next_start = rounds[idx + 1].tick_start if idx + 1 < len(rounds) else None
            rnd.tick_end = (next_start - 1) if next_start is not None else last_tick
    return [asdict(rnd) for rnd in rounds]


def _rounds_from_starts_and_ends(
    starts: list[dict[str, Any]],
    ends: list[dict[str, Any]],
    frames: list[dict[str, Any]] | None,
) -> list[RoundInfo]:
    rounds: list[RoundInfo] = []
    start_ticks = [int(event["tick"]) for event in starts]
    previous_end = (_first_frame_tick(frames) or 0) - 1
    start_idx = 0
    for end_event in ends:
        end_tick = int(end_event["tick"])
        if end_tick <= previous_end:
            continue
        candidates: list[int] = []
        while start_idx < len(start_ticks) and start_ticks[start_idx] <= end_tick:
            if start_ticks[start_idx] > previous_end:
                candidates.append(start_ticks[start_idx])
            start_idx += 1
        start_tick = candidates[-1] if candidates else previous_end + 1
        if start_tick > end_tick:
            continue
        rounds.append(
            RoundInfo(
                round_number=len(rounds) + 1,
                tick_start=start_tick,
                tick_end=end_tick,
                winner_team=normalize_winner_team(end_event),
                end_event=end_event.get("event_name"),
            )
        )
        previous_end = end_tick
    return rounds


def _rounds_from_starts_only(
    starts: list[dict[str, Any]],
    frames: list[dict[str, Any]] | None,
) -> list[RoundInfo]:
    if not starts:
        return []
    last_tick = _last_frame_tick(frames)
    rounds: list[RoundInfo] = []
    deduped_starts = [int(event["tick"]) for event in starts]
    for idx, tick in enumerate(deduped_starts):
        next_tick = deduped_starts[idx + 1] if idx + 1 < len(deduped_starts) else None
        end_tick = (next_tick - 1) if next_tick is not None else last_tick
        rounds.append(RoundInfo(round_number=idx + 1, tick_start=tick, tick_end=end_tick))
    return rounds


def _dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for event in sorted(events, key=lambda item: int(item["tick"])):
        key = (
            str(event.get("event_name")),
            int(event["tick"]),
            str(event.get("round", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def assign_round_numbers(frames: list[dict[str, Any]], rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rounds, key=lambda rnd: rnd["tick_start"])
    frame_refs = [(idx, frame) for idx, frame in enumerate(frames) if frame.get("tick") is not None]
    frame_refs.sort(key=lambda item: int(item[1]["tick"]))
    round_idx = 0
    for frame in frames:
        frame["round_number"] = None
        frame["round_winner_team"] = None
    for _, frame in frame_refs:
        tick = frame.get("tick")
        tick_int = int(tick)
        while round_idx + 1 < len(ordered) and tick_int > int(ordered[round_idx].get("tick_end") or ordered[round_idx]["tick_start"]):
            round_idx += 1
        if not ordered:
            continue
        rnd = ordered[round_idx]
        if tick_int >= int(rnd["tick_start"]) and (rnd.get("tick_end") is None or tick_int <= int(rnd["tick_end"])):
            frame["round_number"] = rnd["round_number"]
            frame["round_winner_team"] = rnd.get("winner_team")
    return frames


def _first_frame_tick(frames: list[dict[str, Any]] | None) -> int | None:
    ticks = [int(frame["tick"]) for frame in frames or [] if frame.get("tick") is not None]
    return min(ticks) if ticks else None


def _last_frame_tick(frames: list[dict[str, Any]] | None) -> int | None:
    ticks = [int(frame["tick"]) for frame in frames or [] if frame.get("tick") is not None]
    return max(ticks) if ticks else None
