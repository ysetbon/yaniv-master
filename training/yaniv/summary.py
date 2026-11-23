"""A short Markdown summary of a training run, written when it finishes.

It goes to runs/<timestamp>/summary.md and to results/<timestamp>.md. The
results folder is tracked by git, so pushing it lets anyone (or Claude) read
how a run went without the run's folder of nets.
"""

from __future__ import annotations


def _margin(rounds: dict) -> float:
    return rounds["villain_avg"] - rounds["hero_avg"]


def _matches(matches: dict) -> str:
    total = matches["wins"] + matches["losses"] + matches["ties"]
    return f"{matches['wins']}–{matches['losses']} ({matches['win_rate']:.1%})" if total else "–"


def _minutes(seconds: float | None) -> str:
    if not seconds:
        return "–"
    return f"{seconds / 60:.1f} min" if seconds < 3600 else f"{int(seconds // 3600)} h {int(seconds % 3600 // 60)} min"


def render_summary(name: str, cfg: dict, report: dict, generations: list[dict], device: str, workers: int) -> str:
    lines = [f"# Training run {name}", ""]
    experts = report.get("experts") or 1
    net_kind = f"a decider and {experts} expert nets" if experts > 1 or (cfg.get("experts") or 0) > 1 else "one net"
    settings = [
        ("Recipe", cfg.get("recipe")),
        ("Net", net_kind),
        ("Device", device),
        ("Game processes", workers),
        ("Effort", cfg.get("effort") or "all cores but two"),
        ("Started from", cfg.get("start_from") or "–"),
        ("Copied", cfg.get("copy_from") or ("–" if cfg.get("start_from") else "the card counter")),
        ("Passes", cfg.get("generations")),
        ("Took", _minutes(report.get("seconds"))),
    ]
    if cfg.get("recipe") == "search":
        settings += [
            ("Rounds searched a pass", cfg.get("search_rounds")),
            ("Imagined deals a move", cfg.get("search_samples")),
            ("Confidence bar", cfg.get("search_confidence")),
            ("Teacher", cfg.get("search_teacher")),
            ("Counter moves mixed in", f"{cfg.get('expert_mix_start')} → {cfg.get('expert_mix_end')} per searched move"),
        ]
    settings.append(("Each pass tested on", f"{cfg.get('eval_rounds')} rounds and {cfg.get('eval_matches')} matches"))
    lines += ["| Setting | Value |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in settings] + [""]

    lines += ["## Generations", "",
              "Margin is the counter's points a round minus the net's: above 0, the net takes fewer points.", "",
              "| Gen | Made by | Margin a round | Rounds won | Matches won | Teacher agreement | Decisions learned | Kept | Took |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in generations:
        agree = g.get("teacher_agreement")
        lines.append(
            f"| {g['index']} | {g['label']} | {_margin(g['rounds']):+.2f} | {g['rounds']['win_rate']:.1%} | "
            f"{_matches(g['matches'])} | {f'{agree:.1%}' if agree is not None else '–'} | "
            f"{g.get('winning_decisions', '–')} | {'yes' if g['kept'] else ''} | {_minutes(g.get('seconds'))} |")
    lines.append("")

    replay = report.get("gate_replay")
    if replay:
        lines += [f"## Gate replay", "",
                  f"The top {len(replay['finalists'])} generations by their own tests replayed the same "
                  f"{replay['rounds']} rounds and {replay['matches']} matches.", "",
                  "| Gen | Margin a round | Rounds won | Matches won | Its own test's margin |",
                  "| --- | --- | --- | --- | --- |"]
        for gi in replay["finalists"]:
            r = replay["results"][str(gi)]
            mark = " (pick)" if gi == replay["chosen"] else ""
            lines.append(f"| {gi}{mark} | {_margin(r['rounds']):+.2f} | {r['rounds']['win_rate']:.1%} | "
                         f"{_matches(r['matches'])} | {_margin(r['training_rounds']):+.2f} |")
        lines.append("")

    fresh = report.get("fresh_test") or {}
    if fresh:
        lines += ["## Final check on new games", "", "| Gen | Margin a round | Rounds won | Matches won |",
                  "| --- | --- | --- | --- |"]
        for gi, t in fresh.items():
            lines.append(f"| {gi} | {_margin(t['rounds']):+.2f} | {t['rounds']['win_rate']:.1%} | {_matches(t['matches'])} |")
        lines.append("")
    lines += [f"**Gen {report.get('best_generation')} ships.** Against the greedy player it wins "
              f"{report.get('round_win_rate_vs_greedy', 0):.1%} of rounds, against random play "
              f"{report.get('round_win_rate_vs_random', 0):.1%}.", ""]
    return "\n".join(lines)
