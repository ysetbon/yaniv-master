"""Team fable, thread `yaniv`: candidate Yaniv rules as plain functions of a position record (the fields are
inputs of the net or simple functions of them), for the first-order check `contest_fable_yaniv_tab.flips`.
Constants are hand-picked from the tables in results/contest-fable-logs/yaniv.md."""

PT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 0]


def best_shed(r):
    g: dict = {}
    for k in r["hand_ranks"]:
        g[k] = g.get(k, 0) + PT[k]
    return max(g.values())


def make(a={1: 0.04, 2: 0.021, 3: 0.0055, 4: 0.0021, 5: 0.0}, ply_w=0.35, ply_lo=8, ply_hi=20, ply_mid=15,
         low=0.005, swap=2, hm5=False, pts_w=1.0, pts_mid=5.0):
    """call iff sure, or assaf < low, or (no swap veto and assaf < a[oc] * (pts_eff + ply_w * (ply_mid - clip(ply))))
    swap veto: best_shed - open >= swap (a lower hand is one throw away)."""
    def rule(r):
        if r["sure"] or r["assaf"] < low:
            return True
        if swap and best_shed(r) - r["open"] >= swap:
            return False
        if hm5:
            return r["did_call"]
        ply = min(ply_hi, max(ply_lo, r["ply"]))
        pts = pts_mid + pts_w * (r["pts"] - pts_mid)
        return r["assaf"] < a[r["oc"]] * (pts + ply_w * (ply_mid - ply))
    return rule
