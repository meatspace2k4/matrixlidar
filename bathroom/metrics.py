"""Classification metrics for grounding/validation (pure stdlib).

Scores predicted states against ground-truth labels. Used by validate.py to
grade the rule-based engine today and the trained model later: same numbers,
same report, so the model is held to the same bar as the engine it replaces.
"""


def confusion(pairs, classes):
    """pairs: iterable of (truth, pred) state names. Returns dict[truth][pred]."""
    m = {t: {p: 0 for p in classes} for t in classes}
    for truth, pred in pairs:
        if truth not in m:
            m[truth] = {p: 0 for p in classes}
        if pred not in m[truth]:
            m[truth][pred] = 0
        m[truth][pred] += 1
    return m


def per_class(pairs, classes):
    """Precision/recall/F1/support per class."""
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    support = {c: 0 for c in classes}
    for truth, pred in pairs:
        support[truth] = support.get(truth, 0) + 1
        if truth == pred:
            tp[truth] = tp.get(truth, 0) + 1
        else:
            fp[pred] = fp.get(pred, 0) + 1
            fn[truth] = fn.get(truth, 0) + 1
    out = {}
    for c in classes:
        p_den = tp[c] + fp.get(c, 0)
        r_den = tp[c] + fn.get(c, 0)
        prec = tp[c] / p_den if p_den else 0.0
        rec = tp[c] / r_den if r_den else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": support.get(c, 0)}
    return out


def accuracy(pairs):
    pairs = list(pairs)
    if not pairs:
        return 0.0
    correct = sum(1 for t, p in pairs if t == p)
    return correct / len(pairs)


def macro_f1(pairs, classes):
    pc = per_class(pairs, classes)
    present = [c for c in classes if pc[c]["support"] > 0]
    if not present:
        return 0.0
    return sum(pc[c]["f1"] for c in present) / len(present)


def format_confusion(m, classes):
    """ASCII confusion matrix, rows = truth, cols = pred."""
    present = [c for c in classes if c in m]
    abbr = {c: c[:6] for c in present}
    head = "truth\\pred".ljust(12) + "".join(abbr[c].rjust(8) for c in present)
    lines = [head]
    for t in present:
        row = t[:11].ljust(12) + "".join(str(m[t].get(p, 0)).rjust(8) for p in present)
        lines.append(row)
    return "\n".join(lines)


def format_report(pairs, classes):
    pairs = list(pairs)
    pc = per_class(pairs, classes)
    lines = ["per-class:                prec   recall      f1   support"]
    for c in classes:
        if pc[c]["support"] == 0:
            continue
        lines.append("  %-22s %6.2f  %6.2f  %6.2f   %5d"
                     % (c, pc[c]["precision"], pc[c]["recall"],
                        pc[c]["f1"], pc[c]["support"]))
    lines.append("")
    lines.append(format_confusion(confusion(pairs, classes), classes))
    lines.append("")
    lines.append("accuracy = %.3f   macro-F1 = %.3f   n = %d"
                 % (accuracy(pairs), macro_f1(pairs, classes), len(pairs)))
    return "\n".join(lines)
