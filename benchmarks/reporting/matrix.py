"""
Render the support matrix as native SVG (primary), Markdown, and an HTML page
embedding the SVG.

Reading the SVG:
- Fixed LR is the reference row at the top (neutral).
- Every other cell is coloured by its difference from Fixed LR in the same
  column, in percentage points: teal = better, rose = worse, grey = within
  0.25 pp. The cell shows the accuracy, the difference and the rank.
- Hatched "pending": planned, not run yet. Partly done cells say "2/3 seeds".
- Blank "n/a": the scenario does not apply to the method.
- The cost column draws epochs trained per cell on a log scale.
The SVG follows the viewer's light or dark theme.
"""
import html
import math

from benchmarks.reporting.labels import BASELINE, METHOD_LABELS, label
from benchmarks.reporting.table import NA, PARTIAL, PENDING, Matrix, cost_text

# difference vs baseline (pp) -> bin; each bin has a light and a dark colour
BINS = [(-math.inf, -2.0, "n3"), (-2.0, -1.0, "n2"), (-1.0, -0.25, "n1"), (-0.25, 0.25, "z"),
        (0.25, 1.0, "p1"), (1.0, 2.0, "p2"), (2.0, math.inf, "p3")]
LEGEND = [("n3", "≤ −2"), ("n2", "−2 to −1"), ("n1", "−1 to −¼"), ("z", "± ¼"),
          ("p1", "¼ to 1"), ("p2", "1 to 2"), ("p3", "≥ 2")]

STYLE = """
.bg{fill:#ffffff}.ink{fill:#1d2733}.soft{fill:#5d6b7a}.rule{stroke:#d8dee6}
.n3{fill:#c2414f}.n2{fill:#e0838c}.n1{fill:#f3c6ca}.z{fill:#e9edf2}.p1{fill:#bfe6dc}.p2{fill:#6cc3ab}.p3{fill:#1f8f74}
.ref{fill:#f4f6f9;stroke:#9aa6b4}.pend{fill:url(#hatch);stroke:#b3bdc8}.na{fill:none;stroke:#e1e6ec}
.hatch-line{stroke:#d6dde5}.bar{fill:#7d8fa3}.track{fill:#eef1f5}
.dark-text{fill:#ffffff}.light-text{fill:#1d2733}
@media (prefers-color-scheme: dark){
.bg{fill:#11161d}.ink{fill:#e6ebf1}.soft{fill:#93a1b0}.rule{stroke:#2a333e}
.n3{fill:#b8404d}.n2{fill:#8c3a44}.n1{fill:#4e2f36}.z{fill:#232b35}.p1{fill:#21453d}.p2{fill:#24735f}.p3{fill:#2fae8c}
.ref{fill:#1a212a;stroke:#5c6b7b}.pend{stroke:#3b4652}.na{stroke:#262e38}
.hatch-line{stroke:#2c3642}.bar{fill:#6f8399}.track{fill:#1c232c}
.dark-text{fill:#ffffff}.light-text{fill:#e6ebf1}}
"""
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

PAD, LABEL_W, CELL_W, CELL_H, GAP, COST_W = 28, 180, 112, 58, 8, 230
HEAD_H, LEGEND_H, GROUP_H, COLHEAD_H, REF_GAP, NOTE_H = 70, 52, 26, 40, 14, 34


def _bin(delta):
    for low, high, name in BINS:
        if low <= delta < high:
            return name
    return "z"


def _on_dark_fill(name):
    """Text on the saturated bins needs to be white in both themes."""
    return name in ("n3", "p3")


def _esc(s):
    return html.escape(str(s), quote=True)


def _text(x, y, s, size=13, cls="ink", weight="normal", anchor="start", family=FONT):
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{_esc(s)}</text>')


def render_svg(matrix: Matrix, title: str, subtitle: str = "") -> str:
    n_cols = len(matrix.columns)
    has_ref = BASELINE in matrix.rows
    others = [m for m in matrix.rows if m != BASELINE]
    width = PAD * 2 + LABEL_W + n_cols * (CELL_W + GAP) + COST_W
    top = PAD + HEAD_H + LEGEND_H + GROUP_H + COLHEAD_H
    rows_h = (CELL_H + GAP + REF_GAP if has_ref else 0) + len(others) * (CELL_H + GAP)
    height = top + rows_h + NOTE_H + PAD
    x0 = PAD + LABEL_W
    cost_x = x0 + n_cols * (CELL_W + GAP) + 10

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">',
           f'<title>{_esc(title)}</title>', f'<style>{STYLE}</style>',
           '<defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" '
           'patternTransform="rotate(45)"><rect width="8" height="8" class="bg"/>'
           '<line x1="0" y1="0" x2="0" y2="8" class="hatch-line" stroke-width="3"/></pattern></defs>',
           f'<rect width="{width}" height="{height}" class="bg"/>']

    # heading
    out.append(_text(PAD, PAD + 26, title, size=24, weight="700"))
    if subtitle:
        out.append(_text(PAD, PAD + 48, subtitle, size=12, cls="soft"))

    # legend: diverging scale vs baseline, pending, n/a
    ly = PAD + HEAD_H
    out.append(_text(PAD, ly + 4, f"vs {label(METHOD_LABELS, BASELINE)} (pp)", size=12, cls="soft"))
    x = PAD + 122
    for name, text in LEGEND:
        out.append(f'<rect x="{x}" y="{ly - 10}" width="54" height="18" rx="3" class="{name}"/>')
        out.append(_text(x + 27, ly + 3, text, size=10, cls="dark-text" if _on_dark_fill(name) else "light-text",
                         anchor="middle"))
        x += 58
    x += 18
    out.append(f'<rect x="{x}" y="{ly - 10}" width="54" height="18" rx="3" class="pend"/>')
    out.append(_text(x + 62, ly + 3, "pending", size=12, cls="soft"))
    x += 128
    out.append(f'<rect x="{x}" y="{ly - 10}" width="54" height="18" rx="3" class="na"/>')
    out.append(_text(x + 62, ly + 3, "n/a", size=12, cls="soft"))

    # dataset groups and column headings
    gy = PAD + HEAD_H + LEGEND_H
    for task in matrix.tasks:
        idx = [i for i, c in enumerate(matrix.columns) if c[0] == task]
        x1 = x0 + idx[0] * (CELL_W + GAP)
        x2 = x0 + (idx[-1] + 1) * (CELL_W + GAP) - GAP
        out.append(_text(x1, gy + 12, matrix.task_title(task), size=13, weight="700"))
        out.append(f'<line x1="{x1}" y1="{gy + 20}" x2="{x2}" y2="{gy + 20}" class="rule" stroke-width="1"/>')
    hy = gy + GROUP_H
    for j, column in enumerate(matrix.columns):
        out.append(_text(x0 + j * (CELL_W + GAP) + 6, hy + 18, matrix.column_title(column), size=12, cls="soft"))
    out.append(_text(cost_x, hy + 18, "training cost per cell", size=12, cls="soft"))

    max_epochs = max([c["epochs"] for c in matrix.cost.values() if c] or [1])

    def draw_row(method, y, reference):
        name = label(METHOD_LABELS, method)
        out.append(_text(PAD, y + CELL_H / 2 + (0 if reference else 5), name, size=14, weight="700"))
        if reference:
            out.append(_text(PAD, y + CELL_H / 2 + 16, "reference", size=11, cls="soft"))
        for j, (task, scenario) in enumerate(matrix.columns):
            c = matrix.cell(method, task, scenario)
            x = x0 + j * (CELL_W + GAP)
            rect = f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="4"'
            if c.status == PENDING:
                out.append(f'{rect} class="pend"><title>{_esc(name)}: not run yet</title></rect>')
                out.append(_text(x + CELL_W / 2, y + CELL_H / 2 + 4, "pending", size=11, cls="soft", anchor="middle"))
                continue
            if c.status == NA:
                out.append(f'{rect} class="na"/>')
                out.append(_text(x + CELL_W / 2, y + CELL_H / 2 + 4, "n/a", size=11, cls="soft", anchor="middle"))
                continue
            if reference or c.delta is None:
                cls, ink = "ref", "light-text"
            else:
                cls = _bin(c.delta)
                ink = "dark-text" if _on_dark_fill(cls) else "light-text"
            detail = f"{c.n}/{c.planned} seeds" if c.status == PARTIAL else f"±{c.std:.2f}"
            tip = f"{name}: {c.mean:.2f}% ± {c.std:.2f} over {c.n} seeds, rank {c.rank}"
            if c.delta is not None:
                tip += f", {c.delta:+.2f} pp vs {label(METHOD_LABELS, BASELINE)}"
            out.append(f'{rect} class="{cls}"><title>{_esc(tip)}</title></rect>')
            out.append(_text(x + 10, y + 25, f"{c.mean:.2f}", size=17, cls=ink, weight="700", family=MONO))
            out.append(_text(x + CELL_W - 10, y + 25, f"#{c.rank}", size=11, cls=ink, anchor="end"))
            sub = detail if reference or c.delta is None else f"{c.delta:+.2f} pp · {detail}"
            out.append(_text(x + 10, y + 46, sub, size=10.5, cls=ink))
        cost = matrix.cost.get(method)
        if cost:
            frac = math.log10(1 + cost["epochs"]) / math.log10(1 + max_epochs)
            out.append(f'<rect x="{cost_x}" y="{y + 14}" width="{COST_W - 30}" height="6" rx="3" class="track"/>')
            out.append(f'<rect x="{cost_x}" y="{y + 14}" width="{(COST_W - 30) * frac:.1f}" height="6" rx="3" '
                       f'class="bar"/>')
        out.append(_text(cost_x, y + 40, cost_text(cost), size=11.5, cls="soft"))

    y = top
    if has_ref:
        draw_row(BASELINE, y, reference=True)
        y += CELL_H + GAP + REF_GAP
        out.append(f'<line x1="{PAD}" y1="{y - REF_GAP / 2 - GAP / 2}" x2="{width - PAD}" '
                   f'y2="{y - REF_GAP / 2 - GAP / 2}" class="rule" stroke-dasharray="2 4"/>')
    for method in others:
        draw_row(method, y, reference=False)
        y += CELL_H + GAP

    out.append(_text(PAD, y + 22, "Test accuracy (%) on held-out data, mean ± std over seeds; colour = difference "
                                  "from the reference row in percentage points; #rank within the column.",
                     size=11, cls="soft"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_markdown(matrix: Matrix, title: str = "Support matrix") -> str:
    header = ["Method"] + [f"{matrix.task_title(t)} · {matrix.column_title((t, s))}" for t, s in matrix.columns] \
        + ["Training cost per cell"]
    lines = [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    rows = ([BASELINE] if BASELINE in matrix.rows else []) + [m for m in matrix.rows if m != BASELINE]
    for method in rows:
        name = label(METHOD_LABELS, method) + (" *(reference)*" if method == BASELINE else "")
        cells = []
        for task, scenario in matrix.columns:
            c = matrix.cell(method, task, scenario)
            if c.status in (NA, PENDING):
                cells.append("n/a" if c.status == NA else "pending")
                continue
            text = f"{c.mean:.2f} ± {c.std:.2f}"
            if c.delta is not None:
                text += f" ({c.delta:+.2f})"
            text += f" #{c.rank}"
            if c.status == PARTIAL:
                text += f" [{c.n}/{c.planned} seeds]"
            cells.append("**" + text + "**" if c.rank == 1 else text)
        lines.append("| " + " | ".join([name] + cells + [cost_text(matrix.cost.get(method))]) + " |")
    lines.append("")
    lines.append("Test accuracy (%) on the held-out split, mean ± std over seeds; (±x) = difference from Fixed LR "
                 "in percentage points; #rank within the column; **bold** = best; pending = not run yet; "
                 "n/a = the scenario does not apply.")
    return "\n".join(lines) + "\n"


def render_html(svg: str, title: str) -> str:
    """A standalone page around the SVG, for sharing or opening in a browser."""
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{_esc(title)}</title>'
            '<style>body{margin:0;display:flex;justify-content:center;padding:24px;background:#fff}'
            '@media (prefers-color-scheme: dark){body{background:#11161d}}'
            f'svg{{max-width:100%;height:auto}}</style></head><body>{svg}</body></html>\n')
