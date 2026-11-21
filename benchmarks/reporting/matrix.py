"""
Render the support matrix as native SVG (the primary artefact), plus Markdown
and an HTML page that embeds the SVG. Colour codes:

- cell shade: rank within the column (lightest = rank 1 = best)
- red triangle: worse than the Fixed LR baseline in that column
- grey "n/a": the scenario does not apply to the method
- dashed row: the Fixed LR baseline
"""
import html

from benchmarks.reporting.labels import BASELINE, METHOD_LABELS, label
from benchmarks.reporting.table import Matrix, cost_text

RANK_COLOURS = ["#c8dbfa", "#9dbdf3", "#6f9de6", "#4a7fd6", "#3463b6", "#274d92", "#1f3d74", "#18305c"]
GROUP_COLOURS = ["#5ab37a", "#e07b5f", "#c9a227", "#8f7ad6"]       # one bar colour per task
BG, PANEL, BORDER = "#0f1319", "#161b22", "#2a313c"
TEXT, MUTED, WORSE, NA_FILL = "#e6edf3", "#9aa6b4", "#f0524f", "#232a34"
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# layout (px)
PAD, LABEL_W, CELL_W, CELL_H, GAP, COST_W = 32, 190, 118, 64, 6, 280
TITLE_H, LEGEND_H, GROUP_H, HEADER_H, BASELINE_GAP, NOTE_H = 64, 44, 34, 46, 18, 40


def _rank_colour(rank):
    return RANK_COLOURS[min(rank, len(RANK_COLOURS)) - 1]


def _ink(rank):
    """Dark text on the light (top-ranked) shades, light text on the dark ones."""
    return "#0f1319" if rank is not None and rank <= 3 else TEXT


def _esc(s):
    return html.escape(str(s), quote=True)


def _text(x, y, s, size=13, fill=TEXT, weight="normal", anchor="start", family=FONT):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{_esc(s)}</text>')


def render_svg(matrix: Matrix, title: str, subtitle: str = "") -> str:
    n_cols = len(matrix.columns)
    body_rows = [m for m in matrix.rows if m != BASELINE]
    has_baseline = BASELINE in matrix.rows
    grid_w = LABEL_W + n_cols * (CELL_W + GAP) + COST_W
    width = PAD * 2 + grid_w
    rows_h = len(body_rows) * (CELL_H + GAP) + (BASELINE_GAP + CELL_H + GAP if has_baseline else 0)
    top = PAD + TITLE_H + LEGEND_H + GROUP_H + HEADER_H
    height = top + rows_h + NOTE_H + PAD

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">',
           f'<title>{_esc(title)}</title>',
           f'<rect width="{width}" height="{height}" fill="{BG}"/>']

    # title
    out.append(_text(PAD, PAD + 30, title, size=26, weight="700"))
    if subtitle:
        out.append(_text(PAD, PAD + 52, subtitle, size=12, fill=MUTED))
    out.append(f'<line x1="{PAD}" y1="{PAD + TITLE_H - 4}" x2="{width - PAD}" y2="{PAD + TITLE_H - 4}" '
               f'stroke="{BORDER}"/>')

    # legend: rank scale, worse-than-baseline marker, n/a
    ly = PAD + TITLE_H + 24
    x = PAD
    out.append(_text(x, ly + 4, "rank 1", size=12, fill=MUTED))
    x += 46
    for i, colour in enumerate(RANK_COLOURS[:6]):
        out.append(f'<rect x="{x + i * 30}" y="{ly - 8}" width="26" height="16" rx="4" fill="{colour}"/>')
    x += 6 * 30 + 4
    out.append(_text(x, ly + 4, "rank 6+", size=12, fill=MUTED))
    x += 84
    out.append(f'<path d="M{x} {ly - 6} L{x + 12} {ly - 6} L{x + 6} {ly + 5} Z" fill="{WORSE}"/>')
    out.append(_text(x + 18, ly + 4, f"worse than {label(METHOD_LABELS, BASELINE)}", size=12, fill=MUTED))
    x += 170
    out.append(f'<rect x="{x}" y="{ly - 8}" width="26" height="16" rx="4" fill="{NA_FILL}" stroke="{BORDER}"/>')
    out.append(_text(x + 34, ly + 4, "n/a: scenario does not apply", size=12, fill=MUTED))

    # group bars (one per task) and column headers
    gx0 = PAD + LABEL_W
    gy = PAD + TITLE_H + LEGEND_H
    for k, task in enumerate(matrix.tasks):
        idx = [i for i, c in enumerate(matrix.columns) if c[0] == task]
        x1 = gx0 + idx[0] * (CELL_W + GAP)
        x2 = gx0 + (idx[-1] + 1) * (CELL_W + GAP) - GAP
        colour = GROUP_COLOURS[k % len(GROUP_COLOURS)]
        out.append(f'<rect x="{x1}" y="{gy + 4}" width="{x2 - x1}" height="4" rx="2" fill="{colour}"/>')
        out.append(_text(x1, gy + 24, matrix.task_title(task).upper(), size=11, fill=MUTED, weight="600"))
    hy = gy + GROUP_H
    for j, column in enumerate(matrix.columns):
        cx = gx0 + j * (CELL_W + GAP) + CELL_W / 2
        out.append(_text(cx, hy + 16, matrix.column_title(column), size=13, weight="700", anchor="middle"))
        out.append(_text(cx, hy + 33, "test acc % ↑", size=11, fill=MUTED, anchor="middle"))
    cost_x = gx0 + n_cols * (CELL_W + GAP) + 8
    out.append(_text(cost_x, hy + 16, "cost per cell", size=13, weight="700"))
    out.append(_text(cost_x, hy + 33, "epochs trained, models", size=11, fill=MUTED))

    def row(method, y, baseline):
        name = label(METHOD_LABELS, method)
        if baseline:
            out.append(_text(PAD, y + 28, "Baseline", size=15, weight="700", fill=MUTED))
            out.append(_text(PAD, y + 47, name, size=12, fill=MUTED))
        else:
            out.append(_text(PAD, y + CELL_H / 2 + 5, name, size=15, weight="700"))
        for j, (task, scenario) in enumerate(matrix.columns):
            c = matrix.cell(method, task, scenario)
            x = gx0 + j * (CELL_W + GAP)
            dash = f' stroke="{MUTED}" stroke-dasharray="4 3"' if baseline else ""
            if c.mean is None:
                out.append(f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="6" '
                           f'fill="{NA_FILL}" stroke="{BORDER}"/>')
                out.append(_text(x + CELL_W / 2, y + CELL_H / 2 + 5, "n/a", size=13, fill=MUTED, anchor="middle"))
                continue
            ink = _ink(c.rank)
            out.append(f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="6" '
                       f'fill="{_rank_colour(c.rank)}"{dash}><title>{_esc(name)}: '
                       f'{c.mean:.2f} ± {c.std:.2f} % over {c.n} seeds, rank {c.rank}</title></rect>')
            out.append(_text(x + CELL_W / 2, y + 30, f"{c.mean:.2f}", size=19, weight="700", fill=ink,
                             anchor="middle", family=MONO))
            out.append(_text(x + CELL_W / 2, y + 50, f"#{c.rank} · ±{c.std:.2f}", size=11, fill=ink,
                             anchor="middle"))
            if c.worse_than_baseline:
                out.append(f'<path d="M{x + 7} {y + 7} L{x + 21} {y + 7} L{x + 14} {y + 19} Z" fill="{WORSE}">'
                           f'<title>worse than {label(METHOD_LABELS, BASELINE)}</title></path>')
        out.append(_text(cost_x, y + CELL_H / 2 + 5, cost_text(matrix.cost[method]), size=12, fill=MUTED))

    y = top
    for method in body_rows:
        row(method, y, baseline=False)
        y += CELL_H + GAP
    if has_baseline:
        y += BASELINE_GAP
        row(BASELINE, y, baseline=True)
        y += CELL_H + GAP

    out.append(_text(PAD, y + 26, "Cells: test accuracy (%) on the held-out split, mean ± std over seeds, "
                                  "#rank within the column. Hover a cell for details.", size=11, fill=MUTED))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_markdown(matrix: Matrix, title: str = "Support matrix") -> str:
    header = ["Method"] + [f"{matrix.task_title(t)} · {matrix.column_title((t, s))}" for t, s in matrix.columns] \
        + ["Cost per cell"]
    lines = [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for method in matrix.rows:
        name = label(METHOD_LABELS, method) + (" *(baseline)*" if method == BASELINE else "")
        cells = []
        for task, scenario in matrix.columns:
            c = matrix.cell(method, task, scenario)
            if c.mean is None:
                cells.append("n/a")
                continue
            text = f"{c.mean:.2f} ± {c.std:.2f} (#{c.rank})"
            cells.append(("**" + text + "**" if c.rank == 1 else text) + (" ▼" if c.worse_than_baseline else ""))
        lines.append("| " + " | ".join([name] + cells + [cost_text(matrix.cost[method])]) + " |")
    lines.append("")
    lines.append("Test accuracy (%) on the held-out split, mean ± std over seeds; #rank within the column; "
                 "**bold** = best; ▼ = worse than Fixed LR; n/a = the scenario does not apply to the method.")
    return "\n".join(lines) + "\n"


def render_html(svg: str, title: str) -> str:
    """A standalone page around the SVG, for sharing or opening in a browser."""
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{_esc(title)}</title>'
            f'<style>body{{margin:0;background:{BG};display:flex;justify-content:center;padding:24px}}'
            f'svg{{max-width:100%;height:auto}}</style></head><body>{svg}</body></html>\n')
