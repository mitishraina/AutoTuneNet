# Accessibility

AutoTuneNet is a library, a command-line tool and a documentation site. We want all three to be usable by people who use screen readers, keyboard-only navigation, high-contrast or magnified displays, or who cannot rely on colour.

## Our commitment

- **Documentation site.** Every page is plain, semantic HTML from MkDocs Material: real headings, keyboard navigation and search, a visible focus state, light and dark themes that follow your system setting, and text that reflows when zoomed.
- **Figures.** Every image and diagram has alternative text describing what it shows, and the numbers behind every figure are also available as text or tables.
- **Results without colour.** The support matrix colours each cell by its difference from the baseline, but every cell also prints the accuracy, the difference in percentage points and the rank as text. The same table is published as Markdown (`support_matrix.md`).
- **Command line.** Output is plain text. Help is available for every command (`autotunenet <command> --help`), errors say what went wrong and what to do, and no command needs a mouse or a graphical interface.
- **Language.** Documentation uses short sentences and defines terms on first use.

## Supported environments

| Component | Supported |
|---|---|
| Python | 3.10, 3.11 (tested in CI); 3.12+ expected to work |
| Operating systems | Linux and macOS (tested); Windows expected to work, not tested in CI |
| PyTorch | 2.x, on CPU, CUDA and Apple MPS |
| Documentation site | Current versions of Chrome, Firefox, Safari and Edge, with or without JavaScript (search needs JavaScript) |
| Assistive technology | Keyboard navigation and screen readers through standard HTML; no assistive-technology testing has been done yet (see below) |

## Known limitations

- **Diagrams.** Mermaid diagrams on the documentation site render as SVG drawings. Their alternative text is the surrounding description, and each diagram's steps are also written out as a numbered list on the same page.
- **Progress bars.** Dataset downloads use `tqdm` progress bars, which some screen readers announce repeatedly. Downloads also print a plain line when they start and finish.
- **Support matrix hover details.** Extra details in the SVG matrix appear on hover. The same information is in `support_matrix.md`.
- **No formal audit yet.** The site has not been checked with a screen reader or an automated accessibility audit. Reports of problems are especially welcome.
- **Mathematical notation.** The problem definition page uses MathJax, which most screen readers can read; each formula is also described in words.

## Reporting a barrier

If something in the library, the command line or the documentation is hard or impossible for you to use, please [open an issue](https://github.com/mitishraina/AutoTuneNet/issues/new?template=accessibility.yml) using the accessibility template. Tell us what you were trying to do, what you use (screen reader, browser, operating system) and what happened. We treat accessibility barriers as bugs and aim to respond within 7 days.
