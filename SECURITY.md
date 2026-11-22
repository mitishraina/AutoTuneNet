# Security policy

## Supported versions

Security fixes are released for the latest minor version on PyPI.

| Version | Supported |
|---|---|
| 1.1.x | ✅ |
| < 1.1 | ❌ |

## Reporting a vulnerability

**Do not open a public issue for a vulnerability.** Report it privately through GitHub: [Security → Report a vulnerability](https://github.com/mitishraina/AutoTuneNet/security/advisories/new).

Include:

- the affected version and how you installed it;
- what an attacker can do, and the steps or a minimal script to reproduce it;
- any suggested fix.

You can expect an acknowledgement within 7 days and an assessment within 14 days. Once a fix is released, the advisory is published with credit to you unless you prefer otherwise.

## Scope and known trust boundaries

These behaviours are by design. Report them only if they can be triggered in a way the documentation does not describe:

- **`--plugins`** imports and runs the Python file or module you pass. Only pass plugins you trust.
- **Experiment files** are YAML, parsed with `yaml.safe_load`.
- **PBT checkpoints** are loaded with `torch.load(weights_only=False)`. They are written and read by the same run in a temporary directory that is deleted afterwards. Never point AutoTuneNet at checkpoints from an untrusted source.
- **Kaggle credentials.** The `kaggle` runtime only checks that a token file exists under `~/.kaggle/`; it never reads, prints, uploads or stores the token. Notebooks run under your account and contain no credentials. Runtime state, which includes your Kaggle username, is written to a git-ignored `.runtime/` folder.
- **Notebook installs.** Kaggle notebooks install AutoTuneNet from this repository at a specific commit, or from PyPI at a pinned version.
