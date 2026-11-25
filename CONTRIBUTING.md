# Contributing to GraftQ

Thank you for your interest in GraftQ!

This repository is maintained primarily as the official inference code and checkpoint release accompanying the GraftQ paper. For bug reports, questions, or reproduction feedback, please open a GitHub Issue.

## Dual-Licensing Notice for Pull Requests

Because GraftQ is dual-licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** and a **Commercial License** (see [LICENSE](./LICENSE) and [LICENSE-COMMERCIAL.md](./LICENSE-COMMERCIAL.md)), the maintainers must retain the right to distribute the combined codebase under both licenses.

By submitting a pull request to this repository, you confirm that:
1. You have the legal right to submit the contribution.
2. You grant the copyright holders of GraftQ a perpetual, worldwide, non-exclusive, royalty-free, irrevocable copyright and patent license to use, modify, sublicense, and distribute your contribution under both the AGPLv3 and the GraftQ Commercial License.

## Development Checks

Before opening a pull request, please verify that all unit tests pass and no machine-specific paths are introduced:

```bash
PYTHONPATH=. python tests/test_inference.py
```
