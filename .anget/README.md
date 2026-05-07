# Project Agent Assets

This directory stores optional agent skills and workflow templates selected for this Python/OpenCV desktop background-replacement project.

Nothing here is executed automatically.

## Skills

Downloaded skills are under `skills/`.

| Skill | Source | Why it fits |
| --- | --- | --- |
| `openai-gh-fix-ci` | `openai/skills` | Diagnose and repair CI failures when GitHub Actions is added. |
| `openai-security-best-practices` | `openai/skills` | Review Python app changes for common security issues. |
| `openai-security-threat-model` | `openai/skills` | Think through risks around camera input, local files, model assets, and generated outputs. |
| `trailofbits-modern-python` | `trailofbits/skills` | Modern Python project hygiene: uv, ruff, pytest, pyproject guidance. |
| `trailofbits-property-based-testing` | `trailofbits/skills` | Test image/path utilities and algorithm wrappers with generated inputs. |
| `trailofbits-sharp-edges` | `trailofbits/skills` | Find surprising API behavior and brittle config/path handling. |
| `trailofbits-supply-chain-risk-auditor` | `trailofbits/skills` | Review dependency and model-download risk. |
| `trailofbits-codeql` | `trailofbits/skills` | Static analysis workflow guidance with CodeQL. |
| `trailofbits-semgrep` | `trailofbits/skills` | Rule-based scanning guidance for Python and workflow files. |
| `trailofbits-sarif-parsing` | `trailofbits/skills` | Inspect CodeQL/Semgrep SARIF outputs. |
| `trailofbits-agentic-actions-auditor` | `trailofbits/skills` | Audit GitHub Actions workflows for unsafe AI-agent patterns. |

## Workflows

Downloaded GitHub starter workflows are under:

```text
workflows/github-actions/downloaded/
```

Project-tailored templates are under:

```text
workflows/github-actions/recommended/
```

To enable one, copy the chosen YAML file into `.github/workflows/` and review dependencies first.

## Sources

- OpenAI Agent Skills: https://github.com/openai/skills
- Trail of Bits Skills Marketplace: https://github.com/trailofbits/skills
- GitHub Actions starter workflows: https://github.com/actions/starter-workflows
- GitHub Dependency Review Action: https://github.com/actions/dependency-review-action
- GitHub CodeQL Action: https://github.com/github/codeql-action
- Ruff GitHub Action: https://docs.astral.sh/ruff/integrations/
- uv GitHub Actions integration: https://docs.astral.sh/uv/guides/integration/github/
