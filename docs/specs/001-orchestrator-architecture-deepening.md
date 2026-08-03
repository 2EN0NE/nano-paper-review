# Spec: Orchestrator Architecture Deepening

## Problem Statement

The orchestrator module (`orchestrator.py`, 990 lines) has grown into a shallow module — the interface of `run_pipeline` is nearly as complex as its implementation. Seven responsibilities are packed into one 250-line function: config resolution, Subject discovery, task fingerprinting, phase dispatch, environment setup, progress coordination, and report generation. The retry loop is duplicated identically across the pooled and sequential execution paths. The `.md` Agent step executor mixes template resolution, prompt construction, subprocess execution, and output parsing with no seam — making pipeline flow tests impossible without a real `pi` binary. The `pipeline.yaml` schema hardcodes three phases (`pre`/`review`/`post`) with different configuration shapes (`PhaseConfig` vs `ReviewPhaseConfig`), preventing users from declaring custom phase pipelines.

## Solution

Deepen the orchestrator by:

1. **Replacing the three-name schema with a declarative `phases:` list**, where each phase declares its execution `mode` (`batch` or `per_subject`) and carries only the configuration relevant to that mode.
2. **Splitting `run_pipeline`** — extracting config resolution (`PipelineConfig.from_path()`), Subject discovery (`SubjectDiscovery` module), and phase execution into private functions, leaving a thin orchestration loop.
3. **Introducing a `StepExecutor` protocol** — a seam between the orchestrator's phase logic and the actual step execution, enabling in-memory pipeline flow tests without `pi`.
4. **Extracting `_retry_step`** as a shared utility, eliminating the 40-line duplicate retry loop between `_process_single_subject` and the sequential path of `_run_phase_steps`.
5. **Separating `PromptBuilder` and `AgentRunner`** — splitting `_run_md_step`'s 100-line monolith so prompt construction can be tested without I/O.

## User Stories

1. As a pipeline author, I want to declare custom phases beyond pre/review/post (e.g. `validate`, `notify`, `archive-deep`), so that I can model multi-stage review workflows.
2. As a pipeline author, I want to declare each phase's execution mode (`batch` or `per_subject`) explicitly in YAML, so that I don't need to memorize which phase names imply which mode.
3. As a pipeline author, I want batch-mode phases to only expose batch-relevant config (retry, manifest_step), and per-subject phases to only expose per-subject config (pool, subject_order, subject_source), so that YAML validation catches misconfigurations early.
4. As a developer writing tests, I want to test the pipeline's retry/abort/skip/phase-ordering logic without a real `pi` binary, so that pipeline flow tests run fast and deterministically.
5. As a developer writing tests, I want to test prompt template resolution and agent prefix generation independently of subprocess execution, so that prompt correctness can be verified in unit tests.
6. As a developer modifying retry policy, I want retry logic defined in exactly one place, so that I don't accidentally fix a bug in the pooled path but miss the sequential path.
7. As an operator running `paper-review review`, I want `--phase <name>` to work with any phase name in the pipeline, not just the hardcoded three, so that I can re-run a specific custom phase.
8. As a developer reading the orchestrator, I want `run_pipeline` to be a short, high-level function that reads like a pipeline definition, so that I can understand the flow without tracing through seven responsibilities.

## Implementation Decisions

### Pipeline YAML schema — `phases:` list

The `pre:` / `review:` / `post:` three-section format is replaced with a single `phases:` list. Each phase declares:

```yaml
phases:
  - name: pre
    mode: batch
    directory: pre-review/
    manifest_step: "00-convert"
    duplicate_policy: skip
    retry:
      max_attempts: 2
      on_failure: skip
  - name: review
    mode: per_subject
    directory: review-pipeline/
    subject_source:
      type: manifest
      path: "{{ output_dir }}/subject-manifest.json"
    duplicate_policy: skip
    retry:
      max_attempts: 1
      on_failure: skip
    subject_order:
      sort_by: name
      direction: asc
    pool:
      workers: 5
      timeout: 600
      ordered: true
  - name: post
    mode: batch
    directory: post-review/
    duplicate_policy: skip
    retry:
      max_attempts: 2
      on_failure: skip
```

- `mode` is required — `batch` or `per_subject`.
- `pool`, `subject_order`, and `subject_source` are valid only in `mode: per_subject`. They are absent (not defaulted) in `mode: batch`.
- `manifest_step` is valid only in `mode: batch`. It is absent in `mode: per_subject`.
- `duplicate_policy` and `retry` apply to both modes.
- Phase execution order is the list order. Top-level `name` and `version` remain unchanged.

### Unified PhaseConfig

A single `PhaseConfig` dataclass replaces the current `PhaseConfig` / `ReviewPhaseConfig` split:

```python
@dataclass
class PhaseConfig:
    name: str
    mode: str  # 'batch' | 'per_subject'
    directory: str = ""
    retry: RetryConfig = field(default_factory=RetryConfig)
    duplicate_policy: str = "skip"

    # batch-only
    manifest_step: str = ""

    # per_subject-only
    subject_source: SubjectSourceConfig | None = None
    subject_order: SubjectOrderConfig | None = None
    pool: PoolConfig | None = None
```

`PipelineConfig` holds `phases: list[PhaseConfig]` instead of `pre` / `review` / `post`. `PipelineConfig.from_path(path: Path)` is a new class method that handles YAML file/directory detection and dict parsing, replacing the inline resolution in `run_pipeline`.

### Phase execution modes

Two private functions replace `_run_phase_steps` (which had `if phase_name == "review"` branches throughout):

- `_execute_batch(phase: PhaseConfig, steps: list[StepFile], output_dir: Path, base_env: dict, executor: StepExecutor, pp: PipelineProgress | None) → dict[str, list[StepResult]]`
  - Runs all steps once against the sentinel subject `_batch_`.
  - Uses `_retry_step` for each step execution.
  - Reports progress via `pp.post_step_done()` / `pp.pre_step_done()`.

- `_execute_per_subject(phase: PhaseConfig, steps: list[StepFile], subjects: list[str], output_dir: Path, base_env: dict, executor: StepExecutor, pool_progress: PoolProgress | None, pp: PipelineProgress | None) → dict[str, list[StepResult]]`
  - When `pool.workers > 1` and `len(subjects) > 1`: uses `ThreadPoolExecutor` pool mode (existing `_run_subjects_pooled` logic, extracted).
  - Otherwise: sequential per-subject loop.
  - Each subject's steps use `_retry_step`.

### StepExecutor protocol

A `Protocol` class defines the seam:

```python
class StepExecutor(Protocol):
    def execute(self, step: StepFile, step_dir: Path, env: dict,
                prior_results: list[StepResult], subject_name: str) -> StepResult: ...
```

Three implementations:

- **`PyStepRunner`** — `runpy.run_path()` for `.py` steps. Does not accept `prior_results` or `subject_name` (they're unused internally; accepted only to satisfy the protocol signature).
- **`MdStepExecutor`** — Composes `PromptBuilder` + `AgentRunner`. `PromptBuilder.build()` resolves template variables and constructs the agent prefix, returning a `str`. `AgentRunner.run()` writes the temp file, invokes `subprocess.run(["pi", ...])`, and parses output.
- **`InMemoryExecutor`** (test-only) — Returns pre-configured `StepResult` values.

A thin dispatch function `_execute_step(step, step_dir, env, prior_results, subject_name, py_runner, md_executor) → StepResult` selects the executor based on `step.step_type`.

### _retry_step shared utility

```python
def _retry_step(step: StepFile, step_dir: Path, env: dict,
                prior_results: list[StepResult], subject_name: str,
                retry_cfg: RetryConfig, executor: StepExecutor) -> StepResult:
```

Implements the retry loop: for attempt in range(max_attempts), call `executor.execute()`, check status, handle exceptions, abort if `on_failure == "abort"`. Both `_execute_batch` and `_execute_per_subject` call this.

### SubjectDiscovery module

A new module `subject_discovery.py` with a single public function:

```python
def discover_subjects(config: PipelineConfig, input_path: Path,
                      output_dir: Path) -> list[str]:
```

Internally: loads manifest if `subject_source.type == "manifest"` for the first per_subject phase, falls back to CLI PDF scanning, applies `duplicate_policy`, applies `subject_order` from the first per_subject phase.

This replaces the ~80 lines of subject discovery logic currently inline in `run_pipeline`.

### PipelineProgress lifecycle

`PipelineProgress` construction, `start()`, and `finish()` remain in `run_pipeline`. The progress instance is passed to each `_execute_batch` / `_execute_per_subject` call. The phase function updates progress via `pre_step_done()`, `post_step_done()`, `review_subject_running()`, `review_step_done()`. (The 6-method interface is not changed in this spec — candidate 5 is deferred.)

### run_pipeline — thin orchestration

The refactored `run_pipeline`:

```python
def run_pipeline(pipeline_yaml: Path, input_path: Path,
                 output_dir: Path | None = None,
                 data_dir: str | None = None,
                 target_phase: str | None = None,
                 target_step: str | None = None,
                 pool_progress: PoolProgress | None = None) -> PipelineResult:
```

1. `config = PipelineConfig.from_path(pipeline_yaml)`
2. Generate task fingerprint + `base_env`
3. `subjects = discover_subjects(config, input_path, task_dir)`
4. Build `PyStepRunner` and `MdStepExecutor` (prod adapters)
5. `pp = PipelineProgress(...)`, `pp.start()`
6. For each `phase` in `config.phases` (filtered by `target_phase`):
   - Discover steps from `phase.directory`
   - If `phase.mode == "batch"`: call `_execute_batch(...)`
   - If `phase.mode == "per_subject"`: call `_execute_per_subject(...)`
   - Error in any step → `overall_success = False`
7. `pp.finish()`
8. `_generate_report(...)` (unchanged)
9. Write `task.json` (unchanged)

### Backward incompatibility

The `pre:` / `review:` / `post:` YAML format is removed entirely — no compatibility layer. All existing YAML files (project `pipeline.yaml`, data dir `pipeline.yaml`, test fixtures) must be updated to the `phases:` list format.

### Affected modules

| Module | Change |
|--------|--------|
| `pipeline_models.py` | `PhaseConfig` unified; `PipelineConfig.phases` list; `PipelineConfig.from_path()`; remove `ReviewPhaseConfig` |
| `orchestrator.py` | `run_pipeline` shrinks to ~80 lines; new `_execute_batch`, `_execute_per_subject`; extract `_retry_step`; remove `_run_phase_steps`; remove `_process_single_subject` (inlined into `_execute_per_subject`); remove `_run_subjects_pooled` (inlined into `_execute_per_subject` pool path); remove inline config resolution and subject discovery |
| `pipeline_steps.py` | `_run_step` replaced by `_execute_step` dispatch; `_run_py_step` → `PyStepRunner`; `_run_md_step` split into `PromptBuilder` + `AgentRunner`; add `StepExecutor` Protocol |
| `subject_discovery.py` | **New module** — `discover_subjects()` |
| `cli.py` | `init` command generates `phases:` YAML; `review` command's `--phase` matches `phases[].name`; `_DEFAULT_PIPELINE_YAML` constant updated |
| `templates/pipeline.yaml` | Updated to `phases:` list format |
| `pipeline/pipeline.yaml` | Updated to `phases:` list format |

## Testing Decisions

### What makes a good test

Tests verify external behavior (what the pipeline produces, not how it loops internally). The `StepExecutor` seam is the primary test boundary: inject `InMemoryExecutor` to simulate any step outcome (success, error, skipped, timeout), then assert on `PipelineResult.step_results`, `task.json`, and `report.md` content.

### Test layers

**Unit tests** (no subprocess, no filesystem beyond tmp_path):

- `PipelineConfig.from_path()` — parse YAML with `phases:` list, verify `PhaseConfig` fields
- `PhaseConfig` validation — reject `pool` on batch mode, reject `manifest_step` on per_subject mode
- `SubjectDiscovery` — manifest JSON parsing, CLI fallback, duplicate_policy (skip/rename/error), subject_order
- `_retry_step` — with `InMemoryExecutor`, verify retries exhaust, abort after failure
- `PromptBuilder` — template variable substitution, agent prefix format
- `AgentRunner` — output.json parsing from mock stdout

**Integration tests** (tmp_path, `InMemoryExecutor`):

- Full `run_pipeline` with 3-phase `phases:` config, verify task.json and report.md
- Pipeline with only batch phases, only per_subject phases, mixed
- `target_phase` filtering
- Pool timeout, pool ordering
- Error propagation (one step fails → overall_success=False)

**E2E tests** (subprocess, real `paper-review` binary):

- Update all YAML fixtures from `pre:/review:/post:` to `phases:` list
- Verify `paper-review review` completes with new YAML format
- Verify intermediates directory structure unchanged
- Verify `paper-review init` generates new-format YAML

### Prior art

Existing test patterns to follow:

- `tests/test_orchestrator.py` — already uses `PipelineConfig.from_dict()` and `monkeypatch` on `_run_step`; updated to use `InMemoryExecutor` injection
- `tests/e2e/test_pipeline_integration.py` — creates `pipeline.yaml` inline; updated to `phases:` format
- `tests/test_template_engine.py` — tests `TemplateContext` and `resolve_variables` in isolation; extended for `PromptBuilder`

## Out of Scope

- Retiring the `PoolProgress` class or its 6-method interface (candidate 5 — deferred)
- Changing the `PipelineProgress` ANSI rendering implementation (candidate 5 — deferred)
- `PipelineConfig.from_dict()` consolidation with a recursive dataclass helper (candidate 4 — deferred; the current manual `from_dict` is updated but not replaced)
- Adding a YAML schema validator (cerberus/jsonschema)
- Adding support for phase-level parallelism (running batch phases concurrently)
- Changing the intermediates directory structure
- Changing the report.md generation format
- Migrating the search subsystem (store.py, retriever.py, etc.)

## Further Notes

- The `PhaseConfig` unified dataclass intentionally has optional fields for both modes. A future PR could add `__post_init__` validation to reject mode-inappropriate fields at parse time. This is left out of scope for now to keep the diff minimal.
- `SubjectDiscovery` uses the `subject_source` from the **first** `per_subject` phase. If multiple per_subject phases declare different `subject_source` configs, only the first is used. This is consistent with current behavior where `subject_source` is a review-phase-only field.
- The `target_phase` and `target_step` CLI parameters work with the new `phases:` list — `--phase` matches `phase.name`, and the same single-phase shortcut logic applies.
