"""
Baselines derived from run config:
- all_independent: the same agents, roles, personas and visible fields, with every sharing rule, channel and
  structural constraint removed, so nothing is delivered (and the message slot disappears on its own). Scored on the
  union of the agents' results.
- all_independent_temp_sweep: all_independent, with each agent sampling at its own temperature
- one_for_all: the same config, with one model hosting every agent in one shared conversation
- one_agent: one neutral agent for the condition's own number of steps
- one_agent_scaled_steps: one neutral agent for N times the condition's steps (N = number of agents)
- one_vs_multi_token / one_vs_multi_completion_token: one neutral agent (no persona, every observation field) with
  the condition's budget in total or completion tokens, read per seed and replicate from the condition's logged runs.
"""

from pathlib import Path

from framework.comm.compiler import ProtocolConfig, Role, compile_protocol
from framework.engine.budget import finished
from framework.engine.config import VERSIONS, BudgetConfig, BudgetMatch, RunConfig
from framework.tasks.registry import message_types

BASELINES = VERSIONS[1:]
BUDGET_UNITS = {"one_vs_multi_token": "total_tokens", "one_vs_multi_completion_token": "completion_tokens"}

# the temperatures the agents cycle through, unless the condition's config sets sampling.agent_temperatures or
# sampling.sweep_temperatures. Their mean is 0.85, below the base 1.0: a grid that tests variety sets its own
# (FINDINGS.md, F21)
TEMP_SWEEP = (0.4, 0.7, 1.0, 1.3)

# one_vs_multi: the task's horizon becomes this many times (number of agents x the condition's horizon). It is only a
# safety cap: the matched budget is meant to end the run first, which `ended_by: budget:<unit>` confirms
SAFETY_FACTOR = 10

NEUTRAL_AGENT = ProtocolConfig(roles={"agent": Role(count=1)}, structure={"generator": "empty"})


def derive(cfg: RunConfig, baseline: str, reference: str | None = None) -> RunConfig:
    """The run config of `baseline` for the condition `cfg`. `reference` is the folder of the condition's runs (from the
    repo root, or absolute), needed by the one_vs_multi baselines; by default runs/<condition name>/base"""
    if baseline not in BASELINES:
        raise ValueError(f"unknown baseline {baseline!r}; known: {list(BASELINES)}")
    if cfg.run.baseline is not None:
        raise ValueError(f"{cfg.run.name} is already a baseline ({cfg.run.baseline}); derive from the condition")
    protocol = cfg.resolved_protocol()
    data = cfg.model_dump(mode="json", by_alias=True)
    data["run"]["baseline"] = baseline

    if baseline in ("all_independent", "all_independent_temp_sweep"):
        # constraints are about the links, which are gone (e.g. `connected` would reject the structure)
        independent = protocol.model_copy(update={"rules": (), "channels": {}, "constraints": ()})
        data["protocol"] = independent.model_dump(mode="json", by_alias=True)
        if baseline == "all_independent_temp_sweep":
            sampling = cfg.sampling
            data["sampling"]["agent_temperatures"] = list(sampling.agent_temperatures or sampling.sweep_temperatures
                                                          or TEMP_SWEEP)
    elif baseline == "one_for_all":
        data["engine"]["persona_host"] = True
    elif baseline in ("one_agent", "one_agent_scaled_steps"):
        agents = len(compile_protocol(protocol, message_types(cfg.task.name))[0].agents)
        data["protocol"] = NEUTRAL_AGENT.model_dump(mode="json", by_alias=True)
        data["personas"] = {}
        if baseline == "one_agent_scaled_steps":
            data["task"] = cfg.task.scale_horizon(agents).model_dump(mode="json")
        # only the task's own step limit (and the wall clock, a safety cap) bounds the single agent
        data["budget"] = BudgetConfig(wall_clock_s=cfg.budget.wall_clock_s).model_dump(mode="json")
    else:
        agents = len(compile_protocol(protocol, message_types(cfg.task.name))[0].agents)
        match = BudgetMatch(reference=reference or f"runs/{cfg.run.name}/base", unit=BUDGET_UNITS[baseline])
        data["protocol"] = NEUTRAL_AGENT.model_dump(mode="json", by_alias=True)
        data["personas"] = {}
        data["task"] = cfg.task.scale_horizon(SAFETY_FACTOR * agents).model_dump(mode="json")
        # only the matched budget (and the wall clock, a safety cap) bounds the single agent
        data["budget"] = BudgetConfig(match=match, wall_clock_s=cfg.budget.wall_clock_s,
                                      show_matched_budget=cfg.budget.show_matched_budget).model_dump(mode="json")
    return RunConfig.model_validate(data)


def version_runs(cfg: RunConfig, folder: Path) -> list[tuple[str, RunConfig, Path]]:
    """(version, config, output folder) of every version in `cfg.run.versions`, for the seed and replicate of `cfg`,
    with "base" first: the one_vs_multi baselines read their budget from its logs. `folder` is the run folder. A
    one_vs_multi baseline without "base" uses the base run already in the folder, and is refused if there is none that
    ended on its own"""
    run_dir = f"seed{cfg.run.seed}_rep{cfg.run.replicate}"
    chosen = [v for v in VERSIONS if v in cfg.run.versions]
    if "base" not in chosen and any(v in BUDGET_UNITS for v in chosen):
        base = folder / "base" / run_dir
        if not finished(base):
            raise ValueError(f"{', '.join(v for v in chosen if v in BUDGET_UNITS)} read their budget from the base "
                             f"version, but {base} has no run that ended on its own: add base to run.versions")
    return [(v, cfg if v == "base" else derive(cfg, v, reference=str(folder / "base")), folder / v / run_dir)
            for v in chosen]
