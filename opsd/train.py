"""Offline OPSD: distill the hinted teacher into a student that never sees the hint.

Offline mode replays each stored reply from `build_dataset.py`: the student scores it on the plain harness prompt,
the frozen teacher (the base model) on the same prompt plus the hint, and the loss is their reverse KL on the reply's
tokens. No environment, grader or reward runs in training, and no inference engine is needed.

Every setting this script sends is a flag below. A flag left unset is not sent, so AC2 uses its own default for it
(learning rate, response and context length, ...). Offline mode itself requires one sample per task, no inference
replicas and no in-run eval, so those are always set. `--dry-run` prints the settings and the exact configuration.

Training has no poker eval: after the run, serve a checkpoint and play it through the harness, which never shows the
hint, so that eval measures the student without it.

`--rollout-mode one_step` makes the student write its own reply to each task's prompt (thinking on, up to
`--max-response-len` tokens, default the event's 5,000) and trains it toward the teacher on those tokens, so the
decision tokens get direct signal. It needs hinted tasks whose input is the prompt only (`pokerbench_tasks.py --mode
hinted`) and one sampling replica (2 GPUs), so about 6 GPUs in all; AC2 refuses `--max-total-len` in this mode.

    uv run python opsd/train.py --dataset pokerbots-opsd-train --dry-run
    uv run python opsd/train.py --dataset pokerbots-opsd-train --lr 1e-6 --dry-run   # override an AC2 default
    uv run python opsd/train.py --dataset pokerbench-hinted-4k --rollout-mode one_step --steps 30 --dry-run
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from ac2.sdk import Client
from ac2.sdk.run_config import (
    EvalConfig,
    ExperimentConfig,
    InternalConfig,
    InternalLaunchConfig,
    OPSDConfig,
    RolloutConfig,
    RunConfig,
    StepConfig,
    TasksConfig,
    TrainDatasetConfig,
    TrainerConfig,
)



def _set(**values: Any) -> dict[str, Any]:
    """Only the values that were given: anything None is left to AC2's default."""
    return {key: value for key, value in values.items() if value is not None}


ONE_STEP_RESPONSE_LEN = 5000  # one-step samples one decision: the event's per-decision cap


def rollout(args: argparse.Namespace) -> RolloutConfig:
    if args.rollout_mode == "offline":
        return RolloutConfig(
            n_inference_replicas=0,  # required by offline mode: nothing is sampled
            **_set(max_response_len=args.max_response_len, max_total_len=args.max_total_len),
        )
    if args.max_total_len is not None:
        raise SystemExit("one_step OPSD does not accept --max-total-len")
    return RolloutConfig(  # one sampling engine (2 GPUs) writes the student's replies
        n_inference_replicas=1, max_response_len=args.max_response_len or ONE_STEP_RESPONSE_LEN
    )


def run_config(args: argparse.Namespace, name: str) -> RunConfig:
    launch = _set(tensorboard_enabled=args.tensorboard or None)  # AC2 places the run; teams can't pick a cluster
    mode = args.rollout_mode
    return RunConfig(
        model=args.model,
        name=name,
        description=f"{'Offline' if mode == 'offline' else 'One-step'} OPSD of a prompted hint into {args.model} "
        f"(student without the hint), {args.dataset}.",
        tags={"benchmark": "pokerbots", "method": "opsd", "rollout_mode": mode, "dataset": args.dataset},
        experiment_cfg=ExperimentConfig(
            method=OPSDConfig(rollout_mode=mode),
            step=StepConfig(
                # offline replays one stored reply per task (it must be 1); one-step samples this many per task
                samples_per_problem=1 if mode == "offline" else args.samples,
                **_set(
                    num_train_steps=args.steps,
                    problem_batch_size=args.batch,
                    checkpoint_interval=args.checkpoint_every,
                    keep_last_checkpoints=args.keep_checkpoints,
                ),
            ),
            tasks=TasksConfig(
                datasets=(TrainDatasetConfig(ac2_dataset_name=args.dataset),),
                **_set(apply_chat_template_kwargs={"enable_thinking": True} if args.thinking else None),
            ),
            trainer=TrainerConfig(**_set(learning_rate=args.lr)),
            rollout=rollout(args),
            eval=EvalConfig(mode="off"),  # required here: the poker eval runs through the harness afterwards
        ),
        **_set(internal_cfg=InternalConfig(launch=InternalLaunchConfig(**launch)) if launch else None),
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="vals-hackathon")
    p.add_argument("--dataset", required=True, help="hinted dataset from build_dataset.py")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="base model (arena entries are this model or fine-tunes of it)")
    p.add_argument("--dry-run", action="store_true", help="print the settings and configuration, submit nothing")
    p.add_argument("--rollout-mode", choices=["offline", "one_step"], default="offline",
                   help="offline replays stored replies; one_step samples the student's own (default offline)")
    p.add_argument("--samples", type=int, default=1, help="one_step: student replies per task (default 1)")
    g = p.add_argument_group("set by default (the OPSD recipe); change them with these flags")
    g.add_argument("--steps", type=int, default=20, help="training steps (default 20)")
    g.add_argument("--batch", type=int, default=64, help="decisions per step (default 64)")
    g.add_argument("--checkpoint-every", type=int, default=10, help="steps between checkpoints (default 10)")
    g.add_argument("--no-thinking", dest="thinking", action="store_false",
                   help="don't render with Qwen thinking on (the event has it on)")
    g.add_argument("--no-tensorboard", dest="tensorboard", action="store_false", help="don't record TensorBoard logs")
    a = p.add_argument_group("left to AC2's defaults unless you set them")
    a.add_argument("--lr", type=float, help="learning rate")
    a.add_argument("--max-response-len", type=int, help="per-reply token limit")
    a.add_argument("--max-total-len", type=int, help="prompt plus reply token limit")
    a.add_argument("--keep-checkpoints", type=int, help="checkpoints to keep")
    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    tag = "" if args.rollout_mode == "offline" else "-1step"
    name = f"pokerbots-opsd{tag}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    config = run_config(args, name)
    if args.dry_run:
        print("This launch sets (everything else is AC2's default):")
        skip = {"project", "dry_run", "help"}
        for action in p._actions:  # noqa: SLF001
            value = getattr(args, action.dest, None)
            if action.dest not in skip and value not in (None, False):
                print(f"  {action.dest} = {value}")
        if args.rollout_mode == "offline":
            print("  samples_per_problem = 1, n_inference_replicas = 0, eval = off (required by offline mode)")
            print("GPUs while running: about 4 for Qwen3.6-35B-A3B (training only; offline OPSD samples nothing)")
        else:
            print(f"  samples_per_problem = {args.samples}, n_inference_replicas = 1, eval = off, max_response_len = "
                  f"{config.experiment_cfg.rollout.max_response_len}")
            print("GPUs while running: about 6 for Qwen3.6-35B-A3B (4 training + 2 for the student's sampling engine)")
        print(json.dumps(config.to_json(), indent=1, default=str))
        return
    run = Client(project=args.project).train.run(config)
    print(json.dumps({"train_id": run.train_id, "name": name}))


if __name__ == "__main__":
    main()
