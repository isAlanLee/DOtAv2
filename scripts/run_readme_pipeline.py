#!/usr/bin/env python3
"""Run the README DOtA pipeline with per-step logs and fail-fast shutdown."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Iterable, Sequence

import yaml


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the README DOtA pipeline and save one log per step."
    )
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument(
        "--initial-hypes",
        default="opencood/hypes_yaml/point_pillar_intermediate_fusion_lable_free.yaml",
    )
    parser.add_argument(
        "--dota-hypes",
        default="opencood/hypes_yaml/point_pillar_intermediate_fusion_dota.yaml",
    )
    parser.add_argument("--fusion-method", default="intermediate")
    parser.add_argument("--mbe-output-dir", default="/root/autodl-tmp/out_mbe")
    parser.add_argument(
        "--pseudo-label-root",
        default="/root/autodl-tmp/out_pseudo_lables",
        help="Initial inference output root used by inference.py and MBE.py.",
    )
    parser.add_argument("--log-root", default="pipeline_logs")
    parser.add_argument(
        "--initial-detector-dir",
        default=None,
        help="Use an existing initial detector checkpoint and skip preliminary training.",
    )
    parser.add_argument(
        "--final-checkpoint-dir",
        default=None,
        help="Use an existing final checkpoint and skip pseudo-label training.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Run through training but skip the final README test command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write logs and print commands without executing them.",
    )
    parser.add_argument(
        "--shutdown-command",
        default="shutdown -h now",
        help="Linux shutdown command to run after any pipeline failure.",
    )
    parser.add_argument(
        "--no-system-shutdown",
        action="store_true",
        help="Abort the pipeline on failure but do not power off the server.",
    )
    return parser.parse_args()


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = Path(args.repo_root).resolve()
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = (self.repo_root / args.log_root / timestamp).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.summary_log = self.run_dir / "pipeline_summary.log"
        self.current_process: subprocess.Popen[str] | None = None
        self.initial_detector_dir: Path | None = (
            Path(args.initial_detector_dir).expanduser().resolve()
            if args.initial_detector_dir
            else None
        )
        self.final_checkpoint_dir: Path | None = (
            Path(args.final_checkpoint_dir).expanduser().resolve()
            if args.final_checkpoint_dir
            else None
        )

    def log_summary(self, message: str) -> None:
        line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with self.summary_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def shutdown(self, reason: str, code: int = 1) -> None:
        self.log_summary(f"SHUTDOWN: {reason}")
        self.terminate_current_process()
        if self.args.no_system_shutdown or self.args.dry_run:
            self.log_summary("SYSTEM SHUTDOWN SKIPPED by --no-system-shutdown or --dry-run")
        else:
            self.invoke_system_shutdown(reason)
        raise SystemExit(code)

    def terminate_current_process(self) -> None:
        proc = self.current_process
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        except Exception:
            try:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass

    def invoke_system_shutdown(self, reason: str) -> None:
        command = shlex.split(self.args.shutdown_command)
        log_path = self.run_dir / "shutdown.log"
        if not command:
            self.log_summary("SYSTEM SHUTDOWN SKIPPED because --shutdown-command is empty")
            return

        printable = " ".join(shlex.quote(part) for part in command)
        self.log_summary(f"SYSTEM SHUTDOWN START: {printable}")
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"reason: {reason}\n")
            log.write(f"command: {printable}\n")
            log.write(f"started_at: {dt.datetime.now().isoformat(timespec='seconds')}\n")
            log.flush()
            try:
                result = subprocess.run(
                    command,
                    cwd=str(self.repo_root),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                    check=False,
                )
                log.write(f"return_code: {result.returncode}\n")
                if result.returncode != 0:
                    self.log_summary(
                        f"SYSTEM SHUTDOWN COMMAND FAILED with return code {result.returncode}; see {log_path}"
                    )
                else:
                    self.log_summary(f"SYSTEM SHUTDOWN COMMAND ISSUED: {printable}")
            except Exception as exc:
                log.write(f"exception: {exc!r}\n")
                self.log_summary(f"SYSTEM SHUTDOWN COMMAND FAILED: {exc!r}; see {log_path}")
            finally:
                log.write(f"finished_at: {dt.datetime.now().isoformat(timespec='seconds')}\n")

    def step_log_path(self, number: int, name: str) -> Path:
        return self.run_dir / f"{number:02d}_{name}.log"

    def run_command(
        self,
        number: int,
        name: str,
        command: Sequence[str],
        *,
        env_updates: dict[str, str] | None = None,
    ) -> Path:
        log_path = self.step_log_path(number, name)
        printable = " ".join(shlex.quote(str(part)) for part in command)
        env = os.environ.copy()
        if env_updates:
            env.update(env_updates)
        self.log_summary(f"STEP {number:02d} START {name}: {printable}")
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"step: {name}\n")
            log.write(f"cwd: {self.repo_root}\n")
            log.write(f"command: {printable}\n")
            if env_updates:
                for key, value in sorted(env_updates.items()):
                    log.write(f"env.{key}: {value}\n")
            log.write(f"started_at: {dt.datetime.now().isoformat(timespec='seconds')}\n\n")
            log.flush()

            if self.args.dry_run:
                log.write("DRY RUN: command not executed.\n")
                self.log_summary(f"STEP {number:02d} DRY-RUN {name}")
                return log_path

            creationflags = 0
            popen_kwargs = {}
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            try:
                self.current_process = subprocess.Popen(
                    list(command),
                    cwd=str(self.repo_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                    **popen_kwargs,
                )
                assert self.current_process.stdout is not None
                for line in self.current_process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return_code = self.current_process.wait()
            except KeyboardInterrupt:
                self.shutdown(f"interrupted during step {name}", 130)
            finally:
                self.current_process = None

            log.write(f"\nfinished_at: {dt.datetime.now().isoformat(timespec='seconds')}\n")
            log.write(f"return_code: {return_code}\n")
            log.flush()

        if return_code != 0:
            self.shutdown(
                f"step {number:02d} {name} failed with return code {return_code}; see {log_path}",
                return_code,
            )
        self.log_summary(f"STEP {number:02d} DONE {name}: log={log_path}")
        return log_path

    def write_step_log(self, number: int, name: str, lines: Iterable[str]) -> Path:
        log_path = self.step_log_path(number, name)
        self.log_summary(f"STEP {number:02d} START {name}")
        with log_path.open("w", encoding="utf-8") as log:
            for line in lines:
                print(line, flush=True)
                log.write(line + "\n")
        self.log_summary(f"STEP {number:02d} DONE {name}: log={log_path}")
        return log_path

    def require_file(self, path: Path, description: str) -> None:
        if not path.is_file():
            self.shutdown(f"missing {description}: {path}")

    def require_dir(self, path: Path, description: str) -> None:
        if not path.is_dir():
            self.shutdown(f"missing {description}: {path}")

    def require_glob(self, path: Path, pattern: str, description: str) -> list[Path]:
        matches = sorted(path.glob(pattern))
        if not matches:
            self.shutdown(f"missing {description}: {path / pattern}")
        return matches

    def preflight(self) -> None:
        initial_hypes = self.repo_root / self.args.initial_hypes
        dota_hypes = self.repo_root / self.args.dota_hypes
        required_files = [
            initial_hypes,
            dota_hypes,
            self.repo_root / "opencood/tools/train.py",
            self.repo_root / "opencood/tools/inference.py",
            self.repo_root / "opencood/tools/MBE.py",
            self.repo_root / "opencood/tools/box_score_for_mbe.py",
        ]
        lines = [
            f"repo_root: {self.repo_root}",
            f"run_dir: {self.run_dir}",
            f"python: {self.args.python}",
            f"cuda_devices: {self.args.cuda_devices}",
            f"initial_hypes: {initial_hypes}",
            f"dota_hypes: {dota_hypes}",
            f"mbe_output_dir: {self.args.mbe_output_dir}",
            f"pseudo_label_root: {self.args.pseudo_label_root}",
        ]
        for path in required_files:
            if not path.is_file():
                lines.append(f"MISSING: {path}")
            else:
                lines.append(f"OK: {path}")
        self.write_step_log(0, "preflight", lines)
        for path in required_files:
            self.require_file(path, f"required pipeline file {path.name}")

    def train_command(self, hypes_path: Path) -> list[str]:
        return [
            self.args.python,
            "opencood/tools/train.py",
            "--hypes_yaml",
            str(hypes_path),
        ]

    def checkpoint_from_train_log(self, log_path: Path) -> Path:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"Training Finished, checkpoints saved to\s+(.+)", text)
        if matches:
            return Path(matches[-1].strip()).expanduser().resolve()
        logs_dir = self.repo_root / "opencood" / "logs"
        candidates = []
        if logs_dir.is_dir():
            for path in logs_dir.iterdir():
                if path.is_dir() and list(path.glob("net_epoch*.pth")):
                    candidates.append(path)
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime).resolve()
        self.shutdown(f"could not parse checkpoint directory from {log_path}")
        raise AssertionError("unreachable")

    def verify_checkpoint_dir(self, path: Path, label: str) -> None:
        self.require_dir(path, label)
        if not list(path.glob("net_epoch*.pth")) and not (path / "latest.pth").is_file():
            self.shutdown(f"{label} has no checkpoint file: {path}")

    def make_dota_hypes(self, number: int) -> Path:
        src = self.repo_root / self.args.dota_hypes
        dst_dir = self.run_dir / "generated_hypes"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "point_pillar_intermediate_fusion_dota.pipeline.yaml"
        with src.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        score_dir = str(Path(self.args.mbe_output_dir) / "score")
        data["iterative_training"] = True
        data["pseudo_lable_path"] = score_dir
        with dst.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        self.write_step_log(
            number,
            "prepare_dota_hypes",
            [
                f"source: {src}",
                f"generated: {dst}",
                "set iterative_training: True",
                f"set pseudo_lable_path: {score_dir}",
            ],
        )
        return dst

    def run(self) -> None:
        if self.args.no_system_shutdown or self.args.dry_run:
            shutdown_mode = "abort pipeline only; system poweroff disabled"
        else:
            shutdown_mode = f"abort pipeline and run system command: {self.args.shutdown_command}"
        self.log_summary(f"Pipeline created. Shutdown mode: {shutdown_mode}")
        self.preflight()

        env_updates = {"CUDA_VISIBLE_DEVICES": self.args.cuda_devices}
        initial_hypes = self.repo_root / self.args.initial_hypes

        if self.initial_detector_dir is None:
            initial_train_log = self.run_command(
                1,
                "train_initial_detector",
                self.train_command(initial_hypes),
                env_updates=env_updates,
            )
            if self.args.dry_run:
                self.initial_detector_dir = Path("$INITIAL_DETECTOR_CHECKPOINT_FOLDER")
            else:
                self.initial_detector_dir = self.checkpoint_from_train_log(initial_train_log)
        else:
            self.write_step_log(
                1,
                "train_initial_detector_skipped",
                [f"using existing initial_detector_dir: {self.initial_detector_dir}"],
            )
        if not self.args.dry_run:
            self.verify_checkpoint_dir(self.initial_detector_dir, "initial detector checkpoint directory")

        self.run_command(
            2,
            "generate_initial_pseudo_labels",
            [
                self.args.python,
                "opencood/tools/inference.py",
                "--model_dir",
                str(self.initial_detector_dir),
                "--fusion_method",
                self.args.fusion_method,
                "--pseudo_lable_save",
                "0",
            ],
        )
        if not self.args.dry_run:
            pseudo_root = Path(self.args.pseudo_label_root)
            pre_box_dir = pseudo_root / "pre_box_test_full"
            pre_score_dir = pseudo_root / "pre_score_test_full"
            self.require_glob(pre_box_dir, "pre_*.npy", "initial pseudo-label boxes")
            self.require_glob(pre_score_dir, "score_*.npy", "initial pseudo-label scores")

        self.run_command(3, "mbe_filter", [self.args.python, "opencood/tools/MBE.py"])
        mbe_dir = Path(self.args.mbe_output_dir)
        if not self.args.dry_run:
            self.require_glob(mbe_dir, "out_pseduo_labels_v1_*.npy", "MBE accepted pseudo-labels")
            self.require_glob(mbe_dir, "out_pseduo_labels_noise_v1_*.npy", "MBE rejected pseudo-labels")
            self.require_glob(
                mbe_dir / "multi_agent_point_remove_ground",
                "multi_agent_point*.npy",
                "MBE point caches",
            )
            self.require_glob(
                mbe_dir / "multi_agent_point_pose",
                "multi_agent_point_pose*.npy",
                "MBE pose caches",
            )

        self.run_command(4, "score_mbe_boxes", [self.args.python, "opencood/tools/box_score_for_mbe.py"])
        if not self.args.dry_run:
            score_dir = mbe_dir / "score"
            self.require_glob(score_dir, "out_pseduo_labels_with_score_v4_*.npy", "scored accepted pseudo-labels")
            self.require_glob(score_dir, "out_pseduo_labels_noise_with_score_v4_*.npy", "scored rejected pseudo-labels")

        generated_dota_hypes = self.make_dota_hypes(5)

        if self.final_checkpoint_dir is None:
            final_train_log = self.run_command(
                6,
                "train_with_pseudo_labels",
                self.train_command(generated_dota_hypes),
                env_updates=env_updates,
            )
            if self.args.dry_run:
                self.final_checkpoint_dir = Path("$CHECKPOINT_FOLDER")
            else:
                self.final_checkpoint_dir = self.checkpoint_from_train_log(final_train_log)
        else:
            self.write_step_log(
                6,
                "train_with_pseudo_labels_skipped",
                [f"using existing final_checkpoint_dir: {self.final_checkpoint_dir}"],
            )
        if not self.args.dry_run:
            self.verify_checkpoint_dir(self.final_checkpoint_dir, "final checkpoint directory")

        if self.args.skip_test:
            self.write_step_log(7, "test_final_model_skipped", ["skip_test: True"])
        else:
            self.run_command(
                7,
                "test_final_model",
                [
                    self.args.python,
                    "opencood/tools/inference.py",
                    "--model_dir",
                    str(self.final_checkpoint_dir),
                    "--fusion_method",
                    self.args.fusion_method,
                ],
            )

        self.log_summary(f"PIPELINE COMPLETE. Logs are in {self.run_dir}")


def main() -> None:
    pipeline = Pipeline(parse_args())
    pipeline.run()


if __name__ == "__main__":
    main()
