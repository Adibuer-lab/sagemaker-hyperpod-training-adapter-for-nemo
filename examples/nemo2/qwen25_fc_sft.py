#!/usr/bin/env python3
# NeMo 2.0 Qwen2.5 function-calling SFT using NeMo-Run (direct execution).

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import nemo_run as run
import nemo.collections.llm as llm


def _get_rank() -> int:
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def _normalize_optional_str(value) -> str | None:
    if value is None:
        return None
    val = str(value).strip()
    if not val:
        return None
    if val.lower() in {"none", "null", "off"}:
        return None
    return val


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    val = str(value).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def _wait_for_path(path: Path, timeout_s: int = 7200, poll_s: int = 10) -> None:
    start = time.time()
    while not path.exists():
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(poll_s)


def _get_world_config() -> tuple[int, int]:
    # Derive from torchrun environment where possible.
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            local_world_size = str(len([c for c in cuda_visible.split(",") if c.strip()]))
    gpus_per_node = int(local_world_size or "1")

    world_size = int(os.environ.get("WORLD_SIZE", str(gpus_per_node)))
    nodes = max(1, world_size // max(1, gpus_per_node))
    return nodes, gpus_per_node


def _validate_dataset_root(dataset_root: Path) -> None:
    required = ["training.jsonl", "validation.jsonl", "test.jsonl"]
    missing = [name for name in required if not (dataset_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files under {dataset_root}: {missing}")


def _resolve_finetune_data_module():
    # NeMo 2.0 moved some symbols; keep both import paths for robustness.
    try:
        from nemo.collections.llm import FineTuningDataModule
    except Exception:
        from nemo.collections.llm.data import FineTuningDataModule
    return FineTuningDataModule


def _apply_yarn(model_cfg, seq_length: int) -> bool:
    # Apply YaRN rope scaling if the config supports it.
    yarn_cfg = {
        # Support both NeMo-style and HF/vLLM-style keys.
        "rope_type": "yarn",
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
    }
    applied = False

    # Sync max position / seq length if present.
    for attr in ("seq_length", "max_position_embeddings"):
        if hasattr(model_cfg, attr):
            try:
                setattr(model_cfg, attr, seq_length)
                applied = True
            except Exception:
                pass

    # Common Qwen2.5 rope defaults.
    if hasattr(model_cfg, "rope_theta"):
        try:
            setattr(model_cfg, "rope_theta", 1000000.0)
            applied = True
        except Exception:
            pass
    if hasattr(model_cfg, "rms_norm_eps"):
        try:
            setattr(model_cfg, "rms_norm_eps", 1.0e-6)
            applied = True
        except Exception:
            pass

    if hasattr(model_cfg, "rope_scaling"):
        try:
            rope_scaling = getattr(model_cfg, "rope_scaling")
            if rope_scaling is None:
                setattr(model_cfg, "rope_scaling", dict(yarn_cfg))
            elif isinstance(rope_scaling, dict):
                rope_scaling.update(yarn_cfg)
            else:
                for key, value in yarn_cfg.items():
                    if hasattr(rope_scaling, key):
                        setattr(rope_scaling, key, value)
                    else:
                        try:
                            rope_scaling[key] = value
                        except Exception:
                            pass
            applied = True
        except Exception:
            pass

    return applied


def _apply_recompute_config(
    finetune,
    granularity: str | None,
    method: str | None,
    num_layers: int | None,
) -> bool:
    applied = False
    gran = _normalize_optional_str(granularity)
    meth = _normalize_optional_str(method)
    num = num_layers if isinstance(num_layers, int) and num_layers > 0 else None

    if gran is None and meth is None and num is None:
        return False

    model = getattr(finetune, "model", None)
    for obj in (model, getattr(model, "config", None)):
        if obj is None:
            continue
        if gran is not None and hasattr(obj, "recompute_granularity"):
            try:
                setattr(obj, "recompute_granularity", gran)
                applied = True
            except Exception:
                pass
        if hasattr(obj, "recompute_method"):
            try:
                if meth is not None:
                    setattr(obj, "recompute_method", meth)
                    applied = True
                elif gran is not None and not hasattr(obj, "recompute_granularity"):
                    setattr(obj, "recompute_method", gran)
                    applied = True
            except Exception:
                pass
        if num is not None and hasattr(obj, "recompute_num_layers"):
            try:
                setattr(obj, "recompute_num_layers", num)
                applied = True
            except Exception:
                pass
        if gran is not None and hasattr(obj, "activations_checkpoint_granularity"):
            try:
                setattr(obj, "activations_checkpoint_granularity", gran)
                applied = True
            except Exception:
                pass
        if hasattr(obj, "activations_checkpoint_method"):
            try:
                if meth is not None:
                    setattr(obj, "activations_checkpoint_method", meth)
                    applied = True
                elif gran is not None and not hasattr(obj, "activations_checkpoint_granularity"):
                    setattr(obj, "activations_checkpoint_method", gran)
                    applied = True
            except Exception:
                pass
        if num is not None and hasattr(obj, "activations_checkpoint_num_layers"):
            try:
                setattr(obj, "activations_checkpoint_num_layers", num)
                applied = True
            except Exception:
                pass
    return applied


def _set_attr_if_present(obj, attr: str, value) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        if attr in obj:
            obj[attr] = value
            return True
        return False
    if hasattr(obj, attr):
        try:
            setattr(obj, attr, value)
            return True
        except Exception:
            return False
    return False


def _apply_gradient_accumulation_fusion(finetune, enabled: bool) -> bool:
    applied = False
    model = getattr(finetune, "model", None)
    for obj in (
        finetune,
        model,
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "model", None),
        getattr(getattr(model, "config", None), "megatron", None),
    ):
        if _set_attr_if_present(obj, "gradient_accumulation_fusion", enabled):
            applied = True
    return applied


def _get_qwen25_config():
    for name in ("Qwen25Config14B", "Qwen25Config"):
        cfg = getattr(llm, name, None)
        if cfg is not None:
            return cfg()
    raise AttributeError("Unable to locate Qwen2.5 config in nemo.collections.llm")


def _run_single_process_import(hf_model_id: str, output_path: Path) -> None:
    """Run HF -> NeMo import in a clean, single-process env to avoid DDP init."""
    code = (
        "from pathlib import Path\n"
        "import nemo.collections.llm as llm\n"
        "def _get_qwen25_config():\n"
        "    for name in ('Qwen25Config14B','Qwen25Config'):\n"
        "        cfg = getattr(llm, name, None)\n"
        "        if cfg is not None:\n"
        "            return cfg()\n"
        "    raise AttributeError('Unable to locate Qwen2.5 config in nemo.collections.llm')\n"
        "output_path = Path(" + repr(str(output_path)) + ")\n"
        "llm.import_ckpt(\n"
        "    model=llm.Qwen2Model(_get_qwen25_config()),\n"
        "    source=" + repr(f"hf://{hf_model_id}") + ",\n"
        "    output_path=output_path,\n"
        "    overwrite=False,\n"
        ")\n"
    )
    env = os.environ.copy()
    for key in (
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_NTASKS",
    ):
        env.pop(key, None)
    subprocess.check_call([sys.executable, "-c", code], env=env)


def _maybe_import_ckpt(hf_model_id: str, output_path: Path, skip_import: bool) -> Path:
    if skip_import:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel = Path(str(output_path) + ".done")
    rank = _get_rank()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if output_path.exists() or sentinel.exists():
        if rank == 0:
            sentinel.write_text("ok")
        return output_path

    if world_size > 1 and rank != 0:
        _wait_for_path(sentinel)
        return output_path

    if rank == 0:
        # Convert HF → NeMo once (shared filesystem required).
        if world_size > 1:
            _run_single_process_import(hf_model_id, output_path)
        else:
            llm.import_ckpt(
                model=llm.Qwen2Model(_get_qwen25_config()),
                source=f"hf://{hf_model_id}",
                output_path=output_path,
                overwrite=False,
            )
        sentinel.write_text("ok")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="NeMo 2.0 Qwen2.5 function-calling SFT")
    parser.add_argument("--dataset-root", required=True, help="Directory with training/validation/test.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory for logs/checkpoints")
    parser.add_argument("--hf-model-id", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--import-output", default="", help="Path to write NeMo .nemo checkpoint")
    parser.add_argument("--restore-path", default="", help="Path or nemo:// URL to restore from")
    parser.add_argument("--skip-import", action="store_true", help="Skip HF→NeMo conversion")
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--peft-scheme", default="none", choices=["none", "lora"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--recompute-granularity",
        default="selective",
        help="Activation recompute granularity (e.g., selective, full, none).",
    )
    parser.add_argument(
        "--recompute-method",
        nargs="?",
        const="none",
        default="none",
        help="Activation recompute method for full recompute (e.g., block, uniform).",
    )
    parser.add_argument(
        "--recompute-num-layers",
        type=int,
        default=0,
        help="Number of layers per recompute segment (only used for full recompute).",
    )
    parser.add_argument(
        "--gradient-accumulation-fusion",
        default="false",
        help="Enable fused weight gradient accumulation (true/false).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _validate_dataset_root(dataset_root)
    os.chdir(output_dir)

    nodes, gpus_per_node = _get_world_config()

    if args.peft_scheme == "none" and gpus_per_node < 2 and nodes < 2:
        raise RuntimeError("Full fine-tuning requires at least 2 GPUs; set --peft-scheme lora or scale up.")

    import_output = Path(args.import_output) if args.import_output else output_dir / "qwen25_14b_instruct.nemo"
    restore_path = args.restore_path.strip()

    if not restore_path:
        restore_path = str(import_output)

    restore_path_obj = Path(restore_path) if restore_path else None
    # If user explicitly provides nemo:// restore path or a local artifact exists, skip import.
    skip_import = args.skip_import or restore_path.startswith("nemo://") or (
        restore_path_obj is not None and restore_path_obj.exists()
    )
    if skip_import and not restore_path.startswith("nemo://"):
        if restore_path_obj is None or not restore_path_obj.exists():
            raise FileNotFoundError(
                f"Restore path not found: {restore_path_obj}. "
                "Run the HF→NeMo import first or unset --skip-import."
            )
    _maybe_import_ckpt(args.hf_model_id, import_output, skip_import)

    FineTuningDataModule = _resolve_finetune_data_module()

    finetune = llm.qwen25_14b.finetune_recipe(
        name="qwen25_14b_fc_sft",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        peft_scheme=args.peft_scheme,
    )
    finetune.data = FineTuningDataModule(
        dataset_root=str(dataset_root),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        num_workers=args.num_workers,
    )

    finetune.trainer.max_steps = args.max_steps
    finetune.trainer.log_every_n_steps = args.log_every_n_steps
    finetune.trainer.val_check_interval = args.val_check_interval
    applied = _apply_recompute_config(
        finetune,
        args.recompute_granularity,
        args.recompute_method,
        args.recompute_num_layers,
    )
    if (args.recompute_granularity or args.recompute_method or args.recompute_num_layers) and not applied:
        print("Warning: activation recompute requested but not supported by this recipe/model.")
    if hasattr(finetune.model, "config"):
        _apply_yarn(finetune.model.config, args.seq_length)
    grad_accum_fusion = _parse_bool(args.gradient_accumulation_fusion, default=False)
    grad_applied = _apply_gradient_accumulation_fusion(finetune, grad_accum_fusion)
    if not grad_applied:
        print(
            "Warning: gradient_accumulation_fusion flag not found in model config; "
            "fused wgrad may still be enabled."
        )
    if hasattr(finetune, "optim") and hasattr(finetune.optim, "config"):
        if hasattr(finetune.optim.config, "lr"):
            finetune.optim.config.lr = args.learning_rate
        if hasattr(finetune.optim.config, "warmup_steps"):
            finetune.optim.config.warmup_steps = args.warmup_steps
    if hasattr(finetune, "seed"):
        finetune.seed = args.seed

    finetune.resume.restore_config.path = restore_path

    # Run in the current torchrun context.
    run.run(finetune, direct=True)


if __name__ == "__main__":
    main()
