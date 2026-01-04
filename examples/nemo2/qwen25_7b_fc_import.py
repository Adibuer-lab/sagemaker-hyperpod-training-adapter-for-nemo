#!/usr/bin/env python3
# NeMo 2.0 Qwen2.5-Coder-7B HF -> NeMo import (single-process).

import argparse
from pathlib import Path

import nemo.collections.llm as llm


def _apply_yarn(model_cfg, seq_length: int) -> bool:
    # Apply YaRN rope scaling if the config supports it.
    yarn_cfg = {
        "rope_type": "yarn",
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
    }
    applied = False

    for attr in ("seq_length", "max_position_embeddings"):
        if hasattr(model_cfg, attr):
            try:
                setattr(model_cfg, attr, seq_length)
                applied = True
            except Exception:
                pass

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


def _get_qwen25_config():
    for name in ("Qwen25Config7B", "Qwen25Config"):
        cfg = getattr(llm, name, None)
        if cfg is not None:
            return cfg()
    raise AttributeError("Unable to locate Qwen2.5 config in nemo.collections.llm")


def parse_args():
    parser = argparse.ArgumentParser(description="NeMo 2.0 Qwen2.5-Coder-7B HF -> NeMo import")
    parser.add_argument("--hf-model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output-path", required=True, help="Path to write the NeMo checkpoint")
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"Import skipped (exists): {output_path}")
        return

    cfg = _get_qwen25_config()
    _apply_yarn(cfg, args.seq_length)
    model = llm.Qwen2Model(cfg)

    llm.import_ckpt(
        model=model,
        source=f"hf://{args.hf_model_id}",
        output_path=output_path,
        overwrite=args.overwrite,
    )
    print(f"Imported checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
