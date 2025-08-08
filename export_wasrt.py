"""Utilities to export WaSR-T models to TensorRT engines."""

import argparse
from pathlib import Path

import torch

try:
    import tensorrt as trt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "TensorRT is required to build an engine. "
        "Install it on the Jetson first."
    ) from exc

from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.utils import load_weights


def export_to_onnx(
    weights: Path,
    onnx_path: Path,
    *,
    hist_len: int = 5,
) -> None:
    """Export a WaSR-T model to ONNX format."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = wasr_temporal_resnet101(pretrained=False, hist_len=hist_len)
    state_dict = load_weights(str(weights))
    model.load_state_dict(state_dict)
    model = model.sequential().to(device)
    model.eval()

    dummy = torch.randn(1, 3 * (hist_len + 1), 384, 512, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
) -> None:
    """Create a TensorRT engine from an ONNX model."""
    logger = trt.Logger(trt.Logger.WARNING)
    with (
        trt.Builder(logger) as builder,
        builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network,
        trt.OnnxParser(network, logger) as parser,
    ):
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        engine = builder.build_engine(network, config)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())


def main() -> None:
    """Parse arguments and export the engine."""
    parser = argparse.ArgumentParser(
        description="Export WaSR-T to TensorRT",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to .pth weights",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("wasrt.onnx"),
        help="ONNX output path",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("wasrt.trt"),
        help="TensorRT engine output path",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 precision optimizations",
    )
    args = parser.parse_args()

    export_to_onnx(args.weights, args.onnx)
    build_engine(args.onnx, args.engine, fp16=not args.no_fp16)


if __name__ == "__main__":  # pragma: no cover
    main()