"""CLI entry point: python -m eval.run [phase0|stt|tts|all|report] [--test NAME]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from eval.config import load_config
from eval.utils import setup_logging


def _run_phase0(config):
    from eval.phase0.discovery import run
    return run(config)


def _run_stt_all(config):
    from eval.stt import (
        accuracy, performance, streaming, rest_vs_grpc,
        noise_robustness, accent, long_form, domain,
        output_quality, confidence, format_robustness,
    )
    results = []
    tests = [
        ("1.1", accuracy),
        ("1.2", performance),
        ("1.3", streaming),
        ("1.4", rest_vs_grpc),
        ("1.5", noise_robustness),
        ("1.6", accent),
        ("1.7", long_form),
        ("1.8", domain),
        ("1.9", output_quality),
        ("1.10", confidence),
        ("1.11", format_robustness),
    ]
    for tid, mod in tests:
        logging.getLogger("eval.run").info("--- Running STT Test %s ---", tid)
        try:
            r = mod.run(config)
            results.append(r)
        except Exception as e:
            logging.getLogger("eval.run").error("STT Test %s failed: %s", tid, e)
            results.append({"test": tid, "error": str(e)})
    return results


def _run_tts_all(config):
    from eval.tts import (
        naturalness, intelligibility, prosody, signal_quality,
        latency, concurrency, edge_cases, long_form,
    )
    results = []
    tests = [
        ("2.1", naturalness),
        ("2.2", intelligibility),
        ("2.3", prosody),
        ("2.4", signal_quality),
        ("2.5", latency),
        ("2.6", concurrency),
        ("2.7", edge_cases),
        ("2.8", long_form),
    ]
    for tid, mod in tests:
        logging.getLogger("eval.run").info("--- Running TTS Test %s ---", tid)
        try:
            r = mod.run(config)
            results.append(r)
        except Exception as e:
            logging.getLogger("eval.run").error("TTS Test %s failed: %s", tid, e)
            results.append({"test": tid, "error": str(e)})
    return results


def _run_single_stt(name: str, config):
    mod_map = {
        "accuracy": "eval.stt.accuracy",
        "performance": "eval.stt.performance",
        "streaming": "eval.stt.streaming",
        "rest_vs_grpc": "eval.stt.rest_vs_grpc",
        "noise_robustness": "eval.stt.noise_robustness",
        "accent": "eval.stt.accent",
        "long_form": "eval.stt.long_form",
        "domain": "eval.stt.domain",
        "output_quality": "eval.stt.output_quality",
        "confidence": "eval.stt.confidence",
        "format_robustness": "eval.stt.format_robustness",
    }
    key = name.replace("stt.", "").replace("1.", "").strip()
    # Allow numeric IDs like "1.1" -> "accuracy"
    num_map = {
        "1": "accuracy", "2": "performance", "3": "streaming", "4": "rest_vs_grpc",
        "5": "noise_robustness", "6": "accent", "7": "long_form", "8": "domain",
        "9": "output_quality", "10": "confidence", "11": "format_robustness",
    }
    if key in num_map:
        key = num_map[key]
    if key not in mod_map:
        print(f"Unknown STT test: {name}. Available: {list(mod_map)}")
        sys.exit(1)
    import importlib
    mod = importlib.import_module(mod_map[key])
    return mod.run(config)


def _run_single_tts(name: str, config):
    mod_map = {
        "naturalness": "eval.tts.naturalness",
        "intelligibility": "eval.tts.intelligibility",
        "prosody": "eval.tts.prosody",
        "signal_quality": "eval.tts.signal_quality",
        "latency": "eval.tts.latency",
        "concurrency": "eval.tts.concurrency",
        "edge_cases": "eval.tts.edge_cases",
        "long_form": "eval.tts.long_form",
    }
    key = name.replace("tts.", "").replace("2.", "").strip()
    num_map = {
        "1": "naturalness", "2": "intelligibility", "3": "prosody",
        "4": "signal_quality", "5": "latency", "6": "concurrency",
        "7": "edge_cases", "8": "long_form",
    }
    if key in num_map:
        key = num_map[key]
    if key not in mod_map:
        print(f"Unknown TTS test: {name}. Available: {list(mod_map)}")
        sys.exit(1)
    import importlib
    mod = importlib.import_module(mod_map[key])
    return mod.run(config)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m eval.run",
        description="NVIDIA Riva Evaluation Harness",
    )
    parser.add_argument(
        "command",
        choices=["phase0", "stt", "tts", "all", "report", "download"],
        help="What to run",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Run a single test, e.g. --test accuracy or --test naturalness",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: eval/config.yaml)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override results directory",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and datasets but make no API calls",
    )

    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    log = logging.getLogger("eval.run")

    config = load_config(args.config)
    if args.results_dir:
        config.evaluation.results_dir = args.results_dir

    if args.dry_run:
        log.info("[DRY RUN] Config loaded. Command=%s test=%s", args.command, args.test)
        log.info("  STT gRPC : %s", config.stt.grpc_uri)
        log.info("  TTS gRPC : %s", config.tts.grpc_uri)
        log.info("  STT REST : %s", config.stt.rest_endpoint)
        log.info("  TTS REST : %s", config.tts.rest_endpoint)
        log.info("  Results  : %s", config.evaluation.results_dir)
        return

    log.info("Starting evaluation. Command=%s", args.command)

    if args.command == "phase0":
        result = _run_phase0(config)
        log.info("Phase 0 complete: %s", result.get("overall_status", "done"))

    elif args.command == "stt":
        if args.test:
            result = _run_single_stt(args.test, config)
            log.info("Done: %s", result.get("name", args.test))
        else:
            results = _run_stt_all(config)
            passed = sum(1 for r in results if "error" not in r)
            log.info("STT complete: %d/%d passed", passed, len(results))

    elif args.command == "tts":
        if args.test:
            result = _run_single_tts(args.test, config)
            log.info("Done: %s", result.get("name", args.test))
        else:
            results = _run_tts_all(config)
            passed = sum(1 for r in results if "error" not in r)
            log.info("TTS complete: %d/%d passed", passed, len(results))

    elif args.command == "all":
        log.info("=== Phase 0 ===")
        _run_phase0(config)
        log.info("=== STT Tests ===")
        _run_stt_all(config)
        log.info("=== TTS Tests ===")
        _run_tts_all(config)
        log.info("=== Generating Report ===")
        from eval.report.generate_report import generate
        report_path = generate(config.evaluation.results_dir)
        log.info("Report: %s", report_path)

    elif args.command == "download":
        from eval.data.download_datasets import download_all
        log.info("Pre-downloading all evaluation datasets...")
        results = download_all()
        ok = sum(1 for v in results.values() if v is not None)
        log.info("Downloaded %d/%d datasets", ok, len(results))
        for name, ds in results.items():
            status = f"{len(ds)} examples" if ds is not None else "FAILED"
            log.info("  %-20s %s", name, status)

    elif args.command == "report":
        from eval.report.generate_report import generate
        report_path = generate(config.evaluation.results_dir)
        print(f"Report written to: {report_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
