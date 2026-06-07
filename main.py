"""Entry point: single run or Ray parameter sweep.

Usage:
    python main.py                            # default 500-tick run with visualization
    python main.py --seed 42 --ticks 200      # deterministic run
    python main.py --no-visualize             # headless run
    python main.py --sweep --n-runs 100 --output sweep.duckdb
"""

import argparse
import sys

from config import SimConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Civilization simulation")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--ticks",        type=int, default=500)
    p.add_argument("--width",        type=int, default=80)
    p.add_argument("--height",       type=int, default=60)
    p.add_argument("--cities",       type=int, default=4, help="Cities per civ")
    p.add_argument("--db",           type=str, default="results.duckdb")
    p.add_argument("--snapshot-interval", type=int, default=0,
                   metavar="N",
                   help="Write a world snapshot every N ticks to the DB (0 = off)")
    p.add_argument("--no-visualize",   action="store_true")
    p.add_argument("--terminal-viz",   action="store_true",
                   help="Use ANSI terminal renderer instead of matplotlib (works over SSH)")
    p.add_argument("--provider",  type=str, default=None,
                   choices=["rule_based", "openai_compatible", "anthropic"],
                   help="Decision provider for all civs")
    p.add_argument("--model",     type=str,
                   default="meta-llama/Llama-3.1-70B-Instruct",
                   help="Model name for LLM provider")
    p.add_argument("--base-url",  type=str, default="http://localhost:8000/v1",
                   help="Base URL for OpenAI-compatible endpoint")
    p.add_argument("--api-key",   type=str, default="EMPTY",
                   help="API key (use EMPTY for local vLLM/Ollama)")
    p.add_argument("--config",    type=str, default=None,
                   help="YAML file with per-civ provider config")
    # DGX Spark optimisation flags
    p.add_argument("--batch-mode",       action="store_true",
                   help="Use /v1/completions batch API (vLLM DGX mode)")
    p.add_argument("--max-concurrent",   type=int, default=64,
                   help="Max concurrent LLM requests when batch-mode is off")
    p.add_argument("--prompt-template",  type=str, default="",
                   help="Chat template for completions API; empty = Llama-3.1 default. "
                        "Use {system} and {user} placeholders.")
    p.add_argument("--workers",          type=int, default=0,
                   help="Ray workers for sweep (0 = one per CPU core)")
    p.add_argument("--grid-backend",     type=str, default="numpy",
                   choices=["numpy", "cupy"],
                   help="Array backend for resource grid")
    p.add_argument("--sweep",        action="store_true")
    p.add_argument("--n-runs",       type=int, default=100)
    p.add_argument("--output",       type=str, default="sweep.duckdb")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = SimConfig(
        rng_seed=args.seed,
        max_ticks=args.ticks,
        width=args.width,
        height=args.height,
        cities_per_civ=args.cities,
        db_path=args.db,
        snapshot_interval=args.snapshot_interval,
        visualize=not args.no_visualize,
        num_sweep_workers=args.workers,
        grid_backend=args.grid_backend,
    )

    from config import ProviderConfig

    if args.config:
        import yaml
        with open(args.config) as f:
            data = yaml.safe_load(f)
        cfg.civ_providers = [
            ProviderConfig(**entry)
            for entry in data.get("civ_providers", [])
        ]
    elif args.provider:
        provider_cfg = ProviderConfig(
            type=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            use_completions_api=args.batch_mode,
            max_concurrent=args.max_concurrent,
            prompt_template=args.prompt_template,
        )
        if args.provider == "anthropic" and "llama" in args.model.lower():
            print(f"[warn] Using provider=anthropic with model={args.model!r} — "
                  f"this looks like a local model. Did you mean to use --provider openai_compatible?",
                  file=sys.stderr)
        cfg.civ_providers = [provider_cfg] * cfg.num_civs

    if args.sweep:
        from simulation.runner import run_sweep
        print(f"Starting sweep: {args.n_runs} runs → {args.output}")
        run_sweep(args.n_runs, cfg, args.output, num_workers=args.workers)
        return

    # Single run
    from simulation.model import CivModel

    model = CivModel(cfg)
    renderer = None

    if args.terminal_viz:
        try:
            from visualization.terminal_renderer import TerminalRenderer
            renderer = TerminalRenderer(model)
        except Exception as e:
            print(f"[warn] Terminal visualization failed: {e}", file=sys.stderr)
    elif cfg.visualize:
        try:
            from visualization.renderer import Renderer
            renderer = Renderer(model)
        except Exception as e:
            print(f"[warn] matplotlib display unavailable ({e}); falling back to terminal view",
                  file=sys.stderr)
            try:
                from visualization.terminal_renderer import TerminalRenderer
                renderer = TerminalRenderer(model)
            except Exception as e2:
                print(f"[warn] Terminal visualization also failed: {e2}", file=sys.stderr)

    print(f"Seed={cfg.rng_seed}  Grid={cfg.width}×{cfg.height}  "
          f"Cities/civ={cfg.cities_per_civ}  MaxTicks={cfg.max_ticks}")
    for civ in model.civilizations:
        t = civ.traits
        print(f"  {civ.name}: agg={t.aggressiveness:.2f} trust={t.trust:.2f} "
              f"innov={t.innovation:.2f} trib={t.tribalism:.2f} risk={t.risk_tolerance:.2f}")

    while model.running:
        model.step()
        if renderer is not None:
            renderer.update(model)
        if model.steps % 50 == 0:
            for civ in model.civilizations:
                print(f"  tick={model.steps:4d}  {civ}")

    print(f"\nSimulation ended at tick {model.steps}")
    alive = [c for c in model.civilizations if c.alive]
    if len(alive) == 1:
        print(f"Winner: {alive[0].name}")
    elif len(alive) > 1:
        print("No winner — both civilizations survive")
    else:
        print("Both civilizations collapsed")

    print(f"Events logged to: {cfg.db_path}")
    if renderer:
        input("Press Enter to close…")


if __name__ == "__main__":
    main()
