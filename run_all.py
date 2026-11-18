"""
Full re-run of every experiment on this computer.
Deep RL trains on the GPU (CUDA) when one is available; the simulations and
NSGA-II run on the CPU (all cores).

Started by RUN_ON_MY_PC.bat. Progress is printed and also written to
run_log.txt; a short summary is written to run_summary.txt at the end.
A step that fails is logged and the next step still runs.

Quick test (a few minutes):  python run_all.py --quick
"""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "run_log.txt")
M5 = os.path.join(ROOT, "results", "m5_foods_daily.npz")
QUICK = "--quick" in sys.argv


def steps():
    q = QUICK
    pop, gen = ("12", "3") if q else ("48", "60")
    reps = "10" if q else "200"
    rl_ep = "200" if q else "14000"
    dqn_ep = "60" if q else "8000"
    syn = [
        ("Synthetic | baseline (5,600 runs)", ["experiments.run_baseline", "--reps", reps]),
        ("Synthetic | baseline figures", ["experiments.plot_baseline"]),
        ("Synthetic | NSGA-II optimisation", ["experiments.run_optimization", "--pop", pop, "--gen", gen,
                                             "--train_reps", "3", "--test_reps", "5" if q else "30"]),
        ("Synthetic | frontier analysis", ["experiments.analyze_optimization"]),
        ("Synthetic | frontier figures", ["experiments.plot_optimization"]),
        ("Synthetic | sensitivity analysis", ["experiments.run_sensitivity"]),
        ("Synthetic | sensitivity figures", ["experiments.plot_sensitivity"]),
        ("Synthetic | ANOVA + node criticality", ["experiments.run_extras"]),
        ("Synthetic | tabular Q-learning", ["experiments.run_rl", rl_ep]),
        ("Synthetic | deep Q-network on GPU", ["experiments.train_dqn_torch", "--episodes", dqn_ep,
                                               "--hidden", "256", "--test_reps", "5" if q else "30"]),
    ]
    real = [(label.replace("Synthetic", "Real M5"), mod) for label, mod in syn]
    real += [("Real M5 | transfer test", ["experiments.run_transfer"]),
             ("Real M5 | demand comparison figure", ["experiments.plot_demand"])]
    return [(l, m, False) for l, m in syn] + [(l, m, True) for l, m in real]


def main():
    env0 = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    env0.pop("SCTWIN_M5", None)
    summary = []
    t_all = time.time()
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"\n===== run_all {'QUICK ' if QUICK else ''}started {time.ctime()} =====\n")
        try:
            import torch
            dev = f"CUDA: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "no CUDA GPU (deep RL on CPU)"
        except Exception as e:
            dev = f"torch not importable: {e}"
        print("Device:", dev); log.write(f"Device: {dev}\n")
        todo = steps()
        for i, (label, mod, real) in enumerate(todo, 1):
            env = dict(env0)
            if real:
                env["SCTWIN_M5"] = M5
            head = f"[{i}/{len(todo)}] {label}"
            print(f"\n{head}  ({time.strftime('%H:%M')})", flush=True)
            log.write(f"\n===== {head} {time.ctime()} =====\n"); log.flush()
            t = time.time()
            p = subprocess.Popen([sys.executable, "-m"] + mod, cwd=ROOT, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
            for line in p.stdout:
                if "Warning" in line or "return {k" in line:
                    continue
                print("   " + line.rstrip(), flush=True)
                log.write(line); log.flush()
            p.wait()
            status = "OK" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
            mins = (time.time() - t) / 60
            summary.append(f"{status:18s} {mins:6.1f} min  {label}")
            print(f"   -> {status} in {mins:.1f} min", flush=True)
        total = (time.time() - t_all) / 60
        text = f"Device: {dev}\nTotal: {total:.1f} min\n" + "\n".join(summary) + "\n"
        log.write("\n" + text)
    with open(os.path.join(ROOT, "run_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text + "\nALL DONE. You can close this window and tell Claude it has finished.")


if __name__ == "__main__":
    main()
