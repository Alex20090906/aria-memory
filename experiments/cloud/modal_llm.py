"""LLM server (patched research engine) on Modal.

Image: built from the patched LLM server source (<engine-src>\\llm, commits on top of upstream) with its own Dockerfile.
Volume "llm-data": the model files and the install made by LLM server's own setup on the first start (/data).

  modal run modal_llm.py::build         build the image only (CPU, no GPU billed)
  modal run modal_llm.py::install       first start on a GPU: LLM server downloads IQ3_S into the volume, then stops
  modal run modal_llm.py::probe         list what the install left in the volume

Budget guard: every GPU function has a hard timeout; nothing stays up after it returns.
"""
import modal

SRC = r"<engine-src>\llm"
ARCHS = "86;89;90;120"        # A10G / L4 / L40S / H100 on Modal, RTX 3090 / 5090 / PRO 6000 on GPUtw

image = (
    modal.Image.from_dockerfile(
        SRC + r"\Dockerfile.modal", context_dir=SRC,
        build_args={"CUDA_ARCHITECTURES": ARCHS, "BUILD_VISION": "0", "LLM_COMMIT": "818a934c4e2fbaccc71d557289e9285bc55b2ee1"},
        ignore=[".git", "build", "build*", "third_party/llama.cpp", "**/__pycache__", "apps", "bench/results"],
    )
    .entrypoint([])
    .run_commands("/opt/llm/terminal/python/bin/python3 -m pip install --no-cache-dir numpy pypinyin tokenizers",
                  "ln -sf /opt/llm/terminal/python/bin/python3 /usr/local/bin/python3 && "
                  "ln -sf /opt/llm/terminal/python/bin/python3 /usr/local/bin/python")
)
data = modal.Volume.from_name("llm-data", create_if_missing=True)
app = modal.App("llm-research", image=image)


@app.function(cpu=2, memory=4096, timeout=600)
def build():
    import subprocess
    print(subprocess.run(["ls", "/opt/llm/components/engine"], capture_output=True, text=True).stdout)
    print(subprocess.run(["sh", "-c", "/opt/llm/terminal/bin/llm --version"], capture_output=True, text=True).stdout)


@app.function(cpu=8, memory=16384, volumes={"/data": data}, timeout=3 * 3600)
def install(model: str = "IQ3_S"):
    """CPU only: the two GGUF shards straight from Hugging Face into the volume (pinned revision), then LLM server's own
    setup with the image's engine, no draft layer (its range download stalls) and no start"""
    import os
    import subprocess
    import time
    t0 = time.time()
    rev = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
    d = f"/data/models/{model}"
    os.makedirs(d, exist_ok=True)
    for i in (1, 2):
        fn = f"Qwen3.8-Flash-Next-GSQ-RCO-{model}-0000{i}-of-00002.gguf"
        if not os.path.exists(f"{d}/{fn}.done"):
            from huggingface_hub import hf_hub_download
            pth = hf_hub_download("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", f"{model}/{fn}", revision=rev,
                                  local_dir="/data/hf")
            os.replace(pth, f"{d}/{fn}")
            open(f"{d}/{fn}.done", "w").close()
            print(fn, "downloaded", round((time.time() - t0) / 60, 1), "min", flush=True)
        data.commit()
    env = dict(os.environ, LLM_INSTALL_DIR="/data/install")
    r = subprocess.run(["/opt/llm/terminal/python/bin/python3", "/opt/llm/terminal/setup.py", "--family", "qwen",
                        "--model", model, "--context", "32768", "--vision", "no", "--mtp", "off", "--yes", "--no-start",
                        "--gguf-dir", d], cwd="/opt/llm/terminal", env=env, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    data.commit()
    print(subprocess.run(["sh", "-c", "ls -la /data/install /opt/llm/terminal/llm-*.json 2>/dev/null | head -30"],
                         capture_output=True, text=True).stdout)


LAUNCH = ["/usr/local/bin/llm", "launch", "--model", "IQ3_S", "--model-path", "/data/models/IQ3_S", "--mtp", "off",
          "--drafting", "off", "--vision", "off", "--context", "32768", "--no-tui"]


def _seed_links():
    import os
    os.makedirs("/data/install/third_party", exist_ok=True)
    if not os.path.exists("/data/install/third_party/llama.cpp"):
        os.symlink("/opt/llm/seed/third_party/llama.cpp", "/data/install/third_party/llama.cpp")


@app.function(gpu="L4", cpu=4, memory=49152, volumes={"/data": data}, timeout=3600)
def prep():
    """LLM server's own launch (as the Docker entrypoint does) until its run settings exist, then stop"""
    import glob
    import os
    import subprocess
    import time
    _seed_links()
    for d in ("/data/state", "/data/home", "/data/LLM server-data"):
        os.makedirs(d, exist_ok=True)
    log = open("/tmp/launch.log", "w")
    p = subprocess.Popen(LAUNCH + ["--port", "8080"], stdout=log, stderr=subprocess.STDOUT,
                         env=dict(os.environ, LLM_LAUNCH_KEY="research-only-local-key-0001"))
    t0 = time.time()
    while time.time() - t0 < 3300 and p.poll() is None:
        if glob.glob("/data/install/llm-*.json"):
            time.sleep(30)
            break
        time.sleep(15)
    p.terminate()
    try:
        p.wait(60)
    except Exception:
        p.kill()
    subprocess.run(["sh", "-c", "/usr/local/bin/llm launch --stop >/dev/null 2>&1; pkill -f 'components/engine' ; true"])
    data.commit()
    print(open("/tmp/launch.log").read()[-5000:])
    print(subprocess.run(["sh", "-c", "ls -la /data/install; cat /data/install/llm-*.json | head -c 3000"],
                         capture_output=True, text=True).stdout)


@app.function(cpu=1, memory=2048, volumes={"/data": data}, timeout=300)
def probe(cmd: str = ""):
    import subprocess
    c = cmd or ("du -sh /data/* /data/install/* 2>/dev/null; ls /data/install; "
                "cat /data/install/llm-*.json 2>/dev/null | head -c 4000")
    r = subprocess.run(["sh", "-c", c], capture_output=True, text=True)
    print(r.stdout[-8000:], r.stderr[-2000:])


# ---------------------------------------------------------------- research runs
EXP = r"<workspace>\experiments\llm_fastmem"
rimage = image.add_local_dir(EXP, "/exp", ignore=["results", "*.json.bak", "__pycache__", "*.log", "memory*.bin"])


def _server(env, port=8093):
    """LLM server's server on the saved run settings with the fast-weight memory switched on; wait until loaded"""
    import glob
    import json
    import subprocess
    import time
    import urllib.request
    import re
    cfgs = sorted(glob.glob("/data/install/llm-*.json") + glob.glob("/opt/llm/terminal/llm-*.json"))
    raw = open(cfgs[0]).read()
    raw = re.sub(r"/__modal/volumes/vo-[A-Za-z0-9]+", "/data", raw)   # the volume's own path in the container that wrote it
    cfg = "/tmp/llm-run.json"
    open(cfg, "w").write(raw)
    prof = json.loads(raw)
    log = open("/tmp/serve.log", "w")
    py = "/opt/llm/terminal/python/bin/python3"
    p = subprocess.Popen([py, "/opt/llm/terminal/serve/server.py", "--engine", "llm", "--config", cfg,
                          "--port", str(port)], env=env, stdout=log, stderr=subprocess.STDOUT, cwd="/opt/llm/terminal")
    t0 = time.time()
    while time.time() - t0 < 2400:
        if p.poll() is not None:
            raise RuntimeError("server exited: " + open("/tmp/serve.log").read()[-3000:])
        try:
            h = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read().decode()
            if '"loaded": true' in h or '"loaded":true' in h:
                print("server loaded in", round(time.time() - t0), "s", flush=True)
                return p, prof
        except Exception:
            pass
        time.sleep(10)
    raise RuntimeError("server did not load: " + open("/tmp/serve.log").read()[-3000:])


@app.function(gpu="L40S", cpu=4, memory=65536, volumes={"/data": data}, timeout=2 * 3600, image=rimage)
def research(tag: str, cmd: str, envs: dict = None, D: int = 65536):
    """one experiment in its own container: own memory file on local disk, caches from the volume"""
    import glob
    import os
    import shutil
    import subprocess
    import time
    t0 = time.time()
    run = "/tmp/fmem"
    os.makedirs(run, exist_ok=True)
    cache = "/data/fmem_cache"
    os.makedirs(cache, exist_ok=True)
    for fn in glob.glob(cache + "/*"):
        shutil.copy(fn, run)
    shards = sorted(glob.glob("/data/models/IQ3_S/*IQ3_S-00001-of-00002.gguf"))
    tokz = sorted(glob.glob("/data/packs/*/tokenizer/vocab.json") + glob.glob("/data/install/**/tokenizer/vocab.json", recursive=True) +
                  glob.glob("/opt/llm/terminal/**/packs/*/tokenizer/vocab.json", recursive=True))
    env = dict(os.environ, PYTHONUTF8="1", FMEM_LLM="/opt/llm/terminal", FMEM_GGUF=shards[0],
               FMEM_TOKENIZER=tokz[0] if tokz else "", FMEM_DATA=run, FMEM_URL="http://127.0.0.1:8093/v1/chat/completions",
               LLM_FMEM_FILE=run + "/memory.bin", LLM_FMEM_CAPTURE=run + "/capture.bin", LLM_FMEM_HEAD="1",
               LLM_FMEM_ANCHOR=run + "/anchor.bin", LLM_FMEM_DEBUG="1", LLM_FMEM_SHIFT_NOTOK="1",
               LLM_FMEM_ROWSCALE=run + "/rowscale.bin", FMEM_ROWSCALE_FILE=run + "/rowscale.bin", FMEM_DIRTY="1")
    env.update(envs or {})
    if env.get("FMEM_ROWSCALE_FILE") == "":
        env.pop("LLM_FMEM_ROWSCALE", None)
    py = "/opt/llm/terminal/python/bin/python3"
    fresh = not os.path.exists(run + "/memory.bin")
    if fresh:
        # memory file before the server starts (the engine reads it at load); caches are built once and kept
        subprocess.run([py, "fmem_tool.py", "init", "--layers", "38", "--D", str(D), "--k", "16"], cwd="/exp", env=env, check=True)
    srv, _ = _server(env)
    try:
        if fresh:
            subprocess.run([py, "fmem_tool.py", "calibrate", "calib_sentences.txt"], cwd="/exp", env=env, check=True)
            for fn in ("unembed_f16.npy", "unembed_mean.npy", "token_embd_f16.npy", "memory.bin"):
                if os.path.exists(f"{run}/{fn}") and not os.path.exists(f"{cache}/{fn}"):
                    shutil.copy(f"{run}/{fn}", cache)
        out = subprocess.run(["sh", "-c", cmd], cwd="/exp", env=env, capture_output=True, text=True)
        for fn in ("anchor_P_%d.npy" % D, "anchor_stats.npz"):
            if os.path.exists(f"{run}/{fn}") and not os.path.exists(f"{cache}/{fn}"):
                shutil.copy(f"{run}/{fn}", cache)
        os.makedirs("/data/results/" + tag, exist_ok=True)
        open(f"/data/results/{tag}/stdout.log", "w").write(out.stdout + "\n--- stderr ---\n" + out.stderr[-20000:])
        for fn in glob.glob("/exp/*.json"):
            if os.path.getmtime(fn) > t0:
                shutil.copy(fn, f"/data/results/{tag}/")
        shutil.copy("/tmp/serve.log", f"/data/results/{tag}/serve.log")
        data.commit()
        tail = "\n".join([l for l in out.stdout.splitlines() if l.startswith(("MODE", "  recall", "SUMMARY", "  no-memory"))][-8:])
        return {"tag": tag, "rc": out.returncode, "minutes": round((time.time() - t0) / 60, 1), "tail": tail,
                "err": out.stderr[-1500:] if out.returncode else ""}
    finally:
        srv.terminate()


@app.function(cpu=2, memory=8192, volumes={"/data": data}, timeout=1800, image=rimage)
def bundle():
    """the built engine + Terminal + experiment code as one archive, for GPUtw (same Ubuntu 24.04 userland there)"""
    import subprocess
    os_ = __import__("os")
    os_.makedirs("/data/bundle", exist_ok=True)
    r = subprocess.run(["sh", "-c", "cp /usr/local/bin/llm /opt/llm/llm-wrapper.sh && "
                        "tar -C / -czf /data/bundle/llm_opt.tgz opt/llm && "
                        "tar -C / -czf /data/bundle/exp.tgz exp && ls -la /data/bundle"], capture_output=True, text=True)
    data.commit()
    print(r.stdout, r.stderr[-2000:])


@app.local_entrypoint()
def batch(plan: str, max_usd: float = 0.0):
    """plan: a JSON file [{"tag", "cmd", "envs"}]; every run in its own L40S container, in parallel.
    The local ledger (budget.py) gets each run's cost from its minutes; refuses to start past the cap."""
    import json
    import sys
    sys.path.insert(0, r"<workspace>\experiments\cloud")
    import budget
    runs = json.load(open(plan, encoding="utf-8"))
    per_hour = budget.modal_cost("L40S", 4, 64, 3600)
    est = per_hour * 2.0 * len(runs)            # worst case = the 2 h function timeout
    left = budget.modal_left()
    cap = max_usd or left
    print(f"{len(runs)} runs, worst case USD {est:.2f}, ledger left USD {left:.2f}", flush=True)
    if est > cap:
        raise SystemExit("over budget: shrink the plan")
    calls = [research.spawn(r["tag"], r["cmd"], r.get("envs", {})) for r in runs]
    out = []
    for c, r in zip(calls, runs):
        try:
            res = c.get()
        except Exception as e:
            res = {"tag": r["tag"], "rc": -1, "minutes": 240, "err": repr(e)[:1500]}
        budget.add("modal", budget.modal_cost("L40S", 4, 64, 60 * (res.get("minutes") or 240)), "research " + r["tag"])
        print(json.dumps(res, ensure_ascii=False), flush=True)
        out.append(res)
    json.dump(out, open(plan.replace(".json", "_out.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
