from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from mageflow_native.inference.runner import run_generation
from mageflow_native.models.manifest import sha256_file
from mageflow_native.runtime.manager import RuntimeManager
from mageflow_native.runtime.spec import BackendSpec
from mageflow_native.telemetry import read_mem_available_kb

Q8_PROFILE = "q8-reference"
BF16_PROFILE = "bf16-high-memory-cpu"
CPU_BACKEND = "cpu"
SCHEMA_VERSION = 2

DEFAULT_MANIFEST_RESOLUTION = 768
DEFAULT_STEPS = 4
DEFAULT_CFG = 1.0
DEFAULT_THREADS = 4

RUNTIME_SHA256 = "7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c"
RUNTIME_COMMIT = "6b3edaaf32cc19e5bb2d819c788bd557eddc8eba"
Q8_DIFFUSION_SHA256 = "4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0"
BF16_DIFFUSION_SHA256 = "6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d"
SHARED_TEXT_ENCODER_SHA256 = "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a"
SHARED_VAE_SHA256 = "34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_PROMPT_IDS = {f"P{i:02d}" for i in range(1, 11)}
_EXPECTED_SEEDS = set(range(1001, 1011))


class VisualBenchmarkError(RuntimeError):
    pass


def _validate_hex64(value: str, label: str) -> None:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise VisualBenchmarkError(f"{label} must be a 64-hex SHA-256, got {value!r}")


@dataclass(frozen=True)
class HostFingerprint:
    hostname: str
    linux_boot_id: str
    cpu_model: str
    mem_total_kb: int

    def to_dict(self) -> dict:
        return {
            "hostname": self.hostname,
            "linux_boot_id": self.linux_boot_id,
            "cpu_model": self.cpu_model,
            "mem_total_kb": self.mem_total_kb,
        }

    @classmethod
    def current(cls) -> "HostFingerprint":
        hostname = platform.node() or ""
        boot_id = ""
        try:
            boot_id = (
                Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            )
        except OSError:
            pass
        cpu_model = ""
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        mem_total_kb = _mem_total_kb()
        return cls(
            hostname=hostname,
            linux_boot_id=boot_id,
            cpu_model=cpu_model,
            mem_total_kb=mem_total_kb,
        )


def _mem_total_kb() -> int:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
    return (page_size * phys_pages) // 1024


def _source_head(repo_dir: Path) -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        )
    except Exception:
        return "unknown"


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import os as _os
    tmp = path.with_name(path.name + f".tmp{_os.getpid()}{time.monotonic_ns()}")
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    try:
        fh = _os.open(str(tmp), _os.O_WRONLY | _os.O_CREAT | _os.O_EXCL, 0o644)
        with _os.fdopen(fh, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


@dataclass
class Manifest:
    path: Path
    sha256: str
    data: dict
    source_head: str
    prompts: list[dict]
    paired_profiles: list[str]
    resolution: int
    steps: int
    cfg: float
    threads: int
    paired_run_size: int

    @property
    def benchmark_id(self) -> str:
        return self.data["benchmark_id"]

    @property
    def seeds_by_prompt(self) -> dict[str, int]:
        return {p["id"]: p["seed"] for p in self.prompts}


def load_benchmark_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    if not path.is_file():
        raise VisualBenchmarkError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(path)

    required_top = {
        "schema_version",
        "benchmark_id",
        "design_status",
        "source_head",
        "same_host_policy",
        "provenance",
        "generation",
        "prompts",
        "paired_run_size",
        "evidence_contract",
    }
    missing = required_top - set(data.keys())
    if missing:
        raise VisualBenchmarkError(f"manifest missing fields: {sorted(missing)}")

    source_head = data["source_head"]
    if source_head.startswith("__FREEZE"):
        raise VisualBenchmarkError(
            "manifest source_head is a placeholder; an execution manifest is required"
        )
    if not isinstance(source_head, str) or not _HEX40.match(source_head):
        raise VisualBenchmarkError(
            f"source_head must be a 40-hex Git SHA-1, got {source_head!r}"
        )

    gen = data["generation"]
    resolution = gen["resolution"]["width"]
    if resolution != gen["resolution"]["height"]:
        raise VisualBenchmarkError("manifest resolution must be square")
    if resolution != DEFAULT_MANIFEST_RESOLUTION:
        raise VisualBenchmarkError(
            f"manifest resolution must be {DEFAULT_MANIFEST_RESOLUTION}, got {resolution}"
        )
    if gen.get("backend") != CPU_BACKEND:
        raise VisualBenchmarkError(f"manifest backend must be cpu, got {gen.get('backend')}")
    if gen.get("steps") != DEFAULT_STEPS:
        raise VisualBenchmarkError(f"manifest steps must be {DEFAULT_STEPS}")
    if gen.get("cfg") != DEFAULT_CFG:
        raise VisualBenchmarkError(f"manifest cfg must be {DEFAULT_CFG}")
    if gen.get("threads") != DEFAULT_THREADS:
        raise VisualBenchmarkError(f"manifest threads must be {DEFAULT_THREADS}")

    paired_profiles = list(gen["paired_profiles"])
    if paired_profiles != [Q8_PROFILE, BF16_PROFILE]:
        raise VisualBenchmarkError(
            f"paired_profiles must be {[Q8_PROFILE, BF16_PROFILE]}, got {paired_profiles}"
        )

    prompts = data["prompts"]
    if not isinstance(prompts, list):
        raise VisualBenchmarkError("prompts must be a list")
    ids = [p["id"] for p in prompts]
    if set(ids) != _EXPECTED_PROMPT_IDS:
        raise VisualBenchmarkError(f"prompt ids must be exactly P01..P10, got {sorted(ids)}")
    if len(ids) != len(set(ids)):
        raise VisualBenchmarkError("duplicate prompt id")
    seeds = [p["seed"] for p in prompts]
    if set(seeds) != _EXPECTED_SEEDS:
        raise VisualBenchmarkError(f"seeds must be exactly 1001..1010, got {sorted(seeds)}")
    if len(seeds) != len(set(seeds)):
        raise VisualBenchmarkError("duplicate seed")
    for p in prompts:
        order = list(p["execution_order"])
        if order != _expected_order_for_prompt(p["id"]):
            raise VisualBenchmarkError(
                f"prompt {p['id']} execution_order must be balanced, got {order}"
            )

    runtime = data["provenance"]["runtime"]
    if runtime["commit"] != RUNTIME_COMMIT:
        raise VisualBenchmarkError("runtime commit mismatch")
    if runtime["sha256"] != RUNTIME_SHA256:
        raise VisualBenchmarkError("runtime sha256 mismatch")

    te = data["provenance"]["text_encoder"]["sha256"]
    vae = data["provenance"]["vae"]["sha256"]
    if te != SHARED_TEXT_ENCODER_SHA256:
        raise VisualBenchmarkError("text encoder sha256 mismatch")
    if vae != SHARED_VAE_SHA256:
        raise VisualBenchmarkError("vae sha256 mismatch")

    profiles = data["provenance"]["profiles"]
    if profiles[Q8_PROFILE]["diffusion_sha256"] != Q8_DIFFUSION_SHA256:
        raise VisualBenchmarkError("q8 diffusion sha256 mismatch")
    if profiles[BF16_PROFILE]["diffusion_sha256"] != BF16_DIFFUSION_SHA256:
        raise VisualBenchmarkError("bf16 diffusion sha256 mismatch")

    paired_run_size = int(data["paired_run_size"])
    if paired_run_size != 20:
        raise VisualBenchmarkError("paired_run_size must be 20")

    return Manifest(
        path=path,
        sha256=manifest_sha256,
        data=data,
        source_head=source_head,
        prompts=prompts,
        paired_profiles=paired_profiles,
        resolution=resolution,
        steps=gen["steps"],
        cfg=gen["cfg"],
        threads=gen["threads"],
        paired_run_size=paired_run_size,
    )


def _expected_order_for_prompt(prompt_id: str) -> list[str]:
    num = int(prompt_id[1:])
    if num % 2 == 1:
        return [Q8_PROFILE, BF16_PROFILE]
    return [BF16_PROFILE, Q8_PROFILE]


@dataclass(frozen=True)
class RunPlanItem:
    execution_index: int
    prompt_id: str
    prompt_class: str
    profile: str
    prompt: str
    seed: int


def build_run_plan(manifest: Manifest) -> list[RunPlanItem]:
    plan: list[RunPlanItem] = []
    index = 0
    for prompt in manifest.prompts:
        for profile in prompt["execution_order"]:
            plan.append(
                RunPlanItem(
                    execution_index=index,
                    prompt_id=prompt["id"],
                    prompt_class=prompt["class"],
                    profile=profile,
                    prompt=prompt["prompt"],
                    seed=prompt["seed"],
                )
            )
            index += 1
    if len(plan) != manifest.paired_run_size:
        raise VisualBenchmarkError(
            f"run plan size {len(plan)} != paired_run_size {manifest.paired_run_size}"
        )
    return plan


def plan_to_dict(plan: list[RunPlanItem]) -> dict:
    return {
        "backend": CPU_BACKEND,
        "resolution": DEFAULT_MANIFEST_RESOLUTION,
        "steps": DEFAULT_STEPS,
        "cfg": DEFAULT_CFG,
        "threads": DEFAULT_THREADS,
        "execution": "sequential-paired-balanced-order",
        "runs": [
            {
                "execution_index": item.execution_index,
                "prompt_id": item.prompt_id,
                "prompt_class": item.prompt_class,
                "profile": item.profile,
                "seed": item.seed,
                "prompt": item.prompt,
            }
            for item in plan
        ],
        "run_count": len(plan),
    }


class BenchmarkRoot:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs_dir = root / "evidence" / "runs"
        self.images_dir = root / "evidence" / "images"

    def record_path(self, prompt_id: str, profile: str) -> Path:
        return self.runs_dir / prompt_id / f"{profile}.json"

    def image_path(self, prompt_id: str, profile: str) -> Path:
        return self.images_dir / prompt_id / f"{profile}.png"


class VisualBenchmark:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        repo_dir: Path,
        work_root: Path,
        compile_manifests: bool = False,
        fake: bool = False,
    ) -> None:
        self.manifest = load_benchmark_manifest(manifest_path)
        self.repo_dir = repo_dir
        self.work_root = work_root
        self.root = BenchmarkRoot(work_root / "evidence")
        self.compile_manifests = compile_manifests
        self.fake = fake
        self.fingerprint = HostFingerprint.current()
        self.run_plan = build_run_plan(self.manifest)

    def validate(self) -> None:
        errors: list[str] = []
        try:
            _validate_hex64(RUNTIME_SHA256, "runtime sha256")
        except VisualBenchmarkError as exc:
            errors.append(str(exc))
        try:
            _validate_hex64(Q8_DIFFUSION_SHA256, "q8 diffusion sha256")
        except VisualBenchmarkError as exc:
            errors.append(str(exc))
        try:
            _validate_hex64(BF16_DIFFUSION_SHA256, "bf16 diffusion sha256")
        except VisualBenchmarkError as exc:
            errors.append(str(exc))
        try:
            _validate_hex64(SHARED_TEXT_ENCODER_SHA256, "text encoder sha256")
        except VisualBenchmarkError as exc:
            errors.append(str(exc))
        try:
            _validate_hex64(SHARED_VAE_SHA256, "vae sha256")
        except VisualBenchmarkError as exc:
            errors.append(str(exc))

        current_head = _source_head(self.repo_dir)
        if current_head != self.manifest.source_head:
            errors.append(
                "source_head mismatch: current HEAD "
                f"{current_head} != manifest source_head {self.manifest.source_head}"
            )

        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.repo_dir),
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            if status:
                errors.append("working tree is not clean")
        except Exception as exc:
            errors.append(f"could not inspect git status: {exc}")

        if self.manifest.data["same_host_policy"].get("required") is not True:
            errors.append("same-host policy is not required")

        if errors:
            raise VisualBenchmarkError(";\n".join(errors))

    def _validate_run_item_for_resume(self, item: RunPlanItem, record: dict) -> list[str]:
        errors: list[str] = []
        if record.get("status") != "passed":
            errors.append(f"{item.prompt_id}/{item.profile}: record status not passed")
        if record.get("manifest_sha256") != self.manifest.sha256:
            errors.append(f"{item.prompt_id}/{item.profile}: manifest sha mismatch")
        if record.get("source_head") != self.manifest.source_head:
            errors.append(f"{item.prompt_id}/{item.profile}: source_head mismatch")
        if record.get("host_fingerprint") != self.fingerprint.to_dict():
            errors.append(f"{item.prompt_id}/{item.profile}: host fingerprint mismatch")
        req = record.get("request") or {}
        if req.get("prompt") != item.prompt:
            errors.append(f"{item.prompt_id}/{item.profile}: prompt mismatch")
        if req.get("seed") != item.seed:
            errors.append(f"{item.prompt_id}/{item.profile}: seed mismatch")
        if req.get("width") != self.manifest.resolution:
            errors.append(f"{item.prompt_id}/{item.profile}: width mismatch")
        if req.get("height") != self.manifest.resolution:
            errors.append(f"{item.prompt_id}/{item.profile}: height mismatch")
        if req.get("steps") != self.manifest.steps:
            errors.append(f"{item.prompt_id}/{item.profile}: steps mismatch")
        if req.get("cfg") != self.manifest.cfg:
            errors.append(f"{item.prompt_id}/{item.profile}: cfg mismatch")
        if req.get("threads") != self.manifest.threads:
            errors.append(f"{item.prompt_id}/{item.profile}: threads mismatch")
        img = self.root.image_path(item.prompt_id, item.profile)
        png_sha = record.get("png_sha256")
        if not img.is_file():
            errors.append(f"{item.prompt_id}/{item.profile}: PNG missing")
        elif png_sha:
            actual = sha256_file(img)
            if actual != png_sha:
                errors.append(f"{item.prompt_id}/{item.profile}: PNG sha mismatch")
        return errors

    def _is_resume_safe(self, item: RunPlanItem) -> bool:
        path = self.root.record_path(item.prompt_id, item.profile)
        if not path.is_file():
            return False
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return not self._validate_run_item_for_resume(item, record)

    def plan(self) -> dict:
        return plan_to_dict(self.run_plan)

    def _run_single(self, item: RunPlanItem, sd_cli: Path, identity) -> dict:
        mem_before = read_mem_available_kb()
        start = time.monotonic()
        manifest_stub = type(
            "ManifestStub",
            (),
            {
                "diffusion": type("C", (), {"path": "diffusion-path.gguf"})(),
                "text_encoder": type("C", (), {"path": "text-encoder-path.gguf"})(),
                "vae": type("C", (), {"path": "vae-path.safetensors"})(),
            },
        )()
        result = run_generation(
            sd_cli,
            manifest_stub,
            BackendSpec(backend=CPU_BACKEND),
            prompt=item.prompt,
            seed=item.seed,
            width=self.manifest.resolution,
            height=self.manifest.resolution,
            steps=self.manifest.steps,
            cfg_scale=self.manifest.cfg,
            threads=self.manifest.threads,
            output_dir=self.root.images_dir / item.prompt_id,
            runs_dir=self.work_root / ".runs",
            client_request_id=f"vis-{item.prompt_id}-{item.profile}",
            fake=self.fake,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        mem_after = read_mem_available_kb()
        generated = (
            self.root.images_dir / item.prompt_id / result.artifact.filename
        )
        png = self.root.image_path(item.prompt_id, item.profile)
        png.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(generated, png)
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": self.manifest.benchmark_id,
            "manifest_path": str(self.manifest.path),
            "manifest_sha256": self.manifest.sha256,
            "source_head": self.manifest.source_head,
            "host_fingerprint": self.fingerprint.to_dict(),
            "prompt_id": item.prompt_id,
            "prompt_class": item.prompt_class,
            "profile": item.profile,
            "execution_index": item.execution_index,
            "request": {
                "prompt": item.prompt,
                "seed": item.seed,
                "width": self.manifest.resolution,
                "height": self.manifest.resolution,
                "steps": self.manifest.steps,
                "cfg": self.manifest.cfg,
                "threads": self.manifest.threads,
                "backend": CPU_BACKEND,
            },
            "runtime": {
                "commit": RUNTIME_COMMIT,
                "sha256": RUNTIME_SHA256,
                "path": str(sd_cli),
            },
            "models": {
                "diffusion": {
                    "sha256": (
                        Q8_DIFFUSION_SHA256
                        if item.profile == Q8_PROFILE
                        else BF16_DIFFUSION_SHA256
                    ),
                },
                "text_encoder": {"sha256": SHARED_TEXT_ENCODER_SHA256},
                "vae": {"sha256": SHARED_VAE_SHA256},
            },
            "elapsed_ms": elapsed_ms,
            "memory": {
                "mem_available_before_run_kb": mem_before,
                "minimum_mem_available_kb": result.minimum_mem_available_kb,
                "mem_available_after_run_kb": mem_after,
                "peak_sd_cli_rss_kb": result.peak_sd_cli_rss_kb,
            },
            "png_path": str(png),
            "png_sha256": sha256_file(png),
            "status": "passed",
        }

    def run(self) -> dict:
        self.validate()
        self._ensure_fingerprint_file()
        sd_cli = self._resolve_runtime()
        identity = self._verify_runtime(sd_cli)
        records = []
        errors = []
        for item in self.run_plan:
            if self._is_resume_safe(item):
                record = json.loads(
                    self.root.record_path(item.prompt_id, item.profile).read_text(
                        encoding="utf-8"
                    )
                )
                records.append(record)
                continue
            try:
                record = self._run_single(item, sd_cli, identity)
            except Exception as exc:
                record = self._failed_record(item, exc)
                errors.append(f"{item.prompt_id}/{item.profile}: {exc}")
            atomic_write_json(
                self.root.record_path(item.prompt_id, item.profile), record
            )
            records.append(record)
        aggregate = self.rebuild_aggregate(records)
        atomic_write_json(self.root.root / "aggregate.json", aggregate)
        return aggregate

    def _failed_record(self, item: RunPlanItem, exc: Exception) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": self.manifest.benchmark_id,
            "manifest_sha256": self.manifest.sha256,
            "source_head": self.manifest.source_head,
            "host_fingerprint": self.fingerprint.to_dict(),
            "prompt_id": item.prompt_id,
            "prompt_class": item.prompt_class,
            "profile": item.profile,
            "execution_index": item.execution_index,
            "request": {
                "prompt": item.prompt,
                "seed": item.seed,
                "width": self.manifest.resolution,
                "height": self.manifest.resolution,
                "steps": self.manifest.steps,
                "cfg": self.manifest.cfg,
                "threads": self.manifest.threads,
                "backend": CPU_BACKEND,
            },
            "status": "failed",
            "error": str(exc),
        }

    def _ensure_fingerprint_file(self) -> None:
        meta = self.root.root / "host-fingerprint.json"
        if meta.is_file():
            existing = json.loads(meta.read_text(encoding="utf-8"))
            if existing != self.fingerprint.to_dict():
                raise VisualBenchmarkError(
                    "host fingerprint change detected; fail closed. "
                    "Start a new benchmark evidence root."
                )
            return
        atomic_write_json(meta, self.fingerprint.to_dict())

    def _resolve_runtime(self) -> Path:
        if self.fake:
            existing = self.work_root / ".runs-mock"
            existing.mkdir(parents=True, exist_ok=True)
            return existing
        hint = os.environ.get("MAGE_CPU_PREBUILT_SD_CLI") or os.environ.get("MAGE_SD_CLI")
        if not hint:
            raise VisualBenchmarkError(
                "MAGE_CPU_PREBUILT_SD_CLI is not set; a prebuilt sd-cli is required"
            )
        path = Path(hint).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise VisualBenchmarkError(f"prebuilt sd-cli is not executable: {path}")
        digest = sha256_file(path)
        if digest != RUNTIME_SHA256:
            raise VisualBenchmarkError(
                f"runtime sha256 mismatch: expected {RUNTIME_SHA256}, got {digest}"
            )
        return path.resolve()

    def _verify_runtime(self, sd_cli: Path):
        manager = RuntimeManager(self.work_root, explicit_sd_cli=str(sd_cli))
        return manager.verify(sd_cli, requested_backend=CPU_BACKEND)

    def rebuild_aggregate(self, records: list[dict]) -> dict:
        passed = [
            r
            for r in records
            if r.get("status") == "passed"
            and r.get("manifest_sha256") == self.manifest.sha256
            and r.get("source_head") == self.manifest.source_head
            and r.get("host_fingerprint") == self.fingerprint.to_dict()
        ]
        if not passed:
            raise VisualBenchmarkError("no valid passed per-run records for aggregate")
        pairs: dict[str, dict] = {}
        for rec in passed:
            pid = rec["prompt_id"]
            pairs.setdefault(pid, {})[rec["profile"]] = rec
        incomplete = [pid for pid, r in pairs.items() if len(r) != 2]
        if incomplete:
            raise VisualBenchmarkError(
                f"cannot build aggregate: incomplete pairs {incomplete}"
            )
        return {
            "benchmark_id": self.manifest.benchmark_id,
            "manifest_sha256": self.manifest.sha256,
            "source_head": self.manifest.source_head,
            "host_fingerprint": self.fingerprint.to_dict(),
            "total_pairs": len(pairs),
            "pairs": [
                {
                    "prompt_id": pid,
                    "q8": pairs[pid][Q8_PROFILE],
                    "bf16": pairs[pid][BF16_PROFILE],
                }
                for pid in sorted(pairs)
            ],
        }

    def _collect_pass_db(self) -> dict[str, dict]:
        db: dict[str, dict] = {}
        for prompt in self.manifest.prompts:
            for profile in (Q8_PROFILE, BF16_PROFILE):
                path = self.root.record_path(prompt["id"], profile)
                if not path.is_file():
                    continue
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if self._resume_valid(record):
                    db[f"{prompt['id']}/{profile}"] = record
        return db

    def _resume_valid(self, record: dict) -> bool:
        if record.get("status") != "passed":
            return False
        if record.get("manifest_sha256") != self.manifest.sha256:
            return False
        if record.get("source_head") != self.manifest.source_head:
            return False
        if record.get("host_fingerprint") != self.fingerprint.to_dict():
            return False
        img = Path(record.get("png_path") or "")
        if not img.is_file():
            return False
        if record.get("png_sha256") and sha256_file(img) != record["png_sha256"]:
            return False
        return True

    def finalize_blind_package(self, output: Path) -> dict:
        db = self._collect_pass_db()
        if len(db) != 20:
            raise VisualBenchmarkError(f"only {len(db)} valid passed runs; need 20")
        public_map: dict[str, dict] = {}
        private_map: dict[str, dict] = {}
        blind_dir = output / "blind"
        for prompt in sorted(self.manifest.prompts, key=lambda p: p["id"]):
            q8 = db[f"{prompt['id']}/{Q8_PROFILE}"]
            bf16 = db[f"{prompt['id']}/{BF16_PROFILE}"]
            pair_id = prompt["id"]
            a_label = f"{pair_id}-A.png"
            b_label = f"{pair_id}-B.png"
            (blind_dir / pair_id).mkdir(parents=True, exist_ok=True)
            Path(blind_dir / pair_id / a_label).write_bytes(
                Path(q8["png_path"]).read_bytes()
            )
            Path(blind_dir / pair_id / b_label).write_bytes(
                Path(bf16["png_path"]).read_bytes()
            )
            public_map[pair_id] = {
                "a": a_label,
                "b": b_label,
                "prompt": prompt["prompt"],
                "seed": prompt["seed"],
            }
            private_map[pair_id] = {
                "A": Q8_PROFILE,
                "B": BF16_PROFILE,
            }
        private_bytes = json.dumps(
            private_map, indent=2, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
        commitment = hashlib.sha256(private_bytes).hexdigest()
        atomic_write_json(output / "public-blind-map.json", public_map)
        (output / "private-truth-map.json.sha256").write_text(
            commitment + "\n", encoding="utf-8"
        )
        atomic_write_json(
            output / "score-template.json",
            {
                "scale": [1, 2, 3, 4, 5],
                "dimensions": [
                    "prompt_adherence",
                    "anatomy_coherence",
                    "edge_detail_preservation",
                    "texture_stability",
                    "text_rendering_when_applicable",
                    "global_visual_preference",
                ],
                "explicit_tie_field": True,
            },
        )
        return {
            "public_map": str(output / "public-blind-map.json"),
            "private_truth_map_commitment": commitment,
            "truth_map_sha256_file": str(output / "private-truth-map.json.sha256"),
            "blind_dir": str(blind_dir),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mageflow-visual-benchmark")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--work-root", default="/kaggle/working/mageflow-visual-benchmark-768")
    parser.add_argument("--fake", action="store_true", help="mock generation (tests only)")
    parser.add_argument("--output", default=None)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--validate-only", action="store_true")
    actions.add_argument("--plan", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--finalize-blind-package", action="store_true")
    args = parser.parse_args(argv)

    bench = VisualBenchmark(
        args.manifest,
        repo_dir=Path(args.repo_dir),
        work_root=Path(args.work_root),
        fake=args.fake,
    )
    if args.validate_only:
        bench.validate()
        print("VALIDATE=PASS")
        return 0
    if args.plan:
        bench.validate()
        print(json.dumps(bench.plan(), indent=2))
        return 0
    if args.run:
        bench.validate()
        if not args.fake:
            print("RUN requires --fake in this directive; refusing real generation")
            return 2
        aggregate = bench.run()
        print(json.dumps(aggregate, indent=2))
        return 0
    if args.finalize_blind_package:
        if not args.output:
            print("--output required for --finalize-blind-package")
            return 2
        result = bench.finalize_blind_package(Path(args.output))
        print(json.dumps(result, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
