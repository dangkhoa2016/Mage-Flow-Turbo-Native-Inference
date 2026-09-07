from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from mageflow_native.inference.runner import run_generation
from mageflow_native.models.manifest import (
    ModelManifest,
    load_manifest,
    sha256_file,
    verify_manifest,
)
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

_DEFAULT_INPUT_ROOT = "/kaggle/input"

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
        "model_input_policy",
        "blinding_contract",
    }
    missing = required_top - set(data.keys())
    if missing:
        raise VisualBenchmarkError(f"manifest missing fields: {sorted(missing)}")

    model_input_policy = data.get("model_input_policy") or {}
    if model_input_policy.get("actual_bytes_must_be_sha256_verified") is not True:
        raise VisualBenchmarkError(
            "model_input_policy.actual_bytes_must_be_sha256_verified must be true"
        )
    if model_input_policy.get("all_profiles_must_preflight_before_generation") is not True:
        raise VisualBenchmarkError(
            "model_input_policy.all_profiles_must_preflight_before_generation must be true"
        )
    if model_input_policy.get("download_during_benchmark") is not False:
        raise VisualBenchmarkError(
            "model_input_policy.download_during_benchmark must be false"
        )

    blinding_contract = data.get("blinding_contract") or {}
    if blinding_contract.get("assignment") != "os-random-per-pair":
        raise VisualBenchmarkError(
            "blinding_contract.assignment must be os-random-per-pair"
        )
    if blinding_contract.get("truth_map_commitment_sha256_before_scoring") is not True:
        raise VisualBenchmarkError(
            "blinding_contract.truth_map_commitment_sha256_before_scoring must be true"
        )
    if blinding_contract.get("public_map_must_not_contain_profile_identity") is not True:
        raise VisualBenchmarkError(
            "blinding_contract.public_map_must_not_contain_profile_identity must be true"
        )

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


@dataclass(frozen=True)
class AttemptSpec:
    attempt_id: str
    run_dir: Path
    previous_attempts: list[str]


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


@dataclass(frozen=True)
class PreparedProfileAssets:
    profile: str
    manifest_path: Path
    manifest: ModelManifest
    verified_paths: dict[str, Path]


def verify_truth_map_commitment(
    truth_map_path: Path,
    commitment_path: Path,
) -> bool:
    actual_bytes = truth_map_path.read_bytes()
    actual_hash = hashlib.sha256(actual_bytes).hexdigest()
    expected = commitment_path.read_text(encoding="utf-8").strip()
    return actual_hash == expected


class VisualBenchmark:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        repo_dir: Path,
        work_root: Path,
        input_root: str | Path | None = None,
        compile_manifests: bool = False,
        fake: bool = False,
    ) -> None:
        self.manifest = load_benchmark_manifest(manifest_path)
        self.repo_dir = repo_dir
        self.work_root = work_root
        self.root = BenchmarkRoot(work_root / "evidence")
        self.input_root = Path(
            input_root
            or os.environ.get("MAGE_VISUAL_BENCHMARK_INPUT_ROOT", _DEFAULT_INPUT_ROOT)
        )
        self.compile_manifests = compile_manifests
        self.fake = fake
        self.fingerprint = HostFingerprint.current()
        self.run_plan = build_run_plan(self.manifest)
        self._run_item_index = {
            (i.prompt_id, i.profile): i for i in self.run_plan
        }

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

    def _validate_run_item_contract(self, item: RunPlanItem, record: dict) -> list[str]:
        """Canonical per-run record contract. An empty list means the record is
        valid evidence for this exact run-plan item."""
        errors: list[str] = []
        tag = f"{item.prompt_id}/{item.profile}"
        if record.get("status") != "passed":
            errors.append(f"{tag}: record status not passed")
        if record.get("benchmark_id") != self.manifest.benchmark_id:
            errors.append(f"{tag}: benchmark_id mismatch")
        if record.get("manifest_sha256") != self.manifest.sha256:
            errors.append(f"{tag}: manifest sha mismatch")
        if record.get("source_head") != self.manifest.source_head:
            errors.append(f"{tag}: source_head mismatch")
        if record.get("host_fingerprint") != self.fingerprint.to_dict():
            errors.append(f"{tag}: host fingerprint mismatch")
        if record.get("prompt_id") != item.prompt_id:
            errors.append(f"{tag}: prompt_id mismatch")
        if record.get("prompt_class") != item.prompt_class:
            errors.append(f"{tag}: prompt_class mismatch")
        if record.get("profile") != item.profile:
            errors.append(f"{tag}: profile mismatch")
        if record.get("execution_index") != item.execution_index:
            errors.append(f"{tag}: execution_index mismatch")
        req = record.get("request") or {}
        if req.get("prompt") != item.prompt:
            errors.append(f"{tag}: prompt mismatch")
        if req.get("seed") != item.seed:
            errors.append(f"{tag}: seed mismatch")
        if req.get("width") != self.manifest.resolution:
            errors.append(f"{tag}: width mismatch")
        if req.get("height") != self.manifest.resolution:
            errors.append(f"{tag}: height mismatch")
        if req.get("steps") != self.manifest.steps:
            errors.append(f"{tag}: steps mismatch")
        if req.get("cfg") != self.manifest.cfg:
            errors.append(f"{tag}: cfg mismatch")
        if req.get("threads") != self.manifest.threads:
            errors.append(f"{tag}: threads mismatch")
        if req.get("backend") != CPU_BACKEND:
            errors.append(f"{tag}: backend mismatch")
        rec_runtime = record.get("runtime") or {}
        if rec_runtime.get("sha256") != RUNTIME_SHA256:
            errors.append(f"{tag}: runtime sha mismatch")
        if rec_runtime.get("commit") != RUNTIME_COMMIT:
            errors.append(f"{tag}: runtime commit mismatch")
        if item.profile == Q8_PROFILE:
            expected_diff = Q8_DIFFUSION_SHA256
        elif item.profile == BF16_PROFILE:
            expected_diff = BF16_DIFFUSION_SHA256
        else:
            expected_diff = None
            errors.append(f"{tag}: unknown profile {item.profile!r}")
        rec_models = record.get("models") or {}
        if rec_models.get("diffusion", {}).get("sha256") != expected_diff:
            errors.append(f"{tag}: diffusion model sha mismatch")
        if rec_models.get("text_encoder", {}).get("sha256") != SHARED_TEXT_ENCODER_SHA256:
            errors.append(f"{tag}: text encoder sha mismatch")
        if rec_models.get("vae", {}).get("sha256") != SHARED_VAE_SHA256:
            errors.append(f"{tag}: vae sha mismatch")
        expected_png = self.root.image_path(item.prompt_id, item.profile)
        if record.get("png_path") != str(expected_png):
            errors.append(f"{tag}: png_path must point at canonical PNG")
        png_sha = record.get("png_sha256")
        if not isinstance(png_sha, str) or not _HEX64.match(png_sha):
            errors.append(f"{tag}: png_sha256 missing or invalid")
        if not expected_png.is_file():
            errors.append(f"{tag}: canonical PNG missing")
        else:
            try:
                actual = sha256_file(expected_png)
                if isinstance(png_sha, str) and actual != png_sha:
                    errors.append(f"{tag}: PNG sha mismatch")
            except OSError:
                errors.append(f"{tag}: canonical PNG unreadable")
        attempt = record.get("attempt")
        if not isinstance(attempt, dict):
            errors.append(f"{tag}: attempt metadata missing")
        else:
            if not isinstance(attempt.get("attempt_id"), str) or not attempt.get("attempt_id"):
                errors.append(f"{tag}: attempt_id invalid")
            if not isinstance(attempt.get("run_dir"), str) or not attempt.get("run_dir"):
                errors.append(f"{tag}: attempt run_dir invalid")
            if not isinstance(attempt.get("previous_attempts"), list):
                errors.append(f"{tag}: attempt previous_attempts invalid")
        return errors

    def _is_resume_safe(self, item: RunPlanItem) -> bool:
        path = self.root.record_path(item.prompt_id, item.profile)
        if not path.is_file():
            return False
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return not self._validate_run_item_contract(item, record)

    def plan(self) -> dict:
        return plan_to_dict(self.run_plan)

    def _prepare_profile_assets(self, profile: str) -> PreparedProfileAssets:
        from integrations.kaggle.input_adapter import build_kaggle_manifest
        from integrations.kaggle.profiles import validate_profile_environment

        validate_profile_environment(
            profile,
            backend=CPU_BACKEND,
            mem_total_kb=_mem_total_kb(),
        )

        meta_dir = self.work_root / ".benchmark-meta" / profile
        meta_dir.mkdir(parents=True, exist_ok=True)
        manifest_json_path = meta_dir / "model-manifest.json"

        build_kaggle_manifest(
            input_root=self.input_root,
            output=manifest_json_path,
            profile=profile,
        )

        model_manifest = load_manifest(
            manifest_json_path,
            model_root=self.input_root,
        )

        verified_paths = verify_manifest(model_manifest)

        if profile == Q8_PROFILE:
            expected_diff_sha = Q8_DIFFUSION_SHA256
        else:
            expected_diff_sha = BF16_DIFFUSION_SHA256
        if model_manifest.diffusion.sha256 != expected_diff_sha:
            raise VisualBenchmarkError(
                f"{profile}: resolved diffusion SHA {model_manifest.diffusion.sha256} "
                f"does not match expected {expected_diff_sha}"
            )
        if model_manifest.text_encoder.sha256 != SHARED_TEXT_ENCODER_SHA256:
            raise VisualBenchmarkError(
                f"{profile}: resolved text encoder SHA mismatch"
            )
        if model_manifest.vae.sha256 != SHARED_VAE_SHA256:
            raise VisualBenchmarkError(
                f"{profile}: resolved VAE SHA mismatch"
            )

        return PreparedProfileAssets(
            profile=profile,
            manifest_path=manifest_json_path,
            manifest=model_manifest,
            verified_paths=verified_paths,
        )

    def _prepare_all_profile_assets(self) -> dict[str, PreparedProfileAssets]:
        prepared: dict[str, PreparedProfileAssets] = {}
        for profile in [Q8_PROFILE, BF16_PROFILE]:
            prepared[profile] = self._prepare_profile_assets(profile)
        return prepared

    def _next_attempt(self, item: RunPlanItem) -> AttemptSpec:
        prefix = f"vis-{item.prompt_id}-{item.profile}-attempt-"
        runs_dir = self.work_root / ".runs"
        existing: list[str] = []
        used: set[str] = set()
        if runs_dir.is_dir():
            for entry in runs_dir.iterdir():
                if entry.is_dir() and entry.name.startswith(prefix):
                    existing.append(entry.name)
                    used.add(entry.name)
        seq = 1
        while f"{prefix}{seq:03d}" in used:
            seq += 1
        attempt_id = f"{prefix}{seq:03d}"
        return AttemptSpec(
            attempt_id=attempt_id,
            run_dir=runs_dir / attempt_id,
            previous_attempts=sorted(existing),
        )

    def _run_single(
        self,
        item: RunPlanItem,
        sd_cli: Path,
        identity,
        *,
        prepared_assets: dict[str, PreparedProfileAssets],
        attempt: AttemptSpec | None = None,
    ) -> dict:
        if attempt is None:
            attempt = self._next_attempt(item)
        mem_before = read_mem_available_kb()
        start = time.monotonic()
        assets = prepared_assets[item.profile]
        result = run_generation(
            sd_cli,
            assets.manifest,
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
            client_request_id=attempt.attempt_id,
            fake=self.fake,
        )
        if result.request_id != attempt.attempt_id:
            raise VisualBenchmarkError(
                f"{item.prompt_id}/{item.profile}: runner request_id "
                f"{result.request_id} != allocated attempt {attempt.attempt_id}"
            )
        physical_run_dir = self.work_root / ".runs" / result.request_id
        elapsed_ms = int((time.monotonic() - start) * 1000)
        mem_after = read_mem_available_kb()
        generated = (
            self.root.images_dir / item.prompt_id / result.artifact.filename
        )
        png = self.root.image_path(item.prompt_id, item.profile)
        png.parent.mkdir(parents=True, exist_ok=True)
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
                    "path": str(assets.verified_paths["diffusion"]),
                    "sha256": assets.manifest.diffusion.sha256,
                },
                "text_encoder": {
                    "path": str(assets.verified_paths["text_encoder"]),
                    "sha256": assets.manifest.text_encoder.sha256,
                },
                "vae": {
                    "path": str(assets.verified_paths["vae"]),
                    "sha256": assets.manifest.vae.sha256,
                },
            },
            "attempt": {
                "attempt_id": result.request_id,
                "run_dir": str(physical_run_dir),
                "previous_attempts": attempt.previous_attempts,
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
        prepared_assets = self._prepare_all_profile_assets()
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
            attempt = self._next_attempt(item)
            try:
                record = self._run_single(
                    item,
                    sd_cli,
                    identity,
                    prepared_assets=prepared_assets,
                    attempt=attempt,
                )
            except Exception as exc:
                record = self._failed_record(item, exc, attempt=attempt)
                errors.append(f"{item.prompt_id}/{item.profile}: {exc}")
            atomic_write_json(
                self.root.record_path(item.prompt_id, item.profile), record
            )
            records.append(record)
        aggregate = self.rebuild_aggregate(records)
        atomic_write_json(self.root.root / "aggregate.json", aggregate)
        return aggregate

    def _failed_record(
        self,
        item: RunPlanItem,
        exc: Exception,
        *,
        attempt: AttemptSpec | None = None,
    ) -> dict:
        record = {
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
        if attempt is not None:
            record["attempt"] = {
                "attempt_id": attempt.attempt_id,
                "run_dir": str(attempt.run_dir),
                "previous_attempts": attempt.previous_attempts,
            }
        return record

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
            rec
            for rec in records
            if (item := self._run_item_index.get((rec.get("prompt_id"), rec.get("profile"))))
            is not None
            and not self._validate_run_item_contract(item, rec)
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
                item = self._run_item_index.get((prompt["id"], profile))
                if item is None:
                    continue
                path = self.root.record_path(prompt["id"], profile)
                if not path.is_file():
                    continue
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if not self._validate_run_item_contract(item, record):
                    db[f"{prompt['id']}/{profile}"] = record
        return db

    def _assign_blind_ab(self) -> dict[str, dict[str, str]]:
        assignment: dict[str, dict[str, str]] = {}
        for prompt in sorted(self.manifest.prompts, key=lambda p: p["id"]):
            swap = bool(secrets.randbits(1))
            if swap:
                assignment[prompt["id"]] = {"A": BF16_PROFILE, "B": Q8_PROFILE}
            else:
                assignment[prompt["id"]] = {"A": Q8_PROFILE, "B": BF16_PROFILE}
        return assignment

    def finalize_blind_package(
        self,
        output: Path,
        *,
        truth_dir: Path | None = None,
    ) -> dict:
        db = self._collect_pass_db()
        if len(db) != 20:
            raise VisualBenchmarkError(f"only {len(db)} valid passed runs; need 20")

        if truth_dir is not None:
            output = Path(output)
            truth_dir = Path(truth_dir)
            try:
                output.resolve().relative_to(truth_dir.resolve())
                raise VisualBenchmarkError(
                    "output directory must not be inside truth_dir"
                )
            except VisualBenchmarkError:
                raise
            except ValueError:
                pass
            try:
                truth_dir.resolve().relative_to(output.resolve())
                raise VisualBenchmarkError(
                    "truth_dir must not be inside output directory"
                )
            except VisualBenchmarkError:
                raise
            except ValueError:
                pass

        assignment = self._assign_blind_ab()
        public_map: dict[str, dict] = {}
        private_map: dict[str, dict] = {}
        blind_dir = output / "blind"
        for prompt in sorted(self.manifest.prompts, key=lambda p: p["id"]):
            pair_id = prompt["id"]
            q8_rec = db[f"{pair_id}/{Q8_PROFILE}"]
            bf16_rec = db[f"{pair_id}/{BF16_PROFILE}"]
            ab = assignment[pair_id]
            if ab["A"] == Q8_PROFILE:
                a_rec, b_rec = q8_rec, bf16_rec
            else:
                a_rec, b_rec = bf16_rec, q8_rec
            a_label = f"{pair_id}-A.png"
            b_label = f"{pair_id}-B.png"
            (blind_dir / pair_id).mkdir(parents=True, exist_ok=True)
            a_bytes = Path(a_rec["png_path"]).read_bytes()
            b_bytes = Path(b_rec["png_path"]).read_bytes()
            (blind_dir / pair_id / a_label).write_bytes(a_bytes)
            (blind_dir / pair_id / b_label).write_bytes(b_bytes)
            public_map[pair_id] = {
                "a": a_label,
                "b": b_label,
                "prompt": prompt["prompt"],
                "seed": prompt["seed"],
                "a_png_sha256": hashlib.sha256(a_bytes).hexdigest(),
                "b_png_sha256": hashlib.sha256(b_bytes).hexdigest(),
            }
            private_map[pair_id] = {
                "A": ab["A"],
                "B": ab["B"],
                "a_png_sha256": hashlib.sha256(a_bytes).hexdigest(),
                "b_png_sha256": hashlib.sha256(b_bytes).hexdigest(),
            }

        if truth_dir is not None:
            truth_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                truth_dir / "private-truth-map.json",
                {
                    "benchmark_id": self.manifest.benchmark_id,
                    "manifest_sha256": self.manifest.sha256,
                    "source_head": self.manifest.source_head,
                    "pairs": private_map,
                },
            )
            truth_bytes = (truth_dir / "private-truth-map.json").read_bytes()
            commitment = hashlib.sha256(truth_bytes).hexdigest()
        else:
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
            "truth_dir": str(truth_dir) if truth_dir else None,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mageflow-visual-benchmark")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--work-root", default="/kaggle/working/mageflow-visual-benchmark-768")
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--fake", action="store_true", help="mock generation (tests only)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--truth-dir", default=None)
    parser.add_argument("--truth-map", default=None)
    parser.add_argument("--commitment-file", default=None)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--validate-only", action="store_true")
    actions.add_argument("--plan", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--finalize-blind-package", action="store_true")
    actions.add_argument("--verify-truth-map", action="store_true")
    args = parser.parse_args(argv)

    bench = VisualBenchmark(
        args.manifest,
        repo_dir=Path(args.repo_dir),
        work_root=Path(args.work_root),
        input_root=args.input_root,
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
        aggregate = bench.run()
        print(json.dumps(aggregate, indent=2))
        return 0
    if args.verify_truth_map:
        if not args.truth_map or not args.commitment_file:
            print("--truth-map and --commitment-file required for --verify-truth-map")
            return 2
        ok = verify_truth_map_commitment(
            Path(args.truth_map),
            Path(args.commitment_file),
        )
        print(f"TRUTH_MAP_COMMITMENT={'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    if args.finalize_blind_package:
        if not args.output:
            print("--output required for --finalize-blind-package")
            return 2
        if not args.truth_dir:
            print("--truth-dir required for --finalize-blind-package")
            return 2
        result = bench.finalize_blind_package(
            Path(args.output),
            truth_dir=Path(args.truth_dir),
        )
        print(json.dumps(result, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
