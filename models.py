"""Load / unload models on demand and track the memory each one uses."""

import gc
import threading
from dataclasses import dataclass
from typing import Any, Callable

import psutil
import torch

from tajweed import MUAALEM_MODEL, TajweedEngine
from transcriber import AZURE_MODEL, WHISPER_MODELS, AzureEngine, WhisperEngine


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], Any]
    local: bool  # runs on this machine (uses VRAM / RAM)
    approx_gb: float  # fp16 weights, for display before loading


SPECS = {
    s.name: s
    for s in [
        *(ModelSpec(n, lambda n=n: WhisperEngine(n), True, gb) for n, gb in zip(WHISPER_MODELS, (0.15, 0.08))),
        ModelSpec(MUAALEM_MODEL, TajweedEngine, True, 1.3),
        ModelSpec(AZURE_MODEL, AzureEngine, False, 0),
    ]
}


class ModelManager:
    def __init__(self):
        self._engines: dict[str, Any] = {}
        self._mem: dict[str, int] = {}
        self._lock = threading.Lock()
        self.loading: set[str] = set()
        self.errors: dict[str, str] = {}

    def get(self, name: str):
        return self._engines.get(name)

    def is_loaded(self, name: str) -> bool:
        return name in self._engines

    def mem_bytes(self, name: str) -> int:
        return self._mem.get(name, 0)

    def load(self, name: str):
        with self._lock:
            if name in self._engines:
                return self._engines[name]
            self.loading.add(name)
            try:
                before = _used_by_models()
                engine = SPECS[name].factory()
                self._mem[name] = max(0, _used_by_models() - before)
                self._engines[name] = engine
                self.errors.pop(name, None)
                return engine
            finally:
                self.loading.discard(name)

    def preload(self, names: list[str]):
        """Load models one after another in the background (server start-up)."""

        def run():
            for name in names:
                try:
                    self.load(name)
                except Exception as e:
                    self.errors[name] = str(e)

        self.loading.update(n for n in names if n not in self._engines)
        threading.Thread(target=run, daemon=True, name="preload-models").start()

    def unload(self, name: str):
        with self._lock:
            self._engines.pop(name, None)
            self._mem.pop(name, None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _used_by_models() -> int:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return psutil.Process().memory_info().rss


def memory() -> tuple[str, int, int]:
    """(label, used, total) bytes for the GPU if there is one, else system RAM."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return "VRAM", total - free, total
    vm = psutil.virtual_memory()
    return "RAM", vm.total - vm.available, vm.total
