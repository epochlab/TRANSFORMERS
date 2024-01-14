#!/usr/bin/env python3

import time, contextlib, cProfile, pstats

class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled=prefix, on_exit, enabled
    def __enter__(self): self.st = time.perf_counter_ns()
    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled: print(f"{self.prefix}{self.et*1e-9:.2f}s" + (self.on_exit(self.et) if self.on_exit else ""))

class Profiling(contextlib.ContextDecorator):
    def __init__(self, enabled=True, sort='cumtime', frac=0.2): self.enabled, self.sort, self.frac=enabled, sort, frac
    def __enter__(self):
        self.pr = cProfile.Profile(timer=lambda: int(time.time()*1e9), timeunit=1e-6)
        if self.enabled: self.pr.enable()
    def __exit__(self, *exc):
        if self.enabled:
            self.pr.disable()
            pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort).print_stats(self.frac)