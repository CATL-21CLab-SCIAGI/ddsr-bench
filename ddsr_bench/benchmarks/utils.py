from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Resources:
    """Container limits shared by benchmark task compilers."""

    image: str
    cpus: int
    memory_mb: int
    timeout_sec: float
