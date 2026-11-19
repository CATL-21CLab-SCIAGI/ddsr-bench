import pytest

from ddsr_bench.benchmarks.utils import benchmark_config


@pytest.mark.parametrize(
    ("name", "edition", "authority", "status"),
    [
        ("critpt", "v2", "official", "revised"),
        ("scicode", "v2.2", "independent", "corrected"),
        ("cmtbench", "v2", "official", "corrected"),
        ("phybench", "loca", "independent", "corrected"),
        ("prismphysics", "official", "official", "unverified"),
        ("cmphysbench", "official", "official", "unverified"),
        ("physics", "verified", "official", "verified"),
    ],
)
def test_selected_dataset_edition(
    name: str, edition: str, authority: str, status: str
) -> None:
    config = benchmark_config(name)

    assert config["edition"] == edition
    assert config["authority"] == authority
    assert config["status"] == status
