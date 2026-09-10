from __future__ import annotations

from ddsr_bench.training.schemas import SftSample, Trajectory


def native_samples(trajectory: Trajectory) -> tuple[SftSample, ...]:
    """Convert recorded model calls into visible-content SFT samples.

    Fixed stages and samples derived after inference are not model calls, so
    they are excluded. Provider reasoning stays in the canonical trajectory;
    only the response content visible to users becomes the SFT completion.
    """
    samples = []
    for generation in trajectory.generations:
        if generation.source != "model" or generation.completion is None:
            continue
        completion = {
            "role": generation.completion["role"],
            "content": generation.completion["content"],
        }
        samples.append(
            SftSample(
                id=f"{trajectory.id}:{generation.id}",
                prompt=generation.prompt,
                completion=(completion,),
                metadata={
                    "trajectory_id": trajectory.id,
                    "benchmark": trajectory.benchmark,
                    "problem_id": trajectory.problem_id,
                    "stage": generation.id,
                    "derived": False,
                    "teacher": trajectory.teacher,
                    "quality": generation.quality or trajectory.quality,
                },
            )
        )
    return tuple(samples)
