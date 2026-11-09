import pytest

from npu_nvme.experiments.timeline import overlap


def test_overlap_unions_tasks_and_clips_to_step():
    result=overlap([dict(engine='cube',start=-1,end=6),dict(engine='cube',start=2,end=8),
                    dict(engine='vector',start=4,end=9)],step_start=0,step_end=10,ready_time=3)
    assert result['durations']==dict(cube_only=4,both=4,vector_only=1,neither=1)
    assert result['candidate_count']==1
    assert result['candidate_windows'][0]['ready_duration']==1
    assert sum(result['fractions'].values())==pytest.approx(1)
    assert result['core_occupancy'] is None
