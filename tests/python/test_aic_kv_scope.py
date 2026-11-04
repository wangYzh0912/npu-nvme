from tools.aic_kv.scope_report import classify
from tools.aic_kv.corun_evidence import overlap
from tools.aic_kv.scope import full_cache_slice


def test_zero_slice_does_not_prove_safe_strided_backend():
    row=dict(slice_count_per_generate=0,k_contiguous=False,v_contiguous=False)
    assert classify(row)=='no_slice_observed_requires_path_evidence'


def test_dense_cache_is_negative_only_when_device_trace_has_no_slice():
    row=dict(slice_count_per_generate=2,k_contiguous=True,v_contiguous=True)
    assert classify(row)=='observed_full_cache_slice'
    row['slice_count_per_generate']=0
    assert classify(row)=='no_slice_with_dense_cache'


def test_task_overlap_unions_each_engine_before_intersection():
    assert overlap([(0,10),(5,20)],[(8,12)])==4


def test_task_overlap_does_not_count_adjacent_envelopes():
    assert overlap([(0,10)],[(10,20)])==0


def test_small_same_named_slice_is_not_full_cache_conversion():
    expected={1654*128*2*128}
    task={'Name':'aclnnContiguous_SliceAiCore_Slice','Output Data Types':'DT_BF16','Output Shapes':'"1,128,256"'}
    assert not full_cache_slice(task,expected)
    task['Output Shapes']='"1654,128,256"'
    assert full_cache_slice(task,expected)
