import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from tools.aic_activity.analysis import metrics, partition

def task(engine, begin, end, name='x', phase='forward', category='compute'):
    return dict(engine=engine,start=begin,end=end,name=name,kind=engine.upper(),phase=phase,category=category)

def test_union_overlap_mixed_and_unknown_conserve_window():
    rows=[task('aiv',0,80),task('aic',20,40),task('mixed',40,60),task('unknown',80,100)]
    parts=partition(rows,0,100)
    assert [(p['start'],p['end'],p['state']) for p in parts]==[
        (0,20,'aiv_only'),(20,40,'both'),(40,60,'unknown'),(60,80,'aiv_only'),(80,100,'unknown')]
    result,_=metrics(rows,0,100)
    assert sum(result['state_ns'].values())==100
    assert result['known_coverage']==.6
    assert result['conditional_known']['aic_absent_given_aiv']==2/3

def test_phase_and_category_are_orthogonal():
    rows=[task('aiv',0,50,name='optimizer-AdamW',phase='optimizer'),
          task('other',50,100,name='hcom_allReduce',phase='optimizer',category='communication')]
    result,_=metrics(rows,0,100)
    assert result['phases']['optimizer']['task_union_ns']==100
    assert result['category_union_ns']['communication']==50

def test_window_edges_clip_without_claiming_missing_coverage():
    result,_=metrics([task('aic',0,100)],25,75)
    assert result['state_ns']==dict(aic_only=50,aiv_only=0,both=0,neither=0,unknown=0)

