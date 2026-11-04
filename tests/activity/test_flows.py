import json
from tools.aic_activity.flows import annotate

def test_device_phase_follows_enqueue_across_async_delay(tmp_path):
    events=[dict(ph='X',name='AIC_ACTIVITY::decode',tid=1,ts=0,dur=10),
            dict(ph='s',cat='async_task_queue',id=8,tid=1,ts=5),
            dict(ph='X',cat='dequeue',tid=2,ts=20,dur=10,args={'correlation_id':8}),
            dict(ph='s',cat='HostToDevice',id=9,tid=2,ts=25),
            dict(ph='f',cat='HostToDevice',id=9,tid=3,ts=50),
            dict(ph='X',tid=3,ts=50,dur=4,args={'Task Type':'AI_CORE','Physic Stream Id':3,'Task Id':4})]
    p=tmp_path/'trace.json';p.write_text(json.dumps(events))
    rows=[dict(stream='3',task_id='4',start=50000,end=54000,engine='aic'),
          dict(stream='3',task_id='5',start=60000,end=61000,engine='aiv')]
    quality=annotate(rows,p)
    assert [r['phase'] for r in rows]==['decode','unclassified']
    assert quality['attributed_engine_task_fraction']==.8
