import numpy as np
import pytest

from npu_nvme.experiments.fidelity import Ages, fragment_error, reduce_error


def test_age_only_advances_selected_blocks():
    ages=Ages([('w',0),('w',1)])
    assert ages.advance(1,[('w',0)])['maximum']==1
    assert ages.advance(2,[('w',0)])['maximum']==2
    assert ages.advance(3,[('w',1)])['maximum']==1
    with pytest.raises(ValueError):ages.advance(5,[])


def test_owned_fragment_error_avoids_replicate_double_count():
    a={'w':np.array([1.,2.,3.,4.])};b={'w':np.array([0.,2.,3.,4.])}
    rows=[dict(name='w',local_element_offset=0,element_count=2)]
    result=reduce_error([fragment_error(b,a,rows)])
    assert result['relative_l2']==pytest.approx(1/np.sqrt(5))


def test_layer_error_aggregates_squared_norms():
    value=reduce_error([{'decoder.layers.3.a':[1.,4.],'decoder.layers.3.b':[3.,12.],
                         'decoder.layers.4.a':[0.,9.]}])
    assert value['layers']['layer.3']==pytest.approx(.5)
    assert value['worst_layer']=='layer.3'
