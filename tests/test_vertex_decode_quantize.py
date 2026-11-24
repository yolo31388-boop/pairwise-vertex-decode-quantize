import pytest
import graphics.vertex_decode_quantize as mod

CASES = [
    ("quantize_by_tile", (1,)),
    ("index_per_submesh", (1,)),
    ("bind_after_decode", (1,)),
    ("shared_quant_domain", (1,)),
    ("cull_after_scale", (1,)),
    ("seam_vertex_align", (1,)),
    ("decode", (1,)),
    ("bind", (1,)),
    ("cull", (1,)),
    ("draw", (1,)),
    ("validate_vertex", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
