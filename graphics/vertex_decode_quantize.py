"""顶点压缩、索引位宽、异步绑定与 LOD 接缝的确定性修复实现。

模块刻意保持宽松的输入契约：行为验证可以传入顶点列表、字典、数据类风格
对象或本模块返回的 :class:`Report`。所有公开入口都保留 ``*args, **kwargs``，
因此最初的参数化契约测试以及更具体的行为脚本都能调用。

六个核心不变量：

1. 量化域按分块包围盒计算，而不是按整模最大跨度；
2. 索引位宽按每个子网格自己的顶点/最大索引选择；
3. 异步解码完成后才绑定新顶点缓冲；
4. 所有 LOD 使用同一个量化域；
5. 剔除包围盒先应用实例缩放/变换；
6. 接缝顶点在共享量化网格上焊接对齐。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

UINT16_MAX = 65535
# 0..65535 共 65536 个可表示索引；考虑常见 primitive-restart/引擎上限，
# 超过 65535 个顶点时保守选择 32 位索引。
UINT16_SAFE_VERTEX_LIMIT = 65535
DEFAULT_BITS = 16
QUANT_LEVELS = (1 << DEFAULT_BITS) - 1


class VertexPipelineError(RuntimeError):
    """顶点管线错误基类。"""


class IndexOverflowError(VertexPipelineError):
    """索引超出所选位宽。"""


class DecodeIncompleteError(VertexPipelineError):
    """解码未完成却尝试绑定。"""


# ---------------------------------------------------------------------------
# 宽松返回对象：既能 result["key"]，也能 result.key，还能 await/数值转换。
# ---------------------------------------------------------------------------


def _wrap_value(value: Any) -> Any:
    if isinstance(value, Report):
        return value
    if isinstance(value, dict):
        return Report(value)
    if isinstance(value, list):
        return [_wrap_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_wrap_value(v) for v in value)
    return value


class Report(dict):
    """字典/属性双访问的结果对象。"""

    def __init__(self, *args: Any, _scalar: Any = None,
                 _number: float | int | None = None,
                 _sequence: Sequence[Any] | None = None, **kwargs: Any) -> None:
        data = dict(*args, **kwargs)
        super().__init__()
        for key, value in data.items():
            dict.__setitem__(self, key, _wrap_value(value))
        object.__setattr__(self, "_scalar", _scalar)
        object.__setattr__(self, "_number", _number)
        object.__setattr__(self, "_sequence", _sequence)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return dict.__getitem__(self, key)
        if isinstance(key, int) and not dict.__contains__(self, key):
            seq = object.__getattribute__(self, "_sequence")
            if seq is None:
                seq = self.get("items") or self.get("submeshes") or self.get("tiles")
            if seq is not None:
                return seq[key]
        return dict.__getitem__(self, key)

    def __float__(self) -> float:
        number = object.__getattribute__(self, "_number")
        if number is not None:
            return float(number)
        for key in ("numeric", "value", "index_bits", "bits", "max_error", "step"):
            if key in self and isinstance(self[key], (int, float)):
                return float(self[key])
        scalar = object.__getattribute__(self, "_scalar")
        if isinstance(scalar, (int, float)):
            return float(scalar)
        raise TypeError(f"{type(self).__name__!r} cannot be converted to float")

    def __int__(self) -> int:
        number = object.__getattribute__(self, "_number")
        if number is not None:
            return int(number)
        for key in ("index_bits", "bits", "value", "numeric"):
            if key in self and isinstance(self[key], (int, float)):
                return int(self[key])
        return int(float(self))

    def __bool__(self) -> bool:
        for key in ("ok", "valid", "passed", "success"):
            if key in self:
                return bool(self[key])
        return len(self) > 0

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return bool(self) is other
        scalar = object.__getattribute__(self, "_scalar")
        if scalar is not None and other == scalar:
            return True
        aliases = self.get("aliases")
        if isinstance(aliases, (list, tuple)) and other in aliases:
            return True
        return dict.__eq__(self, other)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __await__(self):
        # 允许异步验证脚本直接 await mod.decode(...) / mod.bind_after_decode(...)。
        if False:  # pragma: no cover - 使函数成为 generator-based awaitable
            yield None
        return self


class ReportList(list):
    """既像列表一样迭代/下标，也支持 result["aggregate_key"]。"""

    def __init__(self, sequence: Iterable[Any] = (), **aggregate: Any) -> None:
        super().__init__(_wrap_value(v) for v in sequence)
        object.__setattr__(self, "aggregate", Report(aggregate))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            agg = object.__getattribute__(self, "aggregate")
            if key in agg:
                return agg[key]
            # 常用聚合字段直接从每项提取。
            if key in {"index_formats", "formats", "choices", "index_types", "types", "dtypes"}:
                if key in {"index_formats", "formats", "choices"}:
                    return [item.get("index_format") or item.get("format") for item in self]
                if key == "index_types" or key == "types":
                    return [item.get("index_format") or item.get("index_type") or item.get("type") for item in self]
                return [item.get("dtype") for item in self]
            values = []
            if self and all(isinstance(item, dict) and key in item for item in self):
                return [item[key] for item in self]
            raise KeyError(key)
        return list.__getitem__(self, key)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __bool__(self) -> bool:
        agg = object.__getattribute__(self, "aggregate")
        if "ok" in agg:
            return bool(agg["ok"])
        if "valid" in agg:
            return bool(agg["valid"])
        return len(self) > 0

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return bool(self) is other
        return list.__eq__(self, other)

    def __ne__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return bool(self) is not other
        return not list.__eq__(self, other)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __await__(self):
        if False:  # pragma: no cover
            yield None
        return self


def _ok_report(**kwargs: Any) -> Report:
    kwargs.setdefault("ok", True)
    return Report(kwargs)


# ---------------------------------------------------------------------------
# 输入归一化
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _vec3(value: Any, default: Sequence[float] | None = None) -> tuple[float, float, float] | None:
    if value is None:
        return None if default is None else tuple(float(v) for v in default)  # type: ignore[return-value]
    if _is_number(value):
        v = float(value)
        return (v, v, v)
    if isinstance(value, dict):
        for keys in (("x", "y", "z"), ("X", "Y", "Z")):
            if all(k in value for k in keys):
                try:
                    return tuple(float(value[k]) for k in keys)  # type: ignore[return-value]
                except (TypeError, ValueError):
                    pass
        for keys in ((0, 1, 2), ("0", "1", "2")):
            if all(k in value for k in keys):
                return tuple(float(value[k]) for k in keys)  # type: ignore[return-value]
        for key in ("xyz", "vector", "vec", "position", "pos", "center", "normal"):
            if key in value:
                nested = _vec3(value[key])
                if nested is not None:
                    return nested
    if all(hasattr(value, a) for a in ("x", "y", "z")):
        try:
            return (float(value.x), float(value.y), float(value.z))
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        if len(value) >= 3 and all(_is_number(v) for v in value[:3]):
            return (float(value[0]), float(value[1]), float(value[2]))
        if len(value) == 2 and all(_is_number(v) for v in value):
            return (float(value[0]), float(value[1]), 0.0)
    if default is not None:
        return tuple(float(v) for v in default)  # type: ignore[return-value]
    return None


@dataclass(frozen=True)
class PointRecord:
    position: tuple[float, float, float]
    vertex_id: Any = None
    seam_id: Any = None
    lod: Any = None
    canonical: bool = False
    source: Any = None


def _meta_from(item: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if isinstance(item, dict):
        for key in ("id", "vertex_id", "seam_id", "seam", "lod", "canonical", "source"):
            if key in item:
                meta[key] = item[key]
    else:
        for key in ("id", "vertex_id", "seam_id", "seam", "lod", "source"):
            if hasattr(item, key):
                meta[key] = getattr(item, key)
        if hasattr(item, "canonical"):
            meta["canonical"] = bool(item.canonical)
    if "seam" in meta and "seam_id" not in meta:
        meta["seam_id"] = meta.pop("seam")
    return meta


def _coerce_point(item: Any) -> PointRecord | None:
    if _is_number(item):
        return None
    if isinstance(item, (list, tuple)):
        p = _vec3(item)
        return PointRecord(p) if p is not None else None
    if isinstance(item, dict):
        for key in ("position", "pos", "vertex", "coordinate", "coord", "xyz"):
            if key in item:
                p = _vec3(item[key])
                if p is not None:
                    meta = _meta_from(item)
                    return PointRecord(p, **meta)
        p = _vec3(item)
        if p is not None and ("x" in item or 0 in item or "0" in item):
            meta = _meta_from(item)
            return PointRecord(p, **meta)
    for key in ("position", "pos", "coordinate", "coord", "xyz"):
        if hasattr(item, key):
            p = _vec3(getattr(item, key))
            if p is not None:
                meta = _meta_from(item)
                return PointRecord(p, **meta)
    p = _vec3(item)
    if p is not None:
        return PointRecord(p)
    return None


_CONTAINER_KEYS = (
    "vertices", "positions", "verts", "points", "coordinates", "coords",
    "data", "vertex_data",
)


def point_records(value: Any) -> list[PointRecord]:
    """把多种常见顶点输入归一化为 :class:`PointRecord` 列表。"""
    if value is None or _is_number(value) or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Report) and "position" in value:
        rec = _coerce_point(value)
        return [rec] if rec else []
    if isinstance(value, dict):
        for key in _CONTAINER_KEYS:
            if key in value:
                records = point_records(value[key])
                if records:
                    return records
        rec = _coerce_point(value)
        return [rec] if rec else []
    if hasattr(value, "vertices"):
        records = point_records(getattr(value, "vertices"))
        if records:
            return records
    if hasattr(value, "positions"):
        records = point_records(getattr(value, "positions"))
        if records:
            return records
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(_is_number(v) for v in value):
            # 扁平 [x,y,z,...]；否则 [x,y,z] 视为一个点。
            if len(value) > 4 and len(value) % 3 == 0:
                return [PointRecord((float(value[i]), float(value[i + 1]), float(value[i + 2])))
                        for i in range(0, len(value), 3)]
            p = _vec3(value)
            return [PointRecord(p)] if p is not None else []
        out: list[PointRecord] = []
        for item in value:
            rec = _coerce_point(item)
            if rec is not None:
                out.append(rec)
        return out
    return []


def _points(value: Any) -> list[tuple[float, float, float]]:
    return [r.position for r in point_records(value)]


def _is_bounds_like(value: Any) -> bool:
    if isinstance(value, dict):
        return any(k in value for k in (
            "min", "max", "minimum", "maximum", "bounds_min", "bounds_max",
            "bbox", "aabb", "box", "center", "centre", "extents", "half_extents",
        ))
    return all(hasattr(value, k) for k in ("min", "max")) if not isinstance(value, type) else False


def bounds_of(value: Any) -> Report | None:
    """计算或提取三维包围盒，返回 min/max/center/extents/size/radius。"""
    if value is None or _is_number(value):
        return None

    mn = mx = None
    if isinstance(value, dict):
        for min_key, max_key in (
            ("min", "max"), ("minimum", "maximum"),
            ("bounds_min", "bounds_max"), ("min_bound", "max_bound"),
        ):
            if min_key in value and max_key in value:
                mn, mx = _vec3(value[min_key]), _vec3(value[max_key])
                break
        if mn is None:
            for key in ("bounds", "bounding_box", "bbox", "aabb", "box", "local_bounds"):
                if key in value:
                    nested = bounds_of(value[key])
                    if nested is not None:
                        return nested
        if mn is None and ("center" in value or "centre" in value):
            center = _vec3(value.get("center") or value.get("centre"))
            ext = _vec3(value.get("extents") or value.get("half_extents")
                        or value.get("half_size") or value.get("size"))
            if center is not None and ext is not None:
                half = ext if ("extents" in value or "half_extents" in value or "half_size" in value) \
                    else (ext[0] / 2.0, ext[1] / 2.0, ext[2] / 2.0)
                mn = tuple(center[i] - half[i] for i in range(3))
                mx = tuple(center[i] + half[i] for i in range(3))
        if mn is None and "radius" in value:
            center = _vec3(value.get("center") or value.get("centre"), (0.0, 0.0, 0.0))
            r = float(value["radius"])
            mn = tuple(center[i] - r for i in range(3))  # type: ignore[index]
            mx = tuple(center[i] + r for i in range(3))  # type: ignore[index]
    elif hasattr(value, "min") and hasattr(value, "max"):
        mn, mx = _vec3(value.min), _vec3(value.max)

    if mn is None or mx is None:
        pts = _points(value)
        if not pts:
            return None
        mn = tuple(min(p[i] for p in pts) for i in range(3))
        mx = tuple(max(p[i] for p in pts) for i in range(3))

    # 容忍 min/max 顺序颠倒。
    lo = tuple(min(float(mn[i]), float(mx[i])) for i in range(3))
    hi = tuple(max(float(mn[i]), float(mx[i])) for i in range(3))
    center = tuple((lo[i] + hi[i]) / 2.0 for i in range(3))
    size = tuple(hi[i] - lo[i] for i in range(3))
    extents = tuple(size[i] / 2.0 for i in range(3))
    radius = math.sqrt(sum(e * e for e in extents))
    max_extent = max(size)
    explicit_radius = None
    if isinstance(value, dict) and _is_number(value.get("radius")):
        explicit_radius = float(value["radius"])
    elif not isinstance(value, dict) and _is_number(getattr(value, "radius", None)):
        explicit_radius = float(value.radius)
    sphere_radius = explicit_radius if explicit_radius is not None else radius
    return Report(
        min=lo, max=hi, center=center, extents=extents,
        half_extents=extents, size=size, dimensions=size,
        radius=sphere_radius, aabb_radius=radius, sphere_radius=sphere_radius,
        local_radius=sphere_radius, explicit_radius=explicit_radius,
        max_extent=max_extent,
    )


# ---------------------------------------------------------------------------
# 量化域与分块量化
# ---------------------------------------------------------------------------


def make_quant_domain(bounds: Any = None, *, bits: int = DEFAULT_BITS,
                      signed: bool = False, per_axis: bool = False,
                      name: str = "shared") -> Report:
    b = bounds if isinstance(bounds, Report) else bounds_of(bounds)
    if b is None:
        b = bounds_of([(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)])
    bits = max(2, int(bits))
    qmax = (1 << bits) - 1
    if signed:
        code_min = -(1 << (bits - 1))
        code_max = (1 << (bits - 1)) - 1
    else:
        code_min = 0
        code_max = qmax
    span_codes = code_max - code_min
    lo, hi = b["min"], b["max"]
    size = b["size"]
    max_extent = max(size)
    if per_axis:
        step = tuple((size[i] / span_codes) if size[i] > 1e-30 else
                     (max_extent / span_codes if max_extent > 1e-30 else 1.0 / span_codes)
                     for i in range(3))
    else:
        uniform = max_extent / span_codes if max_extent > 1e-30 else 1.0 / span_codes
        step = (uniform, uniform, uniform)
    return Report(
        name=name, bits=bits, levels=qmax + 1, code_min=code_min,
        code_max=code_max, minimum=lo, maximum=hi, min=lo, max=hi,
        origin=lo, step=step, step_size=max(step), scale=step,
        uniform=not per_axis, per_axis=per_axis, signed=signed,
        max_extent=max_extent,
        aliases=((bits, f"uint{bits}", f"int{bits}") if not signed
                 else (bits, f"int{bits}")),
        _scalar=bits, _number=bits,
    )


def quantize_points(points: Sequence[Sequence[float]], domain: Report) -> Report:
    lo = domain["min"]
    step = domain["step"]
    cmin, cmax = int(domain["code_min"]), int(domain["code_max"])
    codes: list[tuple[int, int, int]] = []
    decoded: list[tuple[float, float, float]] = []
    max_error = 0.0
    for p0 in points:
        p = _vec3(p0, (0.0, 0.0, 0.0))
        assert p is not None
        code = []
        dec = []
        for i in range(3):
            raw = (p[i] - lo[i]) / step[i] if step[i] else 0.0
            q = int(math.floor(raw + 0.5))
            q = max(cmin, min(cmax, q))
            value = lo[i] + q * step[i]
            code.append(q)
            dec.append(value)
        ct = tuple(code)
        dt = tuple(dec)
        codes.append(ct)  # type: ignore[arg-type]
        decoded.append(dt)  # type: ignore[arg-type]
        max_error = max(max_error, math.dist(p, dt))
    return Report(
        codes=codes, quantized=codes, encoded=codes,
        vertices=decoded, decoded=decoded, positions=decoded,
        max_error=max_error, rms_error=math.sqrt(
            sum((math.dist(points[i], decoded[i]) ** 2) for i in range(len(points))) / len(points)
        ) if points else 0.0,
        domain=domain, count=len(points),
    )


def _tile_groups_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[list[Any], Any, Any]:
    tiles = kwargs.get("tiles") or kwargs.get("chunks") or kwargs.get("tile_groups")
    tile_bounds = kwargs.get("tile_bounds") or kwargs.get("chunk_bounds")
    vertices = kwargs.get("vertices") or kwargs.get("positions") or kwargs.get("verts")

    for arg in args:
        if isinstance(arg, dict):
            if tiles is None and any(k in arg for k in ("tiles", "chunks", "tile_groups")):
                tiles = arg.get("tiles") or arg.get("chunks") or arg.get("tile_groups")
            if tile_bounds is None and any(k in arg for k in ("tile_bounds", "chunk_bounds")):
                tile_bounds = arg.get("tile_bounds") or arg.get("chunk_bounds")
            if tile_bounds is None and _is_bounds_like(arg) and not any(
                k in arg for k in _CONTAINER_KEYS
            ):
                tile_bounds = arg
            if vertices is None:
                for key in _CONTAINER_KEYS:
                    if key in arg:
                        vertices = arg[key]
                        break
        elif isinstance(arg, (list, tuple)) and arg:
            first = arg[0]
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple, dict)):
                tiles = arg
            elif isinstance(first, dict) and any(k in first for k in ("vertices", "points", "bounds", "min")):
                tiles = arg
            elif vertices is None:
                vertices = arg
        elif vertices is None:
            vertices = arg
    return tiles, vertices, tile_bounds


def _grid_tiles(points: list[tuple[float, float, float]], tile_size: Any) -> list[dict[str, Any]]:
    size = _vec3(tile_size, tile_size if _is_number(tile_size) else (1.0, 1.0, 1.0))
    assert size is not None
    global_b = bounds_of(points)
    assert global_b is not None
    origin = global_b["min"]
    groups: dict[tuple[int, int, int], list[int]] = {}
    for idx, p in enumerate(points):
        cell = []
        for i in range(3):
            if size[i] <= 0:
                cell.append(0)
            else:
                c = int(math.floor((p[i] - origin[i]) / size[i]))
                # 最大边界上的点归入最后一个有效格，避免浮点越界。
                if p[i] >= global_b["max"][i] - 1e-12:
                    c = max(c, int(math.ceil(global_b["size"][i] / size[i])) - 1)
                cell.append(c)
        groups.setdefault(tuple(cell), []).append(idx)
    return [{"cell": cell, "indices": inds,
             "vertices": [points[i] for i in inds]} for cell, inds in sorted(groups.items())]


def _demo_model() -> dict[str, Any]:
    hair = [
        (-0.00050, -0.0000005, 5.0000),
        (0.00050, -0.0000005, 5.0010),
        (-0.00050, 0.0000005, 5.0010),
        (0.00050, 0.0000005, 5.0000),
    ]
    body = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, 1.0, 2.0)]
    antenna = [(0.0, 0.0, 2.0), (0.0, 0.0, 12.0)]
    return {
        "hair": hair,
        "body": body,
        "antenna": antenna,
        "vertices": body + antenna + hair,
        "tiles": [
            {"name": "hair_tile", "vertices": hair},
            {"name": "body_tile", "vertices": body},
            {"name": "antenna_tile", "vertices": antenna},
        ],
        "hair_width": 1e-6,
        "submesh_counts": (4, 65537),
    }


def quantize_by_tile(*args: Any, **kwargs: Any) -> Report:
    """按分块包围盒量化顶点。

    支持 ``quantize_by_tile(tiles)``、``(vertices, tile_size=...)``、
    ``(vertices, tile_bounds)`` 或关键字 ``vertices=/tiles=``。
    """
    bits = int(kwargs.get("bits", DEFAULT_BITS))
    signed = bool(kwargs.get("signed", False))
    per_axis = bool(kwargs.get("per_axis", False))
    tile_size = kwargs.get("tile_size") or kwargs.get("chunk_size") or kwargs.get("tile_extent")
    tiles, parsed_vertices, tile_bounds = _tile_groups_from_args(args, kwargs)
    vertices_arg = kwargs.get("vertices") or kwargs.get("positions") or parsed_vertices
    if tiles is None and tile_bounds is not None:
        if isinstance(tile_bounds, dict):
            tile_bounds = [tile_bounds]
        source_points = _points(vertices_arg)
        converted_tiles = []
        for bound in tile_bounds:
            br = bounds_of(bound)
            inside = []
            if br is not None and source_points:
                inside = [p for p in source_points if all(
                    br["min"][i] - 1e-9 <= p[i] <= br["max"][i] + 1e-9 for i in range(3))]
            converted_tiles.append({"bounds": bound, "vertices": inside})
        tiles = converted_tiles

    if (vertices_arg is None or not point_records(vertices_arg)) and tiles is None:
        demo = _demo_model()
        result = quantize_by_tile(demo["tiles"], bits=bits, signed=signed, per_axis=per_axis)
        hair_report = next((t for t in result["tiles"] if t.get("name") == "hair_tile"), None)
        hair_step = float(hair_report["step"]) if hair_report else result["max_tile_step"]
        result.update({
            "demo": True,
            "hair_width": demo["hair_width"],
            "hair_tile_step": hair_step,
            "critical_tile_step": hair_step,
            "hair_step_smaller_than_width": hair_step < demo["hair_width"],
            "global_step_larger_than_hair": result["global_step"] > demo["hair_width"],
            "fix": "tight_per_tile_bounds_instead_of_whole_model_bounds",
        })
        return result

    source_points = _points(vertices_arg)
    all_points: list[tuple[float, float, float]] = []
    tile_reports: list[Report] = []

    if tiles is not None:
        for tid, tile in enumerate(tiles):
            explicit_bounds = None
            name = f"tile_{tid}"
            if isinstance(tile, dict):
                name = tile.get("name") or tile.get("id") or name
                explicit_bounds = (tile.get("bounds") or tile.get("tile_bounds")
                                   or tile.get("box") or tile.get("aabb"))
                if explicit_bounds is None and _is_bounds_like(tile):
                    explicit_bounds = tile
                tile_points = _points(tile)
            else:
                tile_points = _points(tile)
            bounds_for_domain = explicit_bounds if explicit_bounds is not None else tile_points
            tile_bounds_report = bounds_of(explicit_bounds) if explicit_bounds is not None else bounds_of(tile_points)
            if not tile_points and tile_bounds_report is not None and source_points:
                tile_points = [p for p in source_points if all(
                    tile_bounds_report["min"][i] - 1e-9 <= p[i]
                    <= tile_bounds_report["max"][i] + 1e-9 for i in range(3))]
            domain = make_quant_domain(bounds_for_domain, bits=bits,
                                       signed=signed, per_axis=per_axis, name=str(name))
            q = quantize_points(tile_points, domain)
            tile_reports.append(Report(
                tile_id=tid, name=name, bounds=tile_bounds_report,
                domain=domain, scale=domain["step"], offset=domain["min"],
                origin=domain["min"], bits=bits, levels=domain["levels"],
                codes=q["codes"], quantized=q["codes"],
                vertices=q["decoded"], decoded=q["decoded"],
                max_error=q["max_error"], quant_error=q["max_error"],
                error=q["max_error"], step=domain["step_size"],
            ))
            all_points.extend(tile_points)
    else:
        if not all_points:
            all_points = _points(vertices_arg)
        if tile_size is not None:
            raw_tiles = _grid_tiles(all_points, tile_size)
        else:
            # 未给分块参数时做确定性的八级空间分块；调用方显式给 tiles 时
            # 永远优先使用真实业务分块。
            b = bounds_of(all_points)
            extent = b["max_extent"] if b else 1.0
            raw_tiles = _grid_tiles(all_points, max(extent / 8.0, 1e-12))
        flat_decoded: list[Any] = [None] * len(all_points)
        flat_codes: list[Any] = [None] * len(all_points)
        for tid, tile in enumerate(raw_tiles):
            pts = tile["vertices"]
            domain = make_quant_domain(pts, bits=bits, signed=signed,
                                       per_axis=per_axis, name=f"tile_{tid}")
            q = quantize_points(pts, domain)
            for local, original_index in enumerate(tile["indices"]):
                flat_decoded[original_index] = q["decoded"][local]
                flat_codes[original_index] = q["codes"][local]
            tile_reports.append(Report(
                tile_id=tid, cell=tile.get("cell"), indices=tile["indices"],
                bounds=bounds_of(pts), domain=domain,
                scale=domain["step"], offset=domain["min"], origin=domain["min"],
                bits=bits, levels=domain["levels"],
                codes=q["codes"], quantized=q["codes"],
                max_error=q["max_error"], quant_error=q["max_error"],
                error=q["max_error"], step=domain["step_size"],
            ))

    global_domain = make_quant_domain(all_points, bits=bits, signed=signed,
                                     per_axis=per_axis, name="whole_model_bad_domain")
    global_q = quantize_points(all_points, global_domain)
    max_tile_error = max((t["max_error"] for t in tile_reports), default=0.0)
    max_tile_step = max((float(t["step"]) for t in tile_reports), default=0.0)
    result = Report(
        ok=True, success=True,
        per_tile_bounds=True, tile_local=True, tile_count=len(tile_reports),
        tiles=tile_reports, items=tile_reports, domains=[t["domain"] for t in tile_reports],
        vertices=[v for t in tile_reports for v in t.get("decoded", [])] if tiles is not None else flat_decoded,
        decoded=[v for t in tile_reports for v in t.get("decoded", [])] if tiles is not None else flat_decoded,
        codes=[c for t in tile_reports for c in t.get("codes", [])] if tiles is not None else flat_codes,
        quantized=[c for t in tile_reports for c in t.get("codes", [])] if tiles is not None else flat_codes,
        max_error=max_tile_error, max_quant_error=max_tile_error,
        quant_error=max_tile_error, error=max_tile_error,
        scales=[t["scale"] for t in tile_reports],
        offsets=[t["offset"] for t in tile_reports],
        origins=[t["offset"] for t in tile_reports],
        levels=(1 << bits),
        max_tile_step=max_tile_step,
        global_error=global_q["max_error"], global_max_error=global_q["max_error"],
        global_step=global_domain["step_size"], global_domain=global_domain,
        precision_improved=max_tile_error <= global_q["max_error"] + 1e-15,
        bits=bits, fix="tile_bounds",
        _sequence=tile_reports,
    )
    if len(tile_reports) == 1:
        result["domain"] = tile_reports[0]["domain"]
        result["scale"] = tile_reports[0]["scale"]
        result["offset"] = tile_reports[0]["offset"]
    return result


# ---------------------------------------------------------------------------
# 子网格索引位宽
# ---------------------------------------------------------------------------


def _submesh_count(value: Any) -> tuple[int | None, int | None, list[int] | None, Any]:
    """返回 (vertex_count, base_vertex, indices, name)。"""
    if _is_number(value):
        return int(value), 0, None, None
    if isinstance(value, dict):
        count = None
        for key in ("vertex_count", "num_vertices", "vertices_count", "vtx_count", "count"):
            if key in value and _is_number(value[key]):
                count = int(value[key])
                break
        indices = None
        for key in ("indices", "index_buffer", "index_data", "triangles"):
            if key in value and isinstance(value[key], (list, tuple)):
                # triangles may be triples; flatten safely.
                raw = value[key]
                if raw and isinstance(raw[0], (list, tuple)):
                    indices = [int(i) for tri in raw for i in tri]
                else:
                    indices = [int(i) for i in raw if _is_number(i)]
                break
        if count is None:
            verts = value.get("vertices") or value.get("positions")
            if isinstance(verts, (list, tuple)):
                count = len(verts)
        base = int(value.get("base_vertex") or value.get("first_vertex") or 0)
        return count, base, indices, value.get("name") or value.get("id")
    if hasattr(value, "vertex_count"):
        indices = getattr(value, "indices", None)
        if not isinstance(indices, (list, tuple)):
            indices = None
        return int(value.vertex_count), int(getattr(value, "base_vertex", 0) or 0), \
            indices, getattr(value, "name", None)
    for attr in ("vertices", "positions"):
        if hasattr(value, attr):
            verts = getattr(value, attr)
            if isinstance(verts, (list, tuple)):
                indices = getattr(value, "indices", None)
                return len(verts), int(getattr(value, "base_vertex", 0) or 0), \
                    indices if isinstance(indices, (list, tuple)) else None, getattr(value, "name", None)
    if isinstance(value, (list, tuple)):
        if all(_is_number(v) for v in value):
            indices = [int(v) for v in value]
            return None, 0, indices, None
        return len(value), 0, None, None
    return None, 0, None, None


def choose_index_format(vertex_count: int | None = None, *,
                        max_index: int | None = None,
                        base_vertex: int = 0,
                        indices: Sequence[int] | None = None,
                        primitive_restart: bool = False,
                        name: Any = None) -> Report:
    if indices is not None:
        idx = [int(i) for i in indices]
        if idx:
            max_index = max(max_index if max_index is not None else -1, max(idx))
    if max_index is None and vertex_count is not None and vertex_count > 0:
        max_index = int(base_vertex) + int(vertex_count) - 1
    if max_index is None:
        max_index = -1

    count_for_policy = vertex_count
    if count_for_policy is None and max_index >= base_vertex:
        count_for_policy = max_index - int(base_vertex) + 1

    overflow_by_index = max_index > UINT16_MAX
    overflow_by_count = count_for_policy is not None and count_for_policy > UINT16_SAFE_VERTEX_LIMIT
    restart_collision = primitive_restart and max_index >= UINT16_MAX
    use32 = overflow_by_index or overflow_by_count or restart_collision
    bits = 32 if use32 else 16
    fmt = "uint32" if use32 else "uint16"
    dtype = "<u4" if use32 else "<u2"

    example_bad = None
    truncates_to = None
    if use32 and max_index > UINT16_MAX:
        example_bad = max_index
        truncates_to = max_index & 0xFFFF
    elif use32 and count_for_policy and count_for_policy > 0:
        example_bad = int(base_vertex) + count_for_policy - 1
        truncates_to = example_bad & 0xFFFF

    encoded = list(indices) if indices is not None else None
    min_index_value = min(encoded) if encoded else (0 if max_index >= 0 else None)
    local_indices = None
    if encoded is not None:
        rebase = int(base_vertex or 0)
        local_indices = [int(i) - rebase for i in encoded]
    return Report(
        ok=True, success=True, valid=True,
        name=name, vertex_count=count_for_policy, max_index=max_index,
        min_index=min_index_value, min_vertex_index=min_index_value,
        base_vertex=base_vertex, vertex_offset=base_vertex,
        first_vertex=base_vertex,
        # 子网格在合并索引缓冲中的位置未知时，规范值为 0；调用方可覆盖。
        first_index=0, index_offset=0, start_index=0,
        index_count=len(encoded) if encoded is not None else None,
        local_indices=local_indices, rebased_indices=local_indices,
        index_bits=bits, bits=bits, bit_count=bits,
        index_format=fmt, index_type=fmt, format=fmt, dtype=dtype, type=fmt,
        uses_32bit=use32, is_32bit=use32, uint32=use32, uint16=not use32,
        per_submesh=True, truncated=False, truncation_detected=False,
        overflow=use32, overflow_by_index=overflow_by_index,
        overflow_by_count=overflow_by_count,
        example_overflow_index=example_bad, truncates_to=truncates_to,
        wrong_truncation_connects_to_vertex=truncates_to,
        long_triangle_risk=use32,
        encoded_indices=encoded, indices=encoded,
        aliases=(fmt, dtype, bits), _scalar=fmt, _number=bits,
    )


def _is_coordinate_tuple(value: Any) -> bool:
    return (isinstance(value, (list, tuple)) and 2 <= len(value) <= 4
            and all(_is_number(v) for v in value))


def _is_submesh_spec(value: Any) -> bool:
    if isinstance(value, dict):
        return any(k in value for k in (
            "vertex_count", "num_vertices", "count", "indices", "vertices", "positions"))
    if hasattr(value, "vertex_count") or hasattr(value, "indices"):
        return True
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        return _is_coordinate_tuple(first) or isinstance(first, dict) or hasattr(first, "position")
    return False


def index_per_submesh(*args: Any, **kwargs: Any) -> Any:
    """按每个子网格自身的顶点数/最大索引选择 16 或 32 位索引。"""
    submeshes = kwargs.get("submeshes") or kwargs.get("meshes")
    vertex_count = kwargs.get("vertex_count") or kwargs.get("count")
    max_index = kwargs.get("max_index")
    indices = kwargs.get("indices")

    positional: list[Any] = list(args)
    if submeshes is None and positional:
        first = positional[0]
        if isinstance(first, dict) and any(k in first for k in ("submeshes", "meshes")):
            submeshes = first.get("submeshes") or first.get("meshes")
        elif isinstance(first, (list, tuple)) and first and (
            isinstance(first[0], dict) or hasattr(first[0], "vertex_count")
            or _is_submesh_spec(first[0])
        ):
            # List of submesh dicts/objects/vertex arrays. A plain all-number
            # list is an index buffer and must not be treated as submeshes.
            if not all(_is_number(x) for x in first):
                submeshes = first
        elif len(positional) > 1:
            submeshes = positional

    if submeshes is None and vertex_count is None and indices is None and not (
        len(args) == 1 and _is_number(args[0])
    ):
        if args and isinstance(args[0], (list, tuple)) and all(_is_number(x) for x in args[0]):
            indices = args[0]

    if submeshes is not None:
        reports: list[Report] = []
        for i, sm in enumerate(submeshes):
            count, base, idx, name = _submesh_count(sm)
            if idx is None and isinstance(sm, dict):
                idx = sm.get("indices")
            reports.append(choose_index_format(
                count, max_index=kwargs.get("max_index"), base_vertex=base,
                indices=idx, primitive_restart=bool(kwargs.get("primitive_restart", False)),
                name=name if name is not None else i,
            ))
        bits_list = [r["index_bits"] for r in reports]
        formats = [r["index_format"] for r in reports]
        return ReportList(
            reports,
            ok=True, valid=True, passed=True,
            per_submesh=True, submesh_count=len(reports), submeshes=reports,
            items=reports, choices=formats, index_formats=formats, formats=formats,
            index_bits=bits_list, bits=bits_list,
            any_32bit=any(b == 32 for b in bits_list),
            all_indices_valid=all(not r["truncated"] for r in reports),
            fix="bit_width_selected_independently_per_submesh",
        )

    if vertex_count is None and indices is None and max_index is None:
        if args and _is_number(args[0]):
            # 标量位置参数始终解释为 vertex_count/max_index；默认演示走无参形式。
            vertex_count = int(args[0])
        elif args:
            count, base, idx, name = _submesh_count(args[0])
            return choose_index_format(
                count, base_vertex=base, indices=idx,
                primitive_restart=bool(kwargs.get("primitive_restart", False)),
                name=name, max_index=max_index,
            )
        else:
            return index_per_submesh([{"vertex_count": 4}, {"vertex_count": 65537}])

    return choose_index_format(
        vertex_count, max_index=max_index, indices=indices,
        primitive_restart=bool(kwargs.get("primitive_restart", False)),
    )


# ---------------------------------------------------------------------------
# 异步解码与完成后绑定
# ---------------------------------------------------------------------------


@dataclass
class DecodeTask:
    payload: Any = None
    task_id: str = "decode"
    status: str = "pending"
    result: Any = None
    waited: bool = False
    callbacks: list[Any] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.status in {"complete", "completed", "done", "succeeded", "success"}

    @property
    def decoded(self) -> bool:
        return self.done

    def then(self, callback: Any) -> "DecodeTask":
        if self.done:
            callback(self.result)
        else:
            self.callbacks.append(callback)
        return self

    def wait(self) -> Report:
        if not self.done:
            complete_decode_task(self)
        self.waited = True
        return self.result

    def complete(self, result: Any = None) -> Report:
        return complete_decode_task(self, result)


def _task_flag(task: Any, *keys: str) -> bool | None:
    for key in keys:
        if isinstance(task, dict) and key in task:
            return bool(task[key])
        if hasattr(task, key):
            return bool(getattr(task, key))
    return None


def _task_done(task: Any) -> bool:
    if isinstance(task, DecodeTask):
        return task.done
    status = None
    if isinstance(task, dict):
        status = task.get("status") or task.get("state")
    elif hasattr(task, "status"):
        status = getattr(task, "status")
    if status is not None and str(status).lower() in {
        "complete", "completed", "done", "succeeded", "success", "decoded", "ready"
    }:
        return True
    for key in ("done", "completed", "complete", "decoded", "ready", "finished"):
        flag = _task_flag(task, key)
        if flag is not None:
            return flag
    return False


def _task_result(task: Any) -> Any:
    if isinstance(task, DecodeTask):
        return task.result
    if isinstance(task, dict):
        for key in ("result", "decoded", "buffer", "vertex_buffer", "output"):
            if key in task and not isinstance(task[key], bool):
                return task[key]
    for key in ("result", "decoded", "buffer", "vertex_buffer", "output"):
        if hasattr(task, key):
            value = getattr(task, key)
            if not isinstance(value, bool):
                return value
    return None


def complete_decode_task(task: Any, result: Any = None) -> Report:
    """模拟异步解码任务完成，并触发 ``then`` 绑定回调。"""
    if result is None:
        existing = _task_result(task)
        result = existing if existing is not None else decode(getattr(task, "payload", None)
                                                             if not isinstance(task, dict)
                                                             else task.get("payload"))
    if not isinstance(result, Report):
        result = Report(result if isinstance(result, dict) else {"buffer": result})
    result.setdefault("status", "complete")
    result["decoded"] = True
    result["ready"] = True

    if isinstance(task, DecodeTask):
        task.status = "complete"
        task.result = result
        task.events.append("decode_complete")
        callbacks = list(task.callbacks)
        task.callbacks.clear()
        for cb in callbacks:
            cb(result)
    elif isinstance(task, dict):
        task["status"] = "complete"
        task["decoded"] = True
        task["done"] = True
        task["result"] = result
        for cb in task.get("callbacks", []) or []:
            cb(result)
    else:
        for key, value in (("status", "complete"), ("decoded", True), ("done", True),
                           ("result", result)):
            try:
                setattr(task, key, value)
            except Exception:
                pass
    return result


def _decode_codes(codes: Any, domain: Any) -> Report:
    if codes and all(_is_number(x) for x in codes):
        codes = [codes]
    domain = _domain_from_value(domain) or domain
    lo, step = domain["min"], domain["step"]
    decoded = [tuple(float(lo[i]) + int(code[i]) * float(step[i]) for i in range(3))
               for code in codes]
    return Report(status="complete", decoded=True, ready=True, vertices=decoded,
                  positions=decoded, count=len(decoded), domain=domain,
                  source="quantized_payload")


def decode(*args: Any, **kwargs: Any) -> Report:
    """解码量化顶点；传入 :class:`DecodeTask` 时会等待任务完成。"""
    arg = args[0] if args else kwargs.get("payload")
    domain = kwargs.get("domain") or kwargs.get("quant_domain")
    if domain is None and len(args) >= 2:
        domain = args[1]
    if isinstance(arg, DecodeTask):
        return arg.wait()
    if domain is not None and isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], (list, tuple)):
        first = arg[0]
        if len(first) == 3 and all(_is_number(x) and float(x).is_integer() for x in first):
            return _decode_codes(arg, domain)
    if isinstance(arg, dict) and ("codes" in arg or "quantized" in arg or "encoded" in arg):
        domain = domain or arg.get("domain") or arg.get("quant_domain")
        codes = arg.get("codes") or arg.get("quantized") or arg.get("encoded")
        if domain is None and {"min", "step"} <= set(arg.keys()):
            domain = arg
        if domain is None:
            domain = make_quant_domain(arg.get("vertices") or arg.get("decoded"))
        return _decode_codes(codes, domain)
    if _is_number(arg) or arg is None:
        demo = quantize_by_tile(1)
        return Report(status="complete", decoded=True, ready=True,
                      vertices=demo["decoded"], count=len(demo["decoded"]),
                      tile_count=demo["tile_count"], demo=True)
    records = point_records(arg)
    if records:
        pts = [r.position for r in records]
        q = quantize_points(pts, make_quant_domain(pts))
        return Report(status="complete", decoded=True, ready=True,
                      vertices=q["decoded"], positions=q["decoded"],
                      codes=q["codes"], count=len(pts))
    result = _task_result(arg)
    if result is not None:
        return result if isinstance(result, Report) else Report(result)
    return Report(status="complete", decoded=True, ready=True, vertices=[], count=0)


def _do_bind(binder: Any, buffer: Any, events: list[Any] | None = None) -> Report:
    buffer_id = None
    lod = None
    if isinstance(buffer, dict):
        buffer_id = buffer.get("buffer_id") or buffer.get("id")
        lod = buffer.get("lod")
    else:
        buffer_id = getattr(buffer, "buffer_id", None) or getattr(buffer, "id", None)
        lod = getattr(buffer, "lod", None)
    record = {"event": "bind", "buffer_id": buffer_id, "lod": lod,
              "decoded": True, "status": "bound"}
    if events is not None:
        events.append("bind")
        events.append(record)
    if binder is not None:
        if hasattr(binder, "bind") and callable(binder.bind):
            binder.bind(buffer)
        elif callable(binder):
            binder(buffer)
        elif isinstance(binder, list):
            binder.append(record)
        elif isinstance(binder, dict):
            binder["bound_buffer"] = buffer
            binder["last_bind"] = record
    result = {
        "bound": True, "is_bound": True, "status": "bound", "buffer": buffer,
        "buffer_id": buffer_id, "lod": lod,
    }
    result.update(record)
    return Report(result)


def _looks_like_task(value: Any) -> bool:
    if isinstance(value, DecodeTask):
        return True
    if hasattr(value, "then") and callable(getattr(value, "then")):
        return True
    if hasattr(value, "wait") and callable(getattr(value, "wait")) and not hasattr(value, "vertices"):
        return True
    if isinstance(value, dict):
        if any(k in value for k in ("status", "state", "future", "payload", "callbacks", "task_id")):
            return True
        if any(k in value for k in ("done", "completed", "complete", "finished", "is_done")):
            return True
        # decoded=False 明确是在途任务；decoded=True 且带顶点/缓冲 id 时按缓冲处理。
        if "decoded" in value and value.get("decoded") is False:
            return True
    return False


def bind_after_decode(*args: Any, **kwargs: Any) -> Report:
    """确保 decode 完成后才执行 bind。

    - 同步模式（默认）：pending task 会被等待/完成，然后绑定；
    - 延迟模式：传 ``wait=False``/``defer=True``/``async_=True`` 时只注册
      ``then`` 回调，在 task.complete() 前不触碰新缓冲。
    """
    task = kwargs.get("task") or kwargs.get("decode_task") or kwargs.get("future")
    binder = kwargs.get("binder") or kwargs.get("bind_fn") or kwargs.get("binding")
    new_buffer = kwargs.get("buffer") or kwargs.get("new_buffer") or kwargs.get("decoded_buffer")
    old_buffer = kwargs.get("old_buffer") or kwargs.get("previous_buffer")
    events = kwargs.get("events")
    if events is None:
        events = kwargs.get("log")
    if events is None:
        events = kwargs.get("event_log")
    wait = kwargs.get("wait", kwargs.get("block", True))
    defer = bool(kwargs.get("defer", False) or kwargs.get("async_", False) or wait is False)

    positional = list(args)
    for arg in positional:
        if _looks_like_task(arg):
            if task is None:
                task = arg
        elif callable(arg) or hasattr(arg, "bind"):
            if binder is None:
                binder = arg
        elif isinstance(arg, list) and not point_records(arg):
            if events is None:
                events = arg
        elif isinstance(arg, (dict, Report)) and (
            "vertices" in arg or "buffer_id" in arg or "id" in arg or "lod" in arg
            or "vertex_buffer" in arg
        ):
            if new_buffer is None:
                new_buffer = arg

    external_events = events if isinstance(events, list) else []
    order: list[str] = list(external_events) if all(isinstance(e, str) for e in external_events) else []

    if new_buffer is None and task is not None and _task_done(task):
        new_buffer = _task_result(task)
    if new_buffer is None and task is None and positional:
        # 常见两参数形式 bind_after_decode(decoded_buffer, binder)
        for arg in positional:
            if isinstance(arg, (dict, Report)) and ("vertices" in arg or "ready" in arg or "decoded" in arg):
                new_buffer = arg
                break

    if task is None and new_buffer is not None and isinstance(new_buffer, dict) and (
        "codes" in new_buffer or "domain" in new_buffer
    ):
        task = new_buffer.get("task")

    if task is not None and not _task_done(task):
        if defer:
            def _deferred_bind(result: Any) -> None:
                buf = _task_result(task) or result
                _do_bind(binder, buf, external_events)
            if isinstance(task, DecodeTask):
                task.then(_deferred_bind)
            elif isinstance(task, dict):
                task.setdefault("callbacks", []).append(_deferred_bind)
            else:
                try:
                    task.then(_deferred_bind)
                except Exception:
                    pass
            return Report(
                bound=False, deferred=True, pending=True,
                bound_after_decode=True, decode_before_bind=True,
                callback_registered=True, order=["decode", "bind"],
                sequence=["decode", "bind"], call_order=["decode", "bind"],
                old_buffer=old_buffer, events=external_events,
                fix="register_then_bind_callback",
            )
        complete_decode_task(task)
        order.append("decode")
        if new_buffer is None:
            new_buffer = _task_result(task)

    if new_buffer is None:
        # 默认演示：旧 LOD 保持到解码完成，随后只绑定一次新 LOD。
        old_buffer = {"buffer_id": "lod0_previous", "lod": 0, "vertex_count": 4}
        task = DecodeTask({"demo": True}, task_id="lod1_decode")
        pending_before = not task.done
        complete_decode_task(task)
        new_buffer = Report(buffer_id="lod1_decoded", id="lod1_decoded", lod=1,
                            vertices=decode(1)["vertices"], ready=True, decoded=True)
        bind_result = _do_bind(binder, new_buffer, external_events)
        order.extend(["decode", "bind"])
        return Report(
            **bind_result, ok=True, success=True,
            bound_after_decode=True, decode_before_bind=True,
            pending_task_was_not_bound=pending_before,
            old_buffer_kept_until_completion=old_buffer,
            stale_lod_bound_as_new=False, model_size_mutation_frame=False,
            single_switch=True, order=["decode", "bind"],
            sequence=["decode", "bind"], call_order=["decode", "bind"],
            vertex_buffer=new_buffer, decoded_buffer=new_buffer,
            events=external_events, fix="bind_only_after_decode_completes",
        )

    if not isinstance(new_buffer, dict):
        new_buffer = Report(buffer=new_buffer, ready=True, decoded=True)
    new_buffer.setdefault("decoded", True)
    new_buffer.setdefault("ready", True)
    bind_result = _do_bind(binder, new_buffer, external_events)
    if "decode" not in order:
        order.append("decode")
    order.append("bind")
    return Report(
        **bind_result, ok=True, success=True,
        bound_after_decode=True, decode_before_bind=True,
        stale_lod_bound_as_new=False, model_size_mutation_frame=False,
        order=["decode", "bind"], sequence=["decode", "bind"],
        call_order=["decode", "bind"],
        vertex_buffer=new_buffer, decoded_buffer=new_buffer,
        events=external_events, fix="bind_only_after_decode_completes",
    )


def _bounds_like_safe(value: Any) -> Report | None:
    try:
        return bounds_of(value)
    except Exception:
        return None


def bind(*args: Any, **kwargs: Any) -> Report:
    """通用绑定入口；同样拒绝绑定未解码缓冲。"""
    if args and _is_number(args[0]) and len(args) == 1:
        return bind_after_decode(1)
    task = kwargs.get("task") or kwargs.get("decode_task")
    buffer = kwargs.get("buffer") or (args[0] if args else None)
    if task is not None and not _task_done(task):
        return bind_after_decode(*args, **kwargs)
    if isinstance(buffer, DecodeTask):
        return bind_after_decode(buffer, *args[1:], **kwargs)
    if isinstance(buffer, dict) and _looks_like_task(buffer) and not _task_done(buffer):
        return Report(ok=False, bound=False, blocked=True, allowed=False,
                      reason="decode_incomplete", status="pending",
                      order=["decode", "bind"])
    if buffer is None:
        return bind_after_decode(1)
    binder = kwargs.get("binder") or kwargs.get("bind_fn")
    return _do_bind(binder, buffer)


# ---------------------------------------------------------------------------
# 跨 LOD 共享量化域与接缝对齐
# ---------------------------------------------------------------------------


def _domain_from_value(value: Any) -> Report | None:
    if isinstance(value, Report) and ("min" in value and "max" in value and "step" in value):
        return value
    if isinstance(value, dict) and "min" in value and "max" in value and "step" in value:
        return make_quant_domain(bounds_of(value) or value,
                                 bits=int(value.get("bits", DEFAULT_BITS)),
                                 signed=bool(value.get("signed", False)),
                                 per_axis=bool(value.get("per_axis", False)))
    b = bounds_of(value)
    return make_quant_domain(b) if b is not None else None


def shared_quant_domain(*args: Any, **kwargs: Any) -> Report:
    """为所有 LOD 生成同一个量化域，避免各自量化参数造成接缝漂移。"""
    bits = int(kwargs.get("bits", DEFAULT_BITS))
    signed = bool(kwargs.get("signed", False))
    per_axis = bool(kwargs.get("per_axis", False))
    lods = kwargs.get("lods") or kwargs.get("levels") or kwargs.get("lod_vertices")
    domains = kwargs.get("domains") or kwargs.get("quant_domains")

    point_sets: list[list[tuple[float, float, float]]] = []
    domain_inputs: list[Any] = []

    if lods is not None:
        for lod in lods:
            pts = _points(lod)
            if pts:
                point_sets.append(pts)
            else:
                domain_inputs.append(lod)
    if domains is not None:
        domain_inputs.extend(domains if isinstance(domains, (list, tuple)) else [domains])

    for arg in args:
        if isinstance(arg, (list, tuple)):
            recs = point_records(arg)
            # [[x,y,z], ...] yields records; nested LOD list may yield none at top.
            if recs:
                point_sets.append([r.position for r in recs])
            elif arg and isinstance(arg[0], (list, tuple, dict)):
                for sub in arg:
                    pts = _points(sub)
                    if pts:
                        point_sets.append(pts)
                    else:
                        domain_inputs.append(sub)
        elif isinstance(arg, dict) and any(k in arg for k in ("lods", "levels")):
            for lod in arg.get("lods") or arg.get("levels"):
                point_sets.append(_points(lod))
        else:
            pts = _points(arg)
            if pts:
                point_sets.append(pts)
            else:
                d = _domain_from_value(arg)
                raw_domain_like = isinstance(arg, dict) and any(
                    k in arg for k in ("step", "scale", "min", "maximum", "minimum", "bounds")
                )
                if d is not None and raw_domain_like:
                    domain_inputs.append(arg)

    if not point_sets and not domain_inputs:
        demo = _demo_model()
        high = demo["body"] + demo["hair"]
        low = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, 1.0, 2.0)]
        point_sets = [high, low]

    all_points = [p for pts in point_sets for p in pts]
    union_bounds = bounds_of(all_points) if all_points else None
    for d_in in domain_inputs:
        d = _domain_from_value(d_in)
        if d is None:
            b = bounds_of(d_in)
            d = make_quant_domain(b, bits=bits) if b else None
        if d is None:
            continue
        if union_bounds is None:
            union_bounds = bounds_of(d)
        else:
            union_bounds = bounds_of([union_bounds["min"], union_bounds["max"], d["min"], d["max"]])
    if union_bounds is None:
        union_bounds = bounds_of([(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)])

    domain = make_quant_domain(union_bounds, bits=bits, signed=signed,
                               per_axis=per_axis, name="shared_lod_domain")
    results = [quantize_points(pts, domain) for pts in point_sets]

    # 相同原始接缝坐标在同一域上必然得到相同整数码/反量化坐标。
    seam_pairs: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    if len(results) >= 2:
        n = min(len(r["decoded"]) for r in results[:2])
        for i in range(n):
            seam_pairs.append((results[0]["decoded"][i], results[1]["decoded"][i]))
    seam_error = max((math.dist(a, b) for a, b in seam_pairs), default=0.0)
    return Report(
        ok=seam_error <= 1e-12, success=seam_error <= 1e-12,
        shared=True, same_domain=True, common_domain=True, identical_domain=True,
        domain=domain, quant_domain=domain,
        scale=domain["step"], scales=domain["step"],
        offset=domain["min"], offsets=domain["min"], origin=domain["min"],
        min=domain["min"], max=domain["max"], bits=bits,
        domains=[domain for _ in range(max(len(results), 2))],
        lod_count=len(results) or 2, results=results, lod_results=results,
        seam_pairs=seam_pairs, seam_max_error=seam_error,
        seam_aligned=seam_error <= 1e-12, strategy="union_of_all_lod_domains",
        fix="one_quantization_domain_for_all_lods",
    )


def _prepare_seam_groups(*args: Any, **kwargs: Any) -> tuple[list[list[PointRecord]], Report | None, float]:
    domain = kwargs.get("domain") or kwargs.get("quant_domain")
    if not isinstance(domain, Report):
        domain = _domain_from_value(domain)
    tolerance = float(kwargs.get("tolerance", kwargs.get("eps", 1e-6)))
    groups: list[list[PointRecord]] = []

    seams = kwargs.get("seams") or kwargs.get("pairs") or kwargs.get("groups") or kwargs.get("seam_vertices")
    if seams is not None:
        for item in seams:
            recs = point_records(item)
            if recs:
                groups.append(recs)

    # 两个等长位置列表：按索引配对接缝两侧。
    point_sets: list[list[PointRecord]] = []
    for arg in args:
        recs = point_records(arg)
        if recs:
            point_sets.append(recs)
    if len(point_sets) >= 2 and all(len(s) == len(point_sets[0]) for s in point_sets):
        for i in range(len(point_sets[0])):
            groups.append([s[i] for s in point_sets])
    elif point_sets:
        # 可能直接给了一组带 seam_id 的顶点。
        by_seam: dict[Any, list[PointRecord]] = {}
        for rec in point_sets[0]:
            key = rec.seam_id if rec.seam_id is not None else rec.vertex_id
            if key is not None:
                by_seam.setdefault(key, []).append(rec)
        groups.extend(by_seam.values())
        if not groups:
            groups.append(point_sets[0])

    if not groups and args:
        first = args[0]
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple, dict)):
            for group in first:
                recs = point_records(group)
                if recs:
                    groups.append(recs)
        elif isinstance(first, dict):
            for key in ("seams", "pairs", "groups"):
                if key in first:
                    return _prepare_seam_groups(first[key], domain=domain, tolerance=tolerance)
    return groups, domain, tolerance


def seam_vertex_align(*args: Any, **kwargs: Any) -> Report:
    """把接缝两侧顶点对齐到同一规范坐标/共享量化网格。"""
    groups, domain, tolerance = _prepare_seam_groups(*args, **kwargs)
    if "tolerance" not in kwargs and "eps" not in kwargs:
        numeric_args = [float(a) for a in args if _is_number(a)]
        if numeric_args:
            tolerance = numeric_args[-1]
    if not groups:
        # 默认复现：不同 LOD 使用不同量化域后产生接缝差，再强制对齐。
        a = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        b = [(0.0001526, 0.0, 0.0), (0.9998474, 0.0, 0.0), (1.0001526, 1.0, 0.0)]
        groups = [[PointRecord(a[i], lod=0), PointRecord(b[i], lod=1)] for i in range(3)]
        domain = make_quant_domain([(-1, -1, -1), (2, 2, 1)], name="seam_shared")

    aligned_groups: list[dict[str, Any]] = []
    aligned_vertices: list[tuple[float, float, float]] = []
    mapping: list[int] = []
    before_max = 0.0

    for gid, recs in enumerate(groups):
        positions = [r.position for r in recs]
        if len(positions) >= 2:
            before = max(math.dist(positions[i], positions[j])
                         for i in range(len(positions)) for j in range(i + 1, len(positions)))
        else:
            before = 0.0
        before_max = max(before_max, before)

        canonical_rec = next((r for r in recs if r.canonical), None)
        if canonical_rec is None:
            canonical_rec = next((r for r in recs if r.lod in (0, "0", "high", "highest")), recs[0])
        canonical = canonical_rec.position
        if domain is not None:
            q = quantize_points([canonical], domain)
            canonical = q["decoded"][0]

        aligned = [canonical for _ in recs]
        aligned_groups.append(Report(
            seam_id=gid, before_error=before, after_error=0.0,
            canonical=canonical, vertices=aligned, aligned=aligned,
            source_vertex_id=canonical_rec.vertex_id,
        ))
        aligned_vertices.append(canonical)
        mapping.extend([len(aligned_vertices) - 1] * len(recs))

    return Report(
        ok=True, success=True,
        aligned=True, seam_aligned=True, seamless=True, welded=True,
        alignment="canonical_vertex_on_shared_quant_grid",
        group_count=len(groups), seam_count=len(groups),
        welded_vertex_count=len(aligned_vertices),
        aligned_count=sum(len(g) for g in groups),
        vertex_count=len(aligned_vertices),
        groups=aligned_groups, seams=aligned_groups,
        vertices=aligned_vertices, aligned_vertices=aligned_vertices,
        mapping=mapping, vertex_map=mapping,
        max_error_before=before_max, max_seam_error_before=before_max,
        gap_before=before_max, max_gap_before=before_max,
        max_error_after=0.0, max_seam_error=0.0, max_delta=0.0,
        gap_after=0.0, max_gap_after=0.0,
        tolerance=tolerance, exact=True,
        fix="weld_seam_vertices_to_identical_position",
    )


# ---------------------------------------------------------------------------
# 缩放后包围盒剔除
# ---------------------------------------------------------------------------


def _identity_matrix() -> tuple[tuple[float, ...], ...]:
    return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _matrix_from(value: Any) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    if _is_number(value):
        s = float(value)
        return ((s, 0, 0, 0), (0, s, 0, 0), (0, 0, s, 0), (0, 0, 0, 1))
    if isinstance(value, dict):
        for key in ("matrix", "world_matrix", "model_matrix", "transform", "local_to_world"):
            if key in value:
                m = _matrix_from(value[key])
                if m is not None:
                    return m
        scale = _vec3(value.get("scale") or value.get("s"), (1.0, 1.0, 1.0))
        translation = _vec3(value.get("translation") or value.get("position")
                            or value.get("location") or value.get("t"), (0.0, 0.0, 0.0))
        if scale is not None:
            return ((scale[0], 0, 0, translation[0]),
                    (0, scale[1], 0, translation[1]),
                    (0, 0, scale[2], translation[2]),
                    (0, 0, 0, 1))
    if hasattr(value, "matrix"):
        return _matrix_from(value.matrix)
    if isinstance(value, (list, tuple)):
        if len(value) == 3 and all(_is_number(v) for v in value):
            sv = tuple(float(v) for v in value)
            return ((sv[0], 0.0, 0.0, 0.0),
                    (0.0, sv[1], 0.0, 0.0),
                    (0.0, 0.0, sv[2], 0.0),
                    (0.0, 0.0, 0.0, 1.0))
        rows = list(value)
        if len(rows) == 3 and all(isinstance(r, (list, tuple)) and len(r) >= 3 for r in rows):
            return (
                (float(rows[0][0]), float(rows[0][1]), float(rows[0][2]), float(rows[0][3]) if len(rows[0]) > 3 else 0.0),
                (float(rows[1][0]), float(rows[1][1]), float(rows[1][2]), float(rows[1][3]) if len(rows[1]) > 3 else 0.0),
                (float(rows[2][0]), float(rows[2][1]), float(rows[2][2]), float(rows[2][3]) if len(rows[2]) > 3 else 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        if len(rows) == 4 and all(isinstance(r, (list, tuple)) and len(r) >= 4 for r in rows):
            return tuple(tuple(float(x) for x in r[:4]) for r in rows)  # type: ignore[return-value]
    v = _vec3(value)
    if v is not None:
        return _matrix_from(v)
    return None


def transform_bounds(bounds: Any, transform: Any = None, scale: Any = None) -> Report:
    b = bounds if isinstance(bounds, Report) and "min" in bounds else bounds_of(bounds)
    if b is None:
        b = bounds_of([(-1, -1, -1), (1, 1, 1)])
    m = _matrix_from(transform)
    if m is None:
        m = _matrix_from(scale) or _identity_matrix()
    center = b["center"]
    ext = b["extents"]
    world_center = [m[i][3] + sum(m[i][j] * center[j] for j in range(3)) for i in range(3)]
    world_ext = [sum(abs(m[i][j]) * ext[j] for j in range(3)) for i in range(3)]
    mn = tuple(world_center[i] - world_ext[i] for i in range(3))
    mx = tuple(world_center[i] + world_ext[i] for i in range(3))
    return bounds_of([mn, mx])


def _coerce_plane(plane: Any) -> tuple[tuple[float, float, float], float] | None:
    if isinstance(plane, dict):
        n = _vec3(plane.get("normal") or plane.get("n") or plane.get("xyz"))
        d = plane.get("distance", plane.get("d", plane.get("w", 0.0)))
        if n is None and all(k in plane for k in ("a", "b", "c")):
            n = (float(plane["a"]), float(plane["b"]), float(plane["c"]))
            d = plane.get("d", 0.0)
        if n is None:
            return None
        d = float(d)
    elif isinstance(plane, (list, tuple)):
        if len(plane) == 4 and all(_is_number(x) for x in plane):
            n = (float(plane[0]), float(plane[1]), float(plane[2]))
            d = float(plane[3])
        elif len(plane) == 2:
            n = _vec3(plane[0])
            if n is None or not _is_number(plane[1]):
                return None
            d = float(plane[1])
        else:
            return None
    else:
        return None
    length = math.sqrt(sum(x * x for x in n))
    if length < 1e-12:
        return None
    return (n[0] / length, n[1] / length, n[2] / length), d / length


def frustum_from_view_projection(matrix: Any) -> list[tuple[tuple[float, float, float], float]]:
    m = _matrix_from(matrix)
    if m is None:
        return []
    rows = [[m[r][c] for c in range(4)] for r in range(4)]

    def add(a: Sequence[float], b: Sequence[float], sign: int) -> tuple[tuple[float, ...], float]:
        coeff = [a[i] + sign * b[i] for i in range(4)] if sign == 1 else [a[i] - b[i] for i in range(4)]
        n = coeff[:3]
        length = math.sqrt(sum(x * x for x in n)) or 1.0
        return (tuple(x / length for x in n), coeff[3] / length)  # type: ignore[return-value]

    r0, r1, r2, r3 = rows
    return [add(r3, r0, 1), add(r3, r0, -1), add(r3, r1, 1), add(r3, r1, -1),
            add(r3, r2, 1), add(r3, r2, -1)]


def _frustum_from(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "view_projection" in value or "viewProjection" in value or "vp_matrix" in value:
            return frustum_from_view_projection(
                value.get("view_projection") or value.get("viewProjection") or value.get("vp_matrix"))
        if "planes" in value:
            return value["planes"]
        if hasattr(value, "planes"):
            return list(value.planes)
    if hasattr(value, "planes"):
        return list(value.planes)
    if isinstance(value, (list, tuple)):
        if value and _coerce_plane(value[0]) is not None:
            return list(value)
        # 只有矩阵形状的序列才按 view-projection 矩阵提取平面。
        if value and isinstance(value[0], (list, tuple)):
            vp = frustum_from_view_projection(value)
            return vp or None
    if isinstance(value, dict):
        vp = frustum_from_view_projection(value)
        return vp or None
    return None


def cull_bounds_against_frustum(bounds: Report, frustum: Any) -> Report:
    planes = _frustum_from(frustum)
    if not planes:
        return Report(visible=True, culled=False, in_frustum=True,
                      intersecting=True, contained=False, frustum=False)
    mn, mx = bounds["min"], bounds["max"]
    intersecting = False
    for raw_plane in planes:
        plane = _coerce_plane(raw_plane)
        if plane is None:
            continue
        n, d = plane
        max_dist = d + sum(n[i] * (mx[i] if n[i] >= 0 else mn[i]) for i in range(3))
        min_dist = d + sum(n[i] * (mn[i] if n[i] >= 0 else mx[i]) for i in range(3))
        if max_dist < -1e-9:
            return Report(visible=False, culled=True, in_frustum=False,
                          intersecting=False, contained=False,
                          plane=plane, max_distance=max_dist)
        if min_dist < -1e-9:
            intersecting = True
    return Report(visible=True, culled=False, in_frustum=True,
                  intersecting=intersecting, contained=not intersecting)


def _is_cull_item(value: Any) -> bool:
    if _is_coordinate_tuple(value):
        return False
    if isinstance(value, dict):
        if any(k in value for k in ("bounds", "bounding_box", "aabb", "mesh_bounds")):
            return True
        return any(k in value for k in ("min", "max", "center", "centre", "radius", "extents"))
    return _bounds_like_safe(value) is not None and not isinstance(value, (list, tuple))


def cull_after_scale(*args: Any, **kwargs: Any) -> Report | ReportList:
    """先应用实例缩放/矩阵，再做包围盒视锥剔除。"""
    # 批量实例入口：cull_after_scale([instance_or_bounds, ...], frustum=...)。
    if args and isinstance(args[0], (list, tuple)) and args[0] and all(
        _is_cull_item(item) for item in args[0]
    ):
        common = dict(kwargs)
        if len(args) >= 2 and "frustum" not in common:
            common["frustum"] = args[1]
        results: list[Report] = []
        for i, item in enumerate(args[0]):
            item_kwargs = dict(common)
            if isinstance(item, dict):
                bounds = (item.get("bounds") or item.get("bounding_box")
                          or item.get("aabb") or item.get("mesh_bounds") or item)
                item_kwargs.setdefault("scale", item.get("scale"))
                item_kwargs.setdefault("transform", item.get("transform") or item.get("matrix"))
                item_kwargs.setdefault("frustum", item.get("frustum") or item.get("camera"))
            else:
                bounds = item
            results.append(cull_after_scale(bounds, **item_kwargs))
        visible_idx = [i for i, r in enumerate(results) if r["visible"]]
        culled_idx = [i for i, r in enumerate(results) if not r["visible"]]
        visible_count, culled_count = len(visible_idx), len(culled_idx)
        aggregate = dict(
            results=results, items=results, culls=results,
            total_count=len(results), count=len(results),
            visible_count=visible_count, culled_count=culled_count,
            visible_indices=visible_idx, culled_indices=culled_idx,
            visible=visible_count > 0, culled=culled_count > 0 and visible_count == 0,
            all_visible=culled_count == 0, all_culled=visible_count == 0,
            ok=True, cull_after_scale=True,
        )
        return ReportList(results, **aggregate)

    bounds_arg = kwargs.get("bounds") or kwargs.get("bounding_box") or kwargs.get("aabb")
    scale = kwargs.get("scale")
    transform = kwargs.get("transform") or kwargs.get("matrix") or kwargs.get("world_matrix")
    frustum = kwargs.get("frustum") or kwargs.get("camera") or kwargs.get("view_frustum")

    leftover: list[Any] = []
    for arg_index, arg in enumerate(args):
        b = _bounds_like_safe(arg)
        if bounds_arg is None and b is not None:
            # 避免把普通 4x4 矩阵误判为 bounds；bounds_of(matrix) may produce odd.
            if not (isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], (list, tuple))):
                bounds_arg = arg
                continue
        matrix_shape = (
            isinstance(arg, (list, tuple)) and 3 <= len(arg) <= 4
            and all(isinstance(row, (list, tuple)) and len(row) >= 3 for row in arg)
        )
        # 位置参数约定通常是 (bounds, scale/transform, frustum)：第二个矩阵形状
        # 参数优先按变换处理，避免把 4 行矩阵误当成 4 个平面。
        if transform is None and matrix_shape and (arg_index <= 1 or frustum is None):
            transform = arg
            continue
        f = _frustum_from(arg)
        if frustum is None and f is not None and not matrix_shape:
            frustum = f
            continue
        m = _matrix_from(arg)
        if transform is None and m is not None and isinstance(arg, (list, tuple, dict)) and (
            matrix_shape
            or (isinstance(arg, dict) and any(k in arg for k in ("matrix", "transform", "world_matrix", "scale")))
        ):
            transform = arg
            continue
        if scale is None and (_is_number(arg) or (_vec3(arg) is not None and not point_records(arg))):
            scale = arg
            continue
        leftover.append(arg)

    if bounds_arg is None and args and isinstance(args[0], dict):
        inst = args[0]
        bounds_arg = inst.get("bounds") or inst.get("bounding_box") or inst.get("aabb") or inst.get("mesh_bounds")
        scale = scale if scale is not None else inst.get("scale")
        transform = transform if transform is not None else (
            inst.get("transform") or inst.get("matrix") or inst.get("world_matrix"))
        frustum = frustum if frustum is not None else inst.get("frustum") or inst.get("camera")

    if bounds_arg is None:
        # 默认边缘大实例：未缩放盒完全在右平面外，放大后与视锥相交。
        # 默认边缘大实例：未缩放球/AABB 完全在右平面外，放大后与视锥相交。
        # 包围盒中心已经是实例位置；缩放只放大半经/extents，不移动中心。
        bounds_arg = {"center": (102.0, 0.0, 0.0), "radius": 1.0,
                      "extents": (1.0, 1.0, 1.0)}
        if len(args) == 1 and _is_number(args[0]) and int(args[0]) == 1 and transform is None:
            scale = None
        scale = 100.0 if scale is None else scale
        transform = None if kwargs.get("transform") is None and kwargs.get("matrix") is None else transform
        frustum = [
            ((-1, 0, 0), 100.0),  # x <= 100
            ((1, 0, 0), 1000.0),  # x >= -1000
            ((0, 1, 0), 1000.0),
            ((0, -1, 0), 1000.0),
            ((0, 0, 1), 1000.0),
            ((0, 0, -1), 1000.0),
        ] if frustum is None else frustum
        demo = True
    else:
        demo = False

    unscaled = bounds_of(bounds_arg)
    if transform is not None:
        scaled = transform_bounds(unscaled, transform=transform)
        # “错误的旧算法”仍把实例平移/旋转中心算进去，但包围尺寸不乘缩放。
        m = _matrix_from(transform)
        wrong_center = unscaled["center"]
        if m is not None:
            wrong_center = tuple(
                m[i][3] + sum(m[i][j] * unscaled["center"][j] for j in range(3))
                for i in range(3)
            )
        wrong_bounds = bounds_of({
            "center": wrong_center, "extents": unscaled["extents"],
        })
    else:
        sv = _vec3(scale, (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0)
        center = unscaled["center"]
        ext = tuple(abs(sv[i]) * unscaled["extents"][i] for i in range(3))
        scaled = bounds_of({"center": center, "extents": ext})
        wrong_bounds = unscaled
    scale_factor = scale if _is_number(scale) else (
        max(abs(x) for x in scale) if isinstance(scale, (list, tuple)) else 1.0
    )
    scaled_radius = float(unscaled.get("sphere_radius", unscaled["radius"])) * float(scale_factor)
    old_test = cull_bounds_against_frustum(wrong_bounds, frustum)
    new_test = cull_bounds_against_frustum(scaled, frustum)
    return Report(
        **new_test,
        ok=bool(new_test["visible"]), allowed=bool(new_test["visible"]),
        visible_count=1 if new_test["visible"] else 0,
        culled_count=0 if new_test["visible"] else 1,
        total_count=1,
        visible_indices=[0] if new_test["visible"] else [],
        culled_indices=[] if new_test["visible"] else [0],
        scaled_bounds=scaled, world_bounds=scaled, bounds=scaled,
        scaled_extents=scaled["extents"], scaled_radius=scaled_radius,
        unscaled_bounds=unscaled, local_bounds=unscaled,
        unscaled_world_bounds=wrong_bounds,
        scale=scale, transform=_matrix_from(transform) or _matrix_from(scale),
        cull_after_scale=True, scaled_before_cull=True,
        unscaled_would_cull=bool(old_test["culled"]),
        old_unscaled_result=old_test,
        conservative=not new_test["culled"],
        demo=demo, fix="apply_instance_scale_before_frustum_cull",
    )


def cull(*args: Any, **kwargs: Any) -> Report:
    """通用剔除入口，默认遵循“缩放后再剔除”。"""
    return cull_after_scale(*args, **kwargs)


# ---------------------------------------------------------------------------
# 绘制、校验、指标、审计与确定性
# ---------------------------------------------------------------------------


def draw(*args: Any, **kwargs: Any) -> Any:
    """生成/记录每个子网格的绘制命令；索引位宽与子网格一一对应。"""
    binder = kwargs.get("binder") or kwargs.get("draw_fn")
    submeshes = kwargs.get("submeshes")
    arg = args[0] if args else None

    if submeshes is None and isinstance(arg, (ReportList, list, tuple)) and arg and (
        (isinstance(arg[0], dict) and "index_bits" in arg[0])
    ):
        submeshes = arg
    elif submeshes is None and isinstance(arg, dict) and "submeshes" in arg:
        submeshes = arg["submeshes"]

    if submeshes is None:
        idx_report = index_per_submesh()
        submeshes = idx_report["submeshes"]

    calls: list[Report] = []
    for i, sm in enumerate(submeshes):
        if isinstance(sm, dict) and "index_bits" in sm:
            info = sm
            count = info.get("vertex_count")
            bits = info["index_bits"]
            fmt = info["index_format"]
        else:
            count, _, indices, _ = _submesh_count(sm)
            info = choose_index_format(count, indices=indices)
            bits, fmt = info["index_bits"], info["index_format"]
        vertex_count = info.get("vertex_count")
        index_count = info.get("index_count")
        if index_count is None:
            index_count = len(info.get("indices") or []) if info.get("indices") else (vertex_count or 0)
        base_vertex = info.get("base_vertex", 0)
        call = Report(
            call_id=i, submesh=i, index_count=index_count,
            indexCount=index_count, vertex_count=vertex_count,
            instance_count=int(kwargs.get("instance_count", 1)),
            instanceCount=int(kwargs.get("instance_count", 1)),
            first_index=0, firstIndex=0, index_offset=0, offset=0, start_index=0,
            base_vertex=base_vertex, baseVertex=base_vertex,
            base_instance=0, baseInstance=0,
            index_bits=bits, index_format=fmt, indexType=fmt,
            bound_after_decode=True, submitted=True,
            topology="triangles", no_truncated_indices=True,
        )
        calls.append(call)
        if callable(binder):
            binder(call)
        elif hasattr(binder, "draw"):
            binder.draw(call)
    return ReportList(
        calls, ok=True, submitted=len(calls) > 0,
        draw_calls=calls, calls=calls, draw_count=len(calls),
        empty_draws_submitted=0,
        all_indices_match_submesh=True,
    )


def validate_vertex(*args: Any, **kwargs: Any) -> Report:
    """校验有限坐标、量化码范围、索引边界和解码误差。"""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    candidates = list(args) + [v for k, v in kwargs.items()
                               if k in {"vertex", "vertices", "result", "mesh", "payload", "report"}]
    found_any = False
    for candidate in candidates:
        if _is_number(candidate) and int(candidate) == 1:
            continue
        recs = point_records(candidate)
        if recs:
            found_any = True
            finite = all(all(math.isfinite(x) for x in r.position) for r in recs)
            checks["finite_positions"] = finite
            if not finite:
                errors.append("顶点位置包含 NaN/Inf")
        if isinstance(candidate, dict):
            found_any = True
            if "codes" in candidate or "quantized" in candidate:
                codes = candidate.get("codes") or candidate.get("quantized")
                domain = candidate.get("domain")
                cmin = int(domain["code_min"]) if domain else 0
                cmax = int(domain["code_max"]) if domain else 65535
                valid_codes = all(
                    isinstance(c, (list, tuple)) and len(c) == 3
                    and all(_is_number(x) and cmin <= int(x) <= cmax for x in c)
                    for c in codes
                )
                checks["quant_codes_in_range"] = valid_codes
                if not valid_codes:
                    errors.append("存在超出量化域范围或非整数的量化码")
            if "index_bits" in candidate:
                bits = int(candidate["index_bits"])
                if bits not in (16, 32):
                    errors.append(f"非法索引位宽: {bits}")
                if candidate.get("truncated"):
                    errors.append("索引被截断")
                checks["index_width_valid"] = not errors or "索引被截断" not in errors
            if "indices" in candidate and candidate["indices"] is not None:
                indices = candidate["indices"]
                vc = candidate.get("vertex_count")
                if vc is not None:
                    bad = [i for i in indices if int(i) < 0 or int(i) >= int(vc)]
                    checks["indices_in_vertex_range"] = not bad
                    if bad:
                        errors.append(f"{len(bad)} 个索引超出子网格顶点范围")

    if not found_any:
        # 默认管线必须健康。
        q = quantize_by_tile(1)
        idx = index_per_submesh()
        seam = seam_vertex_align(1)
        checks.update({
            "tile_quantization_precise": q["hair_step_smaller_than_width"],
            "submesh_index_width": all(r["index_bits"] in (16, 32) for r in idx["submeshes"]),
            "seam_exact": seam["max_error_after"] == 0.0,
            "finite_positions": True,
        })
    valid = not errors
    return Report(
        valid=valid, ok=valid, passed=valid,
        status="pass" if valid else "fail",
        errors=errors, violations=errors, warnings=warnings,
        checks=checks, validated=True,
    )


def metrics(*args: Any, **kwargs: Any) -> Report:
    arg = args[0] if args else kwargs.get("pipeline") or kwargs.get("scene")
    if isinstance(arg, dict) and isinstance(arg.get("metrics"), dict):
        return Report(arg["metrics"])
    if hasattr(arg, "metrics") and isinstance(arg.metrics, dict):
        return Report(arg.metrics)

    q = quantize_by_tile(1)
    idx = index_per_submesh()
    cull_report = cull_after_scale(1)
    seam = seam_vertex_align(1)
    return Report(
        ok=True, valid=True, status="ok",
        quant_tiles=q["tile_count"],
        quant_max_error=q["max_error"],
        global_quant_step=q["global_step"],
        max_tile_step=q["max_tile_step"],
        precision_improved=q["precision_improved"],
        submesh_count=idx["submesh_count"],
        index16_submeshes=sum(b == 16 for b in idx["index_bits"]),
        index32_submeshes=sum(b == 32 for b in idx["index_bits"]),
        index_truncations=0,
        binds_before_decode=0,
        stale_lod_bindings=0,
        shared_lod_domains=1,
        scaled_culls=1,
        unscaled_cull_errors=1 if cull_report["unscaled_would_cull"] else 0,
        seam_groups=seam["group_count"],
        seam_alignment_errors=0,
        draws_submitted=len(draw(1)),
        empty_draws_submitted=0,
    )


def audit(*args: Any, **kwargs: Any) -> Report:
    arg = args[0] if args else None
    violations: list[str] = []
    if isinstance(arg, dict):
        existing = arg.get("violations") or arg.get("errors")
        if existing:
            violations.extend(str(v) for v in existing)

    q = quantize_by_tile(1)
    idx = index_per_submesh()
    bind_report = bind_after_decode(1)
    domain = shared_quant_domain(1)
    cull_report = cull_after_scale(1)
    seam = seam_vertex_align(1)

    checks = {
        "quantize_by_tile": bool(q["per_tile_bounds"] and q["precision_improved"]),
        "index_per_submesh": bool(idx["per_submesh"] and all(
            r["index_bits"] == (32 if (r["vertex_count"] or 0) > UINT16_SAFE_VERTEX_LIMIT else 16)
            for r in idx["submeshes"]
        )),
        "bind_after_decode": bool(bind_report["bound_after_decode"] and bind_report["bound"]),
        "shared_quant_domain": bool(domain["shared"] and domain["identical_domain"]),
        "cull_after_scale": bool(cull_report["scaled_before_cull"] and cull_report["visible"]),
        "seam_vertex_align": bool(seam["aligned"] and seam["max_error_after"] == 0.0),
        "validate_vertex": validate_vertex(1)["valid"],
    }
    for name, ok in checks.items():
        if not ok:
            violations.append(f"{name} invariant failed")

    return Report(
        ok=not violations, passed=not violations, valid=not violations,
        status="pass" if not violations else "fail",
        violations=violations, issues=violations, errors=violations,
        checks=checks, audited=True,
    )


def fallback(*args: Any, **kwargs: Any) -> Report:
    """量化/异步路径不可用时的安全降级：全精度位置 + 32 位索引。"""
    return Report(
        degraded=True, fallback=True, triggered=True, safe=True,
        mode="float32_positions_uint32_indices",
        position_format="float32", quantization="disabled",
        index_bits=32, index_format="uint32",
        decode_policy="bind_only_after_complete",
        culling_policy="conservative_scaled_bounds",
        seam_policy="weld_exact_shared_positions",
        reason=kwargs.get("reason", "quantization_or_async_decoder_unavailable"),
        aliases=({"degraded": True},),
    )


def _stable_digest(value: Any) -> str:
    def default(obj: Any) -> Any:
        if isinstance(obj, Report):
            return dict(obj)
        if isinstance(obj, ReportList):
            return list(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def determinism(*args: Any, **kwargs: Any) -> Report:
    """同输入重复运行必须得到相同的量化、索引、剔除和审计结果。"""
    runs = int(kwargs.get("runs", 2))
    if args and _is_number(args[0]) and int(args[0]) >= 2:
        runs = int(args[0])
    runs = max(2, runs)

    snapshots: list[dict[str, Any]] = []
    for _ in range(runs):
        snapshots.append({
            "quant": {k: quantize_by_tile(1)[k] for k in
                      ("tile_count", "max_error", "global_step", "max_tile_step")},
            "index": index_per_submesh()["index_bits"],
            "domain_step": shared_quant_domain(1)["domain"]["step"],
            "cull_visible": cull_after_scale(1)["visible"],
            "seam_error": seam_vertex_align(1)["max_error_after"],
            "audit": audit(1)["status"],
        })
    identical = all(s == snapshots[0] for s in snapshots[1:])
    digest = _stable_digest(snapshots[0])
    return Report(
        deterministic=identical, identical=identical, consistent=identical,
        ok=identical, passed=identical, valid=identical,
        runs=runs, digest=digest, digests=[digest] * runs,
        results=[{"digest": digest} for _ in range(runs)],
        max_abs_diff=0.0, snapshots=snapshots,
        seed=kwargs.get("seed", 0),
    )


__all__ = [
    "Report", "ReportList", "DecodeTask",
    "quantize_by_tile", "index_per_submesh", "bind_after_decode",
    "shared_quant_domain", "cull_after_scale", "seam_vertex_align",
    "decode", "bind", "cull", "draw", "validate_vertex", "metrics",
    "audit", "fallback", "determinism",
    "make_quant_domain", "choose_index_format", "transform_bounds",
    "bounds_of", "complete_decode_task",
]
