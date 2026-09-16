bl_info = {
    "name": "Star Wars: Bounty Hunter importer",
    "author": "reverse-engineered 2026",
    "version": (1, 1, 2),
    "blender": (3, 2, 0),
    "location": "File > Import > SW: Bounty Hunter Mesh (.bin) / Level Area "
                "(.lvt) / Animation (.abin)",
    "description": "Imports SWBH models, animations and whole level areas; resolves "
                   "materials, textures, shaders and actor data automatically",
    "category": "Import-Export",
}

import os
import re
import struct

import bpy
from bpy.props import (StringProperty, BoolProperty, FloatProperty, IntProperty,
                       EnumProperty, CollectionProperty)
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Quaternion, Vector
from math import radians


# ============================================================================
#  .bin FORMAT  (STAR WARS: Bounty Hunter, Aspyr PC remaster)
# ============================================================================
#
#  The file is a serialised x64 memory image. Every internal offset is relative
#  to BYTE 8, not to the start of the file:   file_offset = stored_value + 8
#  The 0xAAAAAAAA words at the top are uninitialised pointer fields, not magic.
#
#  HEADER
#    +0x10  u32   submesh count
#    +0x14  u32   -> per-node matrix array
#    +0x1C  u32   -> node name/parent array (stride 44)
#    +0x24  u16   node count
#    +0x28  u32   -> submesh array
#    +0x38  4x4   matrix (identity in everything seen so far)
#
#  SUBMESH ARRAY, stride 20:
#    +0x00  u32   -> material   (0 = NONE: collision / logic geometry)
#    +0x08  u32   -> mesh descriptor
#    +0x10  u32   id
#
#  MATERIAL, stride 280:
#    +0x18  char  texture name, asciiz ("O_A_CratesC.tga")
#
#  MESH DESCRIPTOR, stride 32:
#    +0x00  u32   vertex format  0x4040 static | 0x4140 skinned | 0x9044 level
#                 bits 23/24/25 (one-hot: 0x800000 / 0x1000000 / 0x2000000)
#                 select an LOD level = (bit - 23), i.e. LOD0/LOD1/LOD2. Found
#                 on komarivosa.bin: 3 groups of submeshes, same texture names
#                 repeated across groups, triangle count strictly decreasing
#                 with LOD number (1927 / 543 / 243 tris). Measured on one
#                 file so far (METHODOLOGY.md §11 gap) - objects with a single
#                 LOD are assumed to be bit23/LOD0 whether or not the bit is
#                 actually set, so old single-LOD imports are unaffected.
#    +0x08  u16   index count
#    +0x0A  u16   vertex count
#    +0x14  u32   -> stream table
#
#  STREAM TABLE, 8-byte slots, 0 = stream absent:
#    slot 0  skinning (see below)
#    slot 1  indices     u16 x n_idx     (triangle LIST, not a strip)
#    slot 2  positions   f32[3] x n_vert
#    slot 3  normals     f32[3] x n_vert (already unit length)
#    slot 4  UVs         f32[2] x n_vert
#    slot 6  colours     u8[4] RGBA x n_vert   <- BAKED VERTEX LIGHTING
#
#  Level geometry shares one vertex buffer across all its submeshes; only the
#  index stream differs. Submeshes with a NULL material and no UV/normals are
#  collision hulls and are never drawn.
#
#  SKINNING  (stream slot 0)
#    +0x04  u32   set count
#    +0x08  u64   -> set array, stride 24
#
#    set, 24 bytes:
#      +0x00  u32  bone count
#      +0x04  u64  -> array of bone pointers (u64 each)
#      +0x0C  u32  range count
#      +0x10  u64  -> range array, stride 16
#
#    range, 16 bytes:
#      +0x00  u32  first vertex
#      +0x04  u32  vertex count
#      +0x08  u64  -> weights, one f32 per bone of the set
#
#  Weights are shared by the whole range: vertices are sorted so that runs with
#  the same (bones, weights) tuple sit together. Ranges tile [0, n_vert) with no
#  gaps and every weight tuple sums to exactly 1.0 - both verified on every
#  submesh of jangoui.bin and jango.bin.
#
#  A bone pointer resolves to an index into the node array:
#      stride = (hdr[0x1C] - hdr[0x14]) / node_count
#      index  = (bone_ptr - hdr[0x14]) / stride
#  hdr[0x14] addresses the per-node matrix array, hdr[0x1C] the name/parent one.
#  Checked semantically: Jango's visor submesh (ref_glass.tga) comes out bound
#  100% to SOCK_head, the hand submeshes to the finger bones, and so on.
#
#  Coordinate system is Z-up, same as Blender. Triangle winding already agrees
#  with the stored normals, so nothing needs flipping. 1 unit ~= 1 metre.
# ============================================================================

BASE = 8
NODE_STRIDE = 44

SLOT_SKIN, SLOT_IDX, SLOT_POS, SLOT_NRM, SLOT_UV, SLOT_5, SLOT_COL, SLOT_7 = range(8)

TEX_EXTS = (".dds", ".png", ".tga", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")

ATS_EXT = ".ats"
ASET_EXT = ".aset"
COL_EXT = ".col"
LVR_EXT = ".lvr"

_ATS_CACHE = {}
_ASET_CACHE = {}
_COL_CACHE = {}
_LVR_CACHE = {}

GENERIC_NAMES = re.compile(r"^(godnode|LgcGeo\d*|VisGeo\d*)$", re.I)


class BinParseError(Exception):
    pass


def _u16(d, o):
    return struct.unpack_from("<H", d, o)[0]


def _u32(d, o):
    return struct.unpack_from("<I", d, o)[0]


def _u64(d, o):
    return struct.unpack_from("<Q", d, o)[0]


def _cstr(d, o):
    end = d.find(b"\0", o)
    if end < 0:
        end = min(o + 256, len(d))
    return d[o:end].decode("ascii", "replace")


def is_modern_bin(path):
    """Peek at the first 4 bytes: the modern per-actor/room .bin format always
    starts with the 0xAAAAAAAA uninitialised-pointer fill. Two other, older
    containers share the .bin extension and will otherwise happily reach
    parse_bin() and fail confusingly deep in: the "Bin file Version 16Mar01.3"
    legacy exporter (gs_world.bin, and - measured on b1 - a dozen pre-portal-
    split pa_area_*.bin leftovers that are excluded from that level's own
    .lvt manifest for exactly this reason), and a third, still-unidentified
    header seen on 4 of b1's __visgeo*.bin starting 0xDEADDEAD at +4 (a
    different debug-fill convention again - recorded gap, not reverse-
    engineered). Whole-area import uses this instead of guessing from the
    filename, so all three are skipped by construction rather than by name."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\xAA\xAA\xAA\xAA"
    except OSError:
        return False


def is_utility_texture(stem):
    """"_Collision.tga", "_Portal.tga" etc - a leading underscore marks a
    non-visual utility texture (matches the loose _collision.dds/_portal.dds
    files sitting in every level's textures/ folder). Measured on real data:
    c3__pa_shallwayd.bin has 31 submeshes textured "_Collision.tga" that DO
    carry UV and normals - the existing "no material/UV/normals" collision
    rule does not catch them, so they were rendering as ordinary geometry
    with a texture that was never meant to be seen (pink in Blender, since
    nothing named _Collision exists as an actual image). f2__pa_f1.bin is
    the extreme case: all 38 submeshes are "_Collision.tga" - the entire
    file is a collision volume with no visible geometry at all."""
    return stem.startswith("_")


def lod_of(fmt):
    """LOD level from the vertex-format word: one-hot bit 23/24/25 -> 0/1/2.
    No bit set (the common case - most objects only have one LOD) -> 0, so
    they always come out as "the" LOD and nothing regresses for old files."""
    for bit in (23, 24, 25, 26, 27, 28, 29, 30):   # a couple of spare bits,
        if fmt & (1 << bit):                        # in case a model ships
            return bit - 23                         # a 4th/5th LOD someday
    return 0


def _parse_bin_modern(d):
    if len(d) < 0x80:
        raise BinParseError("too small to be a .bin")

    n_sub = _u32(d, 0x10)
    if not (0 < n_sub < 4096):
        raise BinParseError("implausible submesh count: %d" % n_sub)

    # +0x1C -> node array (stride 44), +0x24 (u16) -> node count.
    # A prop has one node; Jango has 47. What used to look like an "object name"
    # is simply the name of node 0.
    node_off = _u32(d, 0x1C) + BASE
    node_cnt = _u16(d, 0x24)
    nodes = []
    if 0 < node_cnt < 4096 and node_off + node_cnt * NODE_STRIDE <= len(d):
        for i in range(node_cnt):
            o = node_off + i * NODE_STRIDE
            nm = d[o:o + 36].split(b"\0")[0].decode("ascii", "replace")
            par = _u32(d, o + 0x28)
            nodes.append((nm, -1 if par == 0xFFFF else par))
    obj_name = nodes[0][0] if nodes else _cstr(d, node_off)

    block = _u32(d, 0x28) + BASE
    if block + n_sub * 20 > len(d):
        raise BinParseError("submesh array runs past end of file")

    subs = []
    for i in range(n_sub):
        e = block + i * 20
        mat_ptr = _u32(d, e)                       # 0 == no material
        dsc = _u32(d, e + 8) + BASE
        if not (0 < dsc < len(d) - 0x20):
            raise BinParseError("submesh %d: bad descriptor pointer" % i)

        tex = _cstr(d, mat_ptr + BASE + 0x18) if mat_ptr else ""

        fmt = _u32(d, dsc)
        n_idx = _u16(d, dsc + 0x08)
        n_vert = _u16(d, dsc + 0x0A)
        table = _u32(d, dsc + 0x14) + BASE
        if table + 64 > len(d):
            raise BinParseError("submesh %d: stream table out of range" % i)

        slot = [_u32(d, table + k * 8) for k in range(8)]
        if not slot[SLOT_IDX] or not slot[SLOT_POS]:
            raise BinParseError("submesh %d: missing index or position stream" % i)

        p_idx = slot[SLOT_IDX] + BASE
        p_pos = slot[SLOT_POS] + BASE
        need = max(p_idx + n_idx * 2, p_pos + n_vert * 12)
        for s, sz in ((SLOT_NRM, 12), (SLOT_UV, 8), (SLOT_COL, 4)):
            if slot[s]:
                need = max(need, slot[s] + BASE + n_vert * sz)
        if need > len(d):
            raise BinParseError("submesh %d: streams run past end of file" % i)

        idx = struct.unpack_from("<%dH" % n_idx, d, p_idx)
        if n_idx and max(idx) >= n_vert:
            raise BinParseError("submesh %d: index %d with %d vertices"
                                % (i, max(idx), n_vert))

        rigid_node = _u32(d, e + 16)

        subs.append({
            "index": i,
            "fmt": fmt,
            "lod": lod_of(fmt),
            "texture": tex,
            "has_material": bool(mat_ptr),
            "skinned": bool(slot[SLOT_SKIN]),
            "rigid_node": rigid_node,
            "indices": idx,
            "n_vert": n_vert,
            "vb_key": slot[SLOT_POS],          # shared buffers pool on this
            "p_skin": slot[SLOT_SKIN] + BASE if slot[SLOT_SKIN] else 0,
            "p_pos": p_pos,
            "p_nrm": slot[SLOT_NRM] + BASE if slot[SLOT_NRM] else 0,
            "p_uv": slot[SLOT_UV] + BASE if slot[SLOT_UV] else 0,
            "p_col": slot[SLOT_COL] + BASE if slot[SLOT_COL] else 0,
        })

    # Collision hulls: no material, no UVs, no normals. The game never draws them.
    for s in subs:
        s["collision"] = (not s["has_material"] and not s["p_uv"] and not s["p_nrm"]) \
            or is_utility_texture(os.path.splitext(s["texture"])[0])

    # Skinning needs the node array to resolve bone pointers.
    mat_base = _u32(d, 0x14)
    name_base = _u32(d, 0x1C)
    stride = (name_base - mat_base) // len(nodes) if nodes else 0
    for s in subs:
        s["skin"] = (parse_skin(d, s, mat_base, stride, len(nodes))
                     if (s["p_skin"] and stride > 0) else None)

    return d, obj_name, subs, nodes


def parse_skin(d, sub, mat_base, stride, n_nodes):
    """-> [[(bone_index, weight), ...]] with one entry per vertex, or None."""
    s = sub["p_skin"]
    n_sets = _u32(d, s + 4)
    if not (0 < n_sets < 4096):
        return None
    arr = _u64(d, s + 8) + BASE

    out = [None] * sub["n_vert"]
    for k in range(n_sets):
        e = arr + k * 24
        if e + 24 > len(d):
            return None
        n_bone = _u32(d, e)
        p_bone = _u64(d, e + 4) + BASE
        n_rng = _u32(d, e + 12)
        p_rng = _u64(d, e + 16) + BASE
        if not (0 < n_bone < 64) or p_bone + n_bone * 8 > len(d):
            return None

        bones = []
        for j in range(n_bone):
            bi = (_u64(d, p_bone + j * 8) - mat_base) // stride
            if not (0 <= bi < n_nodes):
                return None
            bones.append(bi)

        for r in range(n_rng):
            ro = p_rng + r * 16
            if ro + 16 > len(d):
                return None
            first = _u32(d, ro)
            count = _u32(d, ro + 4)
            p_w = _u64(d, ro + 8) + BASE
            if first + count > sub["n_vert"] or p_w + n_bone * 4 > len(d):
                return None
            w = struct.unpack_from("<%df" % n_bone, d, p_w)
            pairs = [(bones[j], w[j]) for j in range(n_bone) if w[j] > 1e-6]
            for v in range(first, first + count):
                out[v] = pairs

    if any(x is None for x in out):
        return None                     # ranges must tile the whole vertex list
    return out


def read_vb(d, s):
    n = s["n_vert"]
    pos = [struct.unpack_from("<3f", d, s["p_pos"] + v * 12) for v in range(n)]
    nrm = ([struct.unpack_from("<3f", d, s["p_nrm"] + v * 12) for v in range(n)]
           if s["p_nrm"] else None)
    uv = ([struct.unpack_from("<2f", d, s["p_uv"] + v * 8) for v in range(n)]
          if s["p_uv"] else None)
    col = None
    if s["p_col"]:
        raw = d[s["p_col"]: s["p_col"] + n * 4]
        col = [(raw[i * 4] / 255.0, raw[i * 4 + 1] / 255.0,
                raw[i * 4 + 2] / 255.0, raw[i * 4 + 3] / 255.0) for i in range(n)]
    return pos, nrm, uv, col


def read_vb_indices(d, s, wanted):
    """Like read_vb(), but only for the specific LOCAL indices in `wanted`
    (in that order) - needed once a fragment (see split_connected) only
    uses a handful of vertices out of a large shared buffer. Reading the
    whole buffer per fragment would be O(fragments x buffer size); a room
    with a submesh that splits into 60 islands (measured: b1__pa_area_b's
    "B1_YellowWall02") made that distinction matter, not just a theoretical
    concern."""
    pos = [struct.unpack_from("<3f", d, s["p_pos"] + v * 12) for v in wanted]
    nrm = ([struct.unpack_from("<3f", d, s["p_nrm"] + v * 12) for v in wanted]
           if s["p_nrm"] else None)
    uv = ([struct.unpack_from("<2f", d, s["p_uv"] + v * 8) for v in wanted]
          if s["p_uv"] else None)
    col = None
    if s["p_col"]:
        col = []
        for v in wanted:
            raw = d[s["p_col"] + v * 4: s["p_col"] + v * 4 + 4]
            col.append((raw[0] / 255.0, raw[1] / 255.0, raw[2] / 255.0, raw[3] / 255.0))
    return pos, nrm, uv, col


# ============================================================================
#  GS_WORLD.BIN FORMAT  (baked static level geometry, legacy pre-Aspyr export)
# ============================================================================
#
#  Every playable area (worlds\<world>\<area>\, e.g. "a1", "f3b") bakes its
#  whole visible static geometry into ONE combined mesh, exported by a much
#  older tool than the per-actor .bin above: same "raw memory dump" idea, but
#  a different struct layout, a versioned text header instead of a bare magic
#  number, and *absolute* file offsets (no +8 BASE shift - checked and it is
#  genuinely absolute, not an off-by-8 coincidence).
#
#  Measured against two independent files, a1__gs_world.bin (9 objects, 19
#  submeshes on the one with geometry) and a1_arena_test__gs_world.bin (10
#  objects, 21 submeshes): every triangle index is in range on both files
#  with zero exceptions, and plotting the raw positions top-down reproduces
#  the arena's octagonal floor plan on both. That is the acceptance test for
#  this format (METHODOLOGY.md §3/§5) - not a byte-for-byte "consume exactly
#  N of N", because large stretches between the streams are unexplained
#  scratch/allocator padding (see TODO below), but every stream that IS read
#  is internally consistent and geometrically sane on both samples.
#
#  HEADER
#    +0x00  char[]  "Bin file Version 16Mar01.3", null-terminated, then
#                    padded with 0xCD (MSVC debug "clean memory" fill - NOT
#                    the 0xAA seen in the per-actor .bin: a different, older
#                    build tool wrote this file)
#    +0x44  u16     object count
#
#  OBJECT ARRAY, stride 0x248 (584) bytes, starting at +0x4C:
#    +0x00  4x4 f32  transform matrix (identity on the object that actually
#                     has geometry; every other entry is an empty Maya
#                     leftover node - "polySurface239" etc - with submesh
#                     count 0. Their own matrix is near-identity but not
#                     exactly, i.e. real transform data with no mesh to use it)
#    +0x40  u32      submesh count       (0 for the empty nodes above)
#    +0x44  u32      -> submesh array     (absolute file offset)
#    +0x58  char[]   object name, asciiz
#
#  SUBMESH ARRAY, stride 8 bytes:
#    +0x00  u32      -> material   (0 = none)
#    +0x04  u32      -> mesh descriptor
#
#  MATERIAL, stride 280 - same stride as the modern format's MATERIAL record,
#  but a different internal layout:
#    +0x00  u32      id
#    +0x04  char[]   texture name, asciiz ("wall18.tga"). Can be an empty
#                     string (submeshes with no visible texture - collision
#                     geometry, going by the naming of the loose "_collision.
#                     dds" / "_portal.dds" utility textures sitting in every
#                     level's textures/ folder, though that link is a
#                     naming inference, not something read from the file)
#
#  MESH DESCRIPTOR, stride 0x148 (328) bytes. Almost entirely unused/
#  uninitialised (0xCD) scratch space; the only fields that matter sit near
#  the end of the record:
#    +0x10C  u16     index count
#    +0x10E  u16     vertex count
#    +0x110  u32     -> index stream     (u16 triangle LIST, absolute offset)
#    +0x114  u32     -> position stream  (f32[3] per vertex, absolute offset)
#    +0x118  u32     (always 0 on everything seen so far)
#    +0x11C  u32     -> UV stream        (f32[2] per vertex, 0 if absent)
#
#  No normal or vertex-colour stream has been located yet - see TODO. Blender
#  falls back to its own smooth autogenerated normals, which is a reasonable
#  default for static architecture and produced no visible faceting in the
#  arena test render.
#
#  TODO (recorded gap, not a guess dressed up as a fact - METHODOLOGY.md §11):
#  two fields in the mesh descriptor are still unexplained - a repeated
#  0xa0 pointer-shaped constant at +0x12C (equal, suspiciously, to a value in
#  the file's global header - possibly a shared default/empty-array pointer,
#  not yet tested), and a packed-u16 pair (96, 16) at +0x138. Neither field
#  ever stopped a triangle from extracting correctly on either sample file,
#  so the working hypothesis is LOD/multitexture/lightmap metadata rather
#  than anything needed for base geometry - but that is a hypothesis, and
#  the next area that fails to import cleanly is the place to revisit it.
# ============================================================================

GSWORLD_MAGIC = b"Bin file Version"
GSWORLD_OBJ_OFF = 0x4C
GSWORLD_OBJ_STRIDE = 0x248
GSWORLD_MESH_STRIDE = 0x148


def _parse_bin_legacy_gsworld(d, name_hint):
    """Parse a "Bin file Version..." legacy container (gs_world.bin and the
    dozen pre-portal-split pa_area_*.bin leftovers found on b1 share this
    exact header). One of the three headers parse_bin() dispatches on - see
    that function. Returns (d, name, subs, nodes) in the exact shape the
    modern parser returns, so build_mesh() and every caller can stay generic.
    nodes is always None: this format carries no skeleton, its submeshes are
    never skinned.
    """
    if len(d) < 0x50 or not d.startswith(GSWORLD_MAGIC):
        raise BinParseError("not a legacy 'Bin file Version' container")

    obj_count = _u16(d, 0x44)
    if not (0 < obj_count < 4096):
        raise BinParseError("implausible object count: %d" % obj_count)

    subs = []
    obj_names = []
    for oi in range(obj_count):
        rec = GSWORLD_OBJ_OFF + oi * GSWORLD_OBJ_STRIDE
        if rec + GSWORLD_OBJ_STRIDE > len(d):
            raise BinParseError("object %d: record runs past end of file" % oi)

        sm_count = _u32(d, rec + 0x40)
        if sm_count == 0:
            continue                        # empty transform/locator node
        if not (0 < sm_count < 4096):
            raise BinParseError("object %d: implausible submesh count %d"
                                % (oi, sm_count))
        sm_arr = _u32(d, rec + 0x44)
        if sm_arr + sm_count * 8 > len(d):
            raise BinParseError("object %d: submesh array runs past end of file" % oi)

        obj_names.append(_cstr(d, rec + 0x58))

        for i in range(sm_count):
            e = sm_arr + i * 8
            mat_off = _u32(d, e)
            dsc = _u32(d, e + 4)
            if not (0 < dsc < len(d) - GSWORLD_MESH_STRIDE):
                raise BinParseError("object %d submesh %d: bad descriptor pointer"
                                    % (oi, i))

            tex = _cstr(d, mat_off + 4) if mat_off else ""

            n_idx = _u16(d, dsc + 0x10C)
            n_vert = _u16(d, dsc + 0x10E)
            p_idx = _u32(d, dsc + 0x110)
            p_pos = _u32(d, dsc + 0x114)
            p_uv = _u32(d, dsc + 0x11C)

            if not p_idx or not p_pos or not n_vert:
                continue                    # empty/degenerate submesh
            need = max(p_idx + n_idx * 2, p_pos + n_vert * 12)
            if p_uv:
                need = max(need, p_uv + n_vert * 8)
            if need > len(d):
                raise BinParseError("object %d submesh %d: streams run past end of file"
                                    % (oi, i))

            idx = struct.unpack_from("<%dH" % n_idx, d, p_idx)
            if n_idx and max(idx) >= n_vert:
                raise BinParseError("object %d submesh %d: index %d with %d vertices"
                                    % (oi, i, max(idx), n_vert))

            subs.append({
                "index": len(subs),
                "fmt": 0,
                "lod": 0,                     # this format has no LOD tagging
                "texture": tex,
                "has_material": bool(mat_off),
                "skinned": False,
                "rigid_node": 0,
                "indices": idx,
                "n_vert": n_vert,
                "vb_key": p_pos,             # each submesh owns its own buffer here
                "p_skin": 0,
                "p_pos": p_pos,
                "p_nrm": 0,
                "p_uv": p_uv,
                "p_col": 0,
                "skin": None,
            })

    for s in subs:
        s["collision"] = (not s["has_material"] and not s["p_uv"]) \
            or is_utility_texture(os.path.splitext(s["texture"])[0])

    name = obj_names[0] if obj_names else name_hint
    return d, name, subs, None


def parse_bin(path):
    """Dispatch on header: the modern per-actor/room format (0xAAAAAAAA),
    or the older "Bin file Version..." container (gs_world.bin, and a
    measured dozen pre-portal-split leftovers on b1 that share its header).
    A file that matches neither raises BinParseError instead of guessing -
    encountering it should mean 'I don't understand this yet', not a crash
    with a confusing message three frames deep in the wrong parser."""
    with open(path, "rb") as f:
        d = f.read()

    if d[:4] == b"\xAA\xAA\xAA\xAA":
        return _parse_bin_modern(d)
    if d[:len(GSWORLD_MAGIC)] == GSWORLD_MAGIC:
        name_hint = os.path.splitext(os.path.basename(path))[0]
        return _parse_bin_legacy_gsworld(d, name_hint)
    raise BinParseError("not a recognised .bin container (unknown header)")


# ============================================================================
#  Asset lookup in the unpacked tree
# ============================================================================
#  A .bin names its texture "O_A_CratesC.tga", but on disk it is a .dds (a few
#  are .jpg/.tif/.png). Beside the model live materials/ and textures/.
#  data1/ is the patch overlay and OVERRIDES data/, exactly as the game loads it.
# ============================================================================

_INDEX_CACHE = {}


def find_data_root(start_dir):
    cur = os.path.abspath(start_dir)
    fallback = None
    for _ in range(14):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        try:
            entries = set(e.lower() for e in os.listdir(parent))
        except OSError:
            entries = set()
        if "data" in entries and "data1" in entries:
            return parent                       # the _unpacked folder
        if "data" in entries and fallback is None and "_unpacked" in parent.lower():
            fallback = parent
        cur = parent
    return fallback


def build_index(root):
    if root in _INDEX_CACHE:
        return _INDEX_CACHE[root]

    tex, mats = {}, {}
    for dirpath, _dn, filenames in os.walk(root):
        overlay = (os.sep + "data1" + os.sep) in (dirpath + os.sep).lower()
        for fn in filenames:
            stem, ext = os.path.splitext(fn)
            key, ext = stem.lower(), ext.lower()
            if ext in TEX_EXTS:
                if overlay or key not in tex:
                    tex[key] = os.path.join(dirpath, fn)
            elif ext == ".mat":
                # Measured: 9 of the longothug*.mat files ship double-named,
                # "longothug1_body.mat.mat" - one splitext leaves the key as
                # "longothug1_body.mat", which locate() (stem + ".mat") never
                # matches. Strip every trailing .mat, not just one.
                while key.endswith(".mat"):
                    key = key[:-4]
                if overlay or key not in mats:
                    mats[key] = os.path.join(dirpath, fn)

    print("[SWBH] indexed %s: %d textures, %d materials" % (root, len(tex), len(mats)))
    _INDEX_CACHE[root] = (tex, mats)
    return tex, mats


def locate(stem, model_dir, subdir, exts, index):
    for folder in (os.path.join(model_dir, subdir),
                   os.path.join(os.path.dirname(model_dir), subdir),
                   model_dir):
        for ext in exts:
            cand = os.path.join(folder, stem + ext)
            if os.path.isfile(cand):
                return cand
    return index.get(stem.lower())


_BIN_INDEX_CACHE = {}


def build_bin_index(root):
    """Map an actor/prop CLASS name (e.g. "O_B_HorizontalDoorC") to its .bin
    path, by stem, across the whole unpacked tree. Used to place .lvt ACTOR
    entries - unlike room geometry these are ordinary single-object meshes in
    local space, positioned the same way as a LIGHT entry (obj.location/
    rotation_quaternion from POS/ROT). A second, separate os.walk from
    build_index()'s texture/material one - simpler and safe to add without
    touching the existing 2-tuple every call site of build_index() expects."""
    if root in _BIN_INDEX_CACHE:
        return _BIN_INDEX_CACHE[root]

    bins = {}
    for dirpath, _dn, filenames in os.walk(root):
        overlay = (os.sep + "data1" + os.sep) in (dirpath + os.sep).lower()
        for fn in filenames:
            stem, ext = os.path.splitext(fn)
            if ext.lower() != ".bin":
                continue
            key = stem.lower()
            if overlay or key not in bins:
                bins[key] = os.path.join(dirpath, fn)

    print("[SWBH] indexed %s: %d .bin models (for actor/prop placement)"
          % (root, len(bins)))
    _BIN_INDEX_CACHE[root] = bins
    return bins


def _strip_inline_comment(value):
    # Comments in ATS/ASET are mostly //, but quoted strings are not used in
    # the shipped descriptors. Keep this deliberately conservative.
    if "//" in value:
        return value.split("//", 1)[0].rstrip()
    return value.strip()


def parse_kv_text(path, allow_equals=True):
    """Read a simple engine descriptor as a case-preserving key/value dict."""
    out = {}
    try:
        with open(path, "r", encoding="latin1", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("//") or line.startswith("/"):
                    continue
                if allow_equals and "=" in line:
                    k, v = line.split("=", 1)
                elif ":" in line:
                    k, v = line.split(":", 1)
                else:
                    continue
                k = k.strip()
                v = _strip_inline_comment(v)
                if k:
                    out[k] = v
    except OSError:
        pass
    return out


def parse_ats(path):
    """Parse an ATS actor/template descriptor.

    Values remain strings intentionally: the original files contain a mix of
    strings, integers, floats, flag strings and symbolic names. Numeric
    coercion is exposed separately so no information is lost.
    """
    path = os.path.abspath(path)
    if path in _ATS_CACHE:
        return _ATS_CACHE[path]
    raw = parse_kv_text(path, allow_equals=True)
    typed = {}
    for k, v in raw.items():
        low = v.strip()
        if re.fullmatch(r"[-+]?\d+", low):
            try:
                typed[k] = int(low)
                continue
            except ValueError:
                pass
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+\.\d+)(?:[eE][-+]?\d+)?", low):
            try:
                typed[k] = float(low)
                continue
            except ValueError:
                pass
        typed[k] = v
    out = {"path": path, "raw": raw, "values": typed}
    _ATS_CACHE[path] = out
    return out


def parse_aset(path):
    """Parse an ASET animation-set descriptor into named ABIN references."""
    path = os.path.abspath(path)
    if path in _ASET_CACHE:
        return _ASET_CACHE[path]
    entries = []
    includes = []
    try:
        with open(path, "r", encoding="latin1", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("//") or line.startswith("/"):
                    continue
                m = re.match(r"#include\s+\"([^\"]+)\"", line, re.I)
                if m:
                    includes.append(m.group(1))
                    continue
                m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\S+)(?:\s+(.+?))?\s*$", line)
                if not m:
                    continue
                name, ref, extra = m.groups()
                entries.append({"name": name, "path": ref, "extra": extra or ""})
    except OSError:
        pass
    out = {"path": path, "entries": entries, "includes": includes}
    _ASET_CACHE[path] = out
    return out


def _float3(text):
    vals = [float(x) for x in text.split()[:3]]
    return tuple(vals) if len(vals) == 3 else None


def parse_col(path):
    """Parse the text collision format; report an opaque binary variant."""
    path = os.path.abspath(path)
    if path in _COL_CACHE:
        return _COL_CACHE[path]
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return {"path": path, "format": "missing", "boxes": [], "spheres": [], "binary": False}
    if data.startswith(b"BBOXES:"):
        text = data.decode("latin1", errors="replace")
        boxes = []
        spheres = []
        dcs = []
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"BBOX:\s*MIN:\s+([^\s]+\s+[^\s]+\s+[^\s]+)\s+MAX:\s+(.+)$", line, re.I)
            if m:
                mn = _float3(m.group(1)); mx = _float3(m.group(2))
                if mn and mx:
                    boxes.append({"min": mn, "max": mx})
                continue
            m = re.match(r"DCS:\s*(\S+)\s+OFF:\s+([^\n]+?)\s+RAD:\s*([-+\d.eE]+)", line, re.I)
            if m:
                off = _float3(m.group(2))
                if off:
                    dcs.append({"bone": m.group(1), "offset": off, "radius": float(m.group(3))})
                continue
            m = re.match(r"NUMSPHERE:\s*(\d+)", line, re.I)
            if m:
                continue
        out = {"path": path, "format": "text", "boxes": boxes, "spheres": spheres,
               "dcs": dcs, "binary": False}
    else:
        out = {"path": path, "format": "binary", "boxes": [], "spheres": [],
               "dcs": [], "binary": True, "size": len(data), "header": data[:32].hex(" ")}
    _COL_CACHE[path] = out
    return out


def parse_lvr(path):
    """Parse LVR actor placement data, including the first actor block."""
    path = os.path.abspath(path)
    if path in _LVR_CACHE:
        return _LVR_CACHE[path]
    try:
        with open(path, "r", encoding="latin1", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return {"version": None, "actors": [], "raw_lines": []}
    actors = []
    cur = None
    for line in lines:
        m = _LVT_ACTOR_RE.search(line)
        if m:
            cur = {
                "name": m.group(1), "cls": m.group(2),
                "pos": tuple(float(x) for x in m.groups()[2:5]),
                "rot": tuple(float(x) for x in m.groups()[5:9]),
                "id": int(m.group(10)), "flags": int(m.group(11)), "areas": []}
            actors.append(cur)
            continue
        if line.strip().startswith("AREAS:") and cur is not None:
            cur["areas"] = [int(x) for x in line.split()[1:] if x.lstrip("-").isdigit()]
    out = {"version": lines[0].strip() if lines else None, "actors": actors, "raw_lines": lines}
    _LVR_CACHE[path] = out
    return out


def find_related(root, name, ext, preferred_dir=None):
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    candidates = []
    if preferred_dir:
        candidates.extend([
            os.path.join(preferred_dir, stem + ext),
            os.path.join(preferred_dir, stem, stem + ext),
        ])
    if root and os.path.isdir(root):
        for base, _dirs, files in os.walk(root):
            for fn in files:
                if os.path.splitext(fn)[0].lower() == stem and os.path.splitext(fn)[1].lower() == ext:
                    candidates.append(os.path.join(base, fn))
                    break
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def resolve_ats_asset(ats_path, ats, root):
    values = ats.get("values", {})
    model = values.get("modelFile", "")
    collision = values.get("collisionFile", "")
    aset = values.get("animSetFile", "")
    model_path = values.get("modelPath", "")
    base_dir = os.path.dirname(ats_path)
    search_dirs = [base_dir]
    if model_path:
        rel = model_path.replace("\\", os.sep).replace("/", os.sep)
        if root:
            search_dirs.append(os.path.join(root, rel.lstrip(os.sep)))
        search_dirs.append(os.path.join(base_dir, rel))
    out = {
        "bin": find_related(root, model, ".bin", preferred_dir=search_dirs[-1]),
        "col": find_related(root, collision, ".col", preferred_dir=search_dirs[-1]),
        "aset": find_related(root, aset, ".aset", preferred_dir=search_dirs[-1]),
    }
    return out


def parse_mat(path):
    raw = parse_kv_text(path, allow_equals=False)
    return {k.lower(): v for k, v in raw.items()}


# ============================================================================
#  Materials
# ============================================================================

def make_material(name, tex_path, shader_type, use_vcol, normal_path=None, metadata=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)

    st = (shader_type or "").lower()
    unlit = ("unlit" in st) or ("skydome" in st) or ("glow" in st)
    cutout = "cutout" in st
    additive = "additive" in st or "modulate" in st  # both fall back to the
                                                       # same BLEND approximation
                                                       # below - neither Add nor
                                                       # true Multiply has a
                                                       # simple cross-version
                                                       # Blender blend_method
    glow = "glow" in st
    reflective = "reflectionmap" in st
    blended = ("transparency" in st) or ("alphablend" in st) or additive or glow

    if unlit:
        shader = nt.nodes.new("ShaderNodeEmission")
        shader.inputs["Strength"].default_value = 1.0
        color_in = shader.inputs["Color"]
        alpha_in = None
    else:
        shader = nt.nodes.new("ShaderNodeBsdfPrincipled")
        if "Roughness" in shader.inputs:
            # Reflective (measured: Jango's chrome-armor materials, "Shader_
            # ReflectionMap1/2" with an EnvironmentMap key) - approximated
            # with Metallic/Roughness rather than wiring the env-map texture
            # through an actual reflection-vector projection. Blender's UV
            # convention for that projection was never checked against this
            # engine's, so a metal tweak that just uses Blender's own
            # environment reflections is the safer approximation, not a
            # reverse-engineered match to the original chrome look.
            shader.inputs["Roughness"].default_value = 0.15 if reflective else 0.6
        if reflective and "Metallic" in shader.inputs:
            shader.inputs["Metallic"].default_value = 0.85
        color_in = shader.inputs["Base Color"]
        alpha_in = shader.inputs.get("Alpha")
    shader.location = (420, 0)
    nt.links.new(shader.outputs[0], out.inputs["Surface"])

    src = None
    alpha_src = None

    if tex_path and os.path.isfile(tex_path):
        img = bpy.data.images.load(tex_path, check_existing=True)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.location = (-200, 0)
        tex.interpolation = "Closest"           # keep the 2002 pixels crisp
        src = tex.outputs["Color"]
        alpha_src = tex.outputs["Alpha"]

    # Baked vertex lighting. The level's light lives in the vertex colours, so
    # it has to multiply the diffuse or the whole map renders flat and dead.
    if use_vcol:
        vc = nt.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = "Col"
        vc.location = (-200, -280)
        if src is None:
            src = vc.outputs["Color"]
        else:
            mix = nt.nodes.new("ShaderNodeMixRGB")
            mix.blend_type = "MULTIPLY"
            mix.inputs["Fac"].default_value = 1.0
            mix.location = (120, -80)
            nt.links.new(src, mix.inputs["Color1"])
            nt.links.new(vc.outputs["Color"], mix.inputs["Color2"])
            src = mix.outputs["Color"]

    if src is not None:
        nt.links.new(src, color_in)

    # Bump/normal map: measured on B1_Door02_a - no .mat file in a 798-file
    # audit of the whole tree carries any bump/normal key at all, but a
    # "B1_Door02_aNormalMap.dds" sits right next to "B1_Door02_a.dds" in the
    # same textures folder. So this is resolved by NAMING CONVENTION
    # (<diffuse_stem>NormalMap), not a .mat key - make_submesh_material()
    # does that lookup and passes the result in here. Only wired for the lit
    # (Principled) branch; Emission has no Normal input to plug into.
    if normal_path and not unlit and "Normal" in shader.inputs:
        nimg = bpy.data.images.load(normal_path, check_existing=True)
        nimg.colorspace_settings.name = "Non-Color"
        ntex = nt.nodes.new("ShaderNodeTexImage")
        ntex.image = nimg
        ntex.location = (-200, -450)
        ntex.interpolation = "Closest"
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.location = (120, -450)
        nt.links.new(ntex.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], shader.inputs["Normal"])

    if alpha_src is not None and (cutout or blended):
        if alpha_in is not None:
            nt.links.new(alpha_src, alpha_in)
        else:
            tr = nt.nodes.new("ShaderNodeBsdfTransparent")
            tr.location = (420, -220)
            mix = nt.nodes.new("ShaderNodeMixShader")
            mix.location = (560, 0)
            nt.links.new(alpha_src, mix.inputs["Fac"])
            nt.links.new(tr.outputs[0], mix.inputs[1])
            nt.links.new(shader.outputs[0], mix.inputs[2])
            nt.links.new(mix.outputs[0], out.inputs["Surface"])

    if cutout:
        mat.blend_method = "CLIP"
    elif blended:
        # Blender has no simple cross-version "Add" blend_method to target -
        # this is alpha-blend as a stand-in, not true additive. Recorded
        # simplification: found on the star skydome layer (Shader_
        # AdditiveUnlit) via d2_skydome_stars.mat; the star texture's own
        # alpha still has to carry the "empty sky = invisible" cutout for
        # this to look right, and that hasn't been checked against a real
        # texture.
        mat.blend_method = "BLEND"

    mat["swbh_shader_type"] = shader_type or ""
    if metadata:
        for _k, _v in metadata.items():
            try:
                mat["swbh_mat_" + str(_k)] = _v
            except Exception:
                pass
    return mat


def make_submesh_material(s, obj_name, model_dir, index, opt, have_col, missing):
    tex_index, mat_index = index
    stem = os.path.splitext(s["texture"])[0]

    info = {}
    tex_file = None
    normal_file = None
    mat_file = None
    if opt["textures"] and stem:
        mat_file = locate(stem, model_dir, "materials", (".mat",), mat_index)
        info = parse_mat(mat_file) if mat_file else {}
        dstem = os.path.splitext(info.get("diffusemap", s["texture"]))[0]
        tex_file = locate(dstem, model_dir, "textures", TEX_EXTS, tex_index)
        if not tex_file:
            missing.add(dstem)
        normal_file = locate(dstem + "NormalMap", model_dir, "textures",
                             TEX_EXTS, tex_index)

    mat_obj = make_material("%s_%d_%s" % (obj_name, s["index"], stem or "none"),
                            tex_file, info.get("shadertype", ""),
                            have_col and opt["vcol"], normal_file, metadata=info)
    if mat_file:
        mat_obj["swbh_source_mat"] = mat_file
    if tex_file:
        mat_obj["swbh_source_texture"] = tex_file
    return mat_obj


# ============================================================================
#  Mesh building
# ============================================================================

def split_connected(subs):
    """Partition submeshes into connected components: two submeshes are
    connected only if they reference at least one common LOCAL vertex index
    within the same vb_key. Sharing a vb_key alone means nothing by itself -
    it is just shared *storage*, e.g. a whole room's walls/floor/props
    packed into one buffer for space, and turned out (measured on
    b1__pa_area_b, 91 visible submeshes) to have ZERO index overlap between
    any two of them - every single one is its own disconnected island even
    though they all technically "share" the same vb_key.

    Deliberately NOT taken further to individual-vertex/triangle level: that
    was tried and measured wrong. Even ONE continuous architectural wall
    submesh (96 triangles) turns out to split into 44 "islands" under
    shared-vertex-index connectivity - this game's export does not weld
    adjacent triangles' vertex indices together at all, so index-sharing is
    not a usable "same physical object" signal here; going that far just
    shatters every room into thousands of unusable fragments. Splitting at
    the submesh (material) boundary remains the useful, safe granularity.

    Returns a list of lists of submeshes, in stable order (first submesh's
    original position decides a component's place in the output).
    """
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s in subs:
        parent.setdefault(s["index"], s["index"])

    by_key = {}
    for s in subs:
        by_key.setdefault(s["vb_key"], []).append(s)

    for group in by_key.values():
        touched_by = {}                 # local vertex index -> first submesh index seen
        for s in group:
            for li in set(s["indices"]):
                if li in touched_by:
                    union(s["index"], touched_by[li])
                else:
                    touched_by[li] = s["index"]

    order = []
    comps = {}
    for s in subs:
        r = find(s["index"])
        if r not in comps:
            comps[r] = []
            order.append(r)
        comps[r].append(s)
    return [comps[r] for r in order]



def build_rigid_mechanism_parts(d, name, subs, model_dir, index, opt, missing,
                                nodes, tracks, actor_location=(0.0, 0.0, 0.0),
                                actor_rotation=(1.0, 0.0, 0.0, 0.0),
                                collection=None, fps=30.0, frame_offset=1):
    """Build rigid mechanism submeshes as independent plain objects.

    Rigid mechanisms (doors, gates, some switches/elevators) are not skinned
    character meshes. A submesh's +0x10 id selects the node that drives it.
    The geometry needs the node's frame-0 translation baked into its vertices,
    while animation should apply only the *delta* from frame 0. Using an
    Armature here applies the same hierarchy transform a second time and is
    especially error-prone for doors. This path deliberately keeps the game
    node transform as ordinary object transforms instead.
    """
    if not nodes or not tracks:
        return [], 0

    by_node = {}
    for sub in subs:
        if sub.get("collision"):
            continue
        by_node.setdefault(sub.get("rigid_node", 0), []).append(sub)

    made = []
    max_frames = 0
    actor_q = Quaternion(actor_rotation)
    sc = opt.get("scale", 1.0)
    actor_loc = Vector(actor_location)

    for node_idx, group in by_node.items():
        if node_idx < 0 or node_idx >= len(nodes):
            continue
        driven_name = rigid_driving_node(nodes, tracks, node_idx)
        track = tracks.get(driven_name) if driven_name else None

        # Baked frame-0 translation in the vertices. Animation object
        # translation is therefore a delta from frame 0, not the absolute
        # game-space track value.
        obj = build_mesh(d, "%s_%s" % (name, nodes[node_idx][0]), group,
                         model_dir, index, opt, missing, nodes=None,
                         rigid_offsets=rigid_rest_offsets(nodes, tracks))
        if obj is None:
            continue
        obj.location = actor_loc
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = actor_q

        obj["swbh_rigid_node"] = nodes[node_idx][0]
        if driven_name:
            obj["swbh_rigid_driver"] = driven_name
        if track:
            # Save actual frame-0 transform for diagnostics and exact delta
            # animation. No armature modifier is added.
            p0 = sample_pos(track["pos"], 0.0) if track.get("pos") else Vector((0, 0, 0))
            q0 = sample_rot(track["rot"], 0.0) if track.get("rot") else Quaternion((1, 0, 0, 0))
            obj["swbh_rigid_rest_pos"] = tuple(float(v) for v in p0)
            obj["swbh_rigid_rest_rot"] = tuple(float(v) for v in q0)

            obj.animation_data_create()
            n_frames = max(1, int(round((track.get("duration") or 0.0) * fps)))
            # The track parser stores key lists but not a duration on every
            # track in older files; use the parent ABIN duration when supplied
            # by the caller via the special key below.
            n_frames = max(1, int(round(tracks.get("__duration__", 0.0) * fps)))
            for f in range(n_frames + 1):
                t = f * tracks.get("__duration__", 0.0) / n_frames if n_frames else 0.0
                frame = frame_offset + f
                if track.get("pos"):
                    pt = sample_pos(track["pos"], t)
                    delta = (pt - p0) * sc
                    obj.location = actor_loc + delta
                    obj.keyframe_insert("location", frame=frame)
                if track.get("rot"):
                    qt = sample_rot(track["rot"], t)
                    # Track rotation is in the actor's local mechanism space.
                    obj.rotation_quaternion = actor_q @ qt
                    obj.keyframe_insert("rotation_quaternion", frame=frame)
            max_frames = max(max_frames, n_frames)

        (collection or bpy.context.collection).objects.link(obj)
        made.append(obj)

    return made, max_frames


def build_mesh(d, name, subs, model_dir, index, opt, missing, nodes=None,
               rigid_offsets=None, rigid_matrices=None):
    verts, faces, loop_mats = [], [], []
    normals, uvs_v, cols_v = [], [], []
    skin_v = []                                  # per vertex: [(bone_index, weight)]
    mats = []
    pool = {}                                    # vb_key -> base vertex index

    have_uv = any(s["p_uv"] for s in subs)
    have_nrm = all(s["p_nrm"] for s in subs)
    have_col = any(s["p_col"] for s in subs)

    def _pool_key(sub):
        # A shared vertex buffer can contain several DIFFERENT rigid pieces.
        # Pooling only by vb_key used to make the first piece's rigid transform
        # and synthetic vertex group leak into the next one.  Keep real-skinned
        # buffers pooled normally; split synthetic rigid pieces by node id.
        synthetic_rigid = (not sub.get("skin") and nodes and len(nodes) > 1
                           and 0 <= sub.get("rigid_node", -1) < len(nodes))
        if synthetic_rigid and (rigid_matrices or rigid_offsets):
            return (sub["vb_key"], sub.get("rigid_node", 0))
        return sub["vb_key"]

    for s in subs:
        key = _pool_key(s)
        if key not in pool:
            # Read only the vertices this specific build_mesh() call actually
            # needs for this buffer - not the whole original n_vert range.
            needed = set()
            for s2 in subs:
                if _pool_key(s2) == key:
                    needed.update(s2["indices"])
            needed = sorted(needed)
            remap = {li: i for i, li in enumerate(needed)}

            pos, nrm, uv, col = read_vb_indices(d, s, needed)

            # Full rigid rest transform (translation + static orientation) is
            # authoritative when available.  Only synthetic rigid_node meshes
            # use it; smooth-skinned character buffers must stay in their
            # original bind/model space.
            rm = (rigid_matrices.get(s["rigid_node"])
                  if rigid_matrices and not s.get("skin") else None)
            if rm is not None:
                pos = [tuple(rm @ Vector(p)) for p in pos]
                if nrm:
                    try:
                        nm = rm.to_3x3().inverted().transposed()
                        nrm = [tuple((nm @ Vector(v)).normalized()) for v in nrm]
                    except Exception:
                        pass
            else:
                off = rigid_offsets.get(s["rigid_node"]) if rigid_offsets else None
                if off:
                    ox, oy, oz = off
                    pos = [(p[0] + ox, p[1] + oy, p[2] + oz) for p in pos]
            pool[key] = (len(verts), remap)
            sc = opt["scale"]
            verts.extend((p[0] * sc, p[1] * sc, p[2] * sc) for p in pos)
            normals.extend(nrm or [(0.0, 0.0, 1.0)] * len(needed))
            uvs_v.extend(uv or [(0.0, 0.0)] * len(needed))
            cols_v.extend(col or [(1.0, 1.0, 1.0, 1.0)] * len(needed))
            skin_full = s["skin"]
            if (not skin_full and nodes and len(nodes) > 1
                    and 0 <= s["rigid_node"] < len(nodes)):
                skin_full = [[(s["rigid_node"], 1.0)] for _ in range(s["n_vert"])]
            skin_v.extend((skin_full[li] for li in needed) if skin_full
                         else [None] * len(needed))
        base, remap = pool[key]

        slot = len(mats)
        mats.append(make_submesh_material(s, name, model_dir, index, opt,
                                          have_col, missing))

        idx = s["indices"]
        for t in range(0, len(idx) - 2, 3):
            a, b, c = idx[t], idx[t + 1], idx[t + 2]
            if a == b or b == c or a == c:
                continue                          # degenerate
            faces.append((base + remap[a], base + remap[b], base + remap[c]))
            loop_mats.append(slot)

    if not faces:
        return None

    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()

    for m in mats:
        me.materials.append(m)
    for i, poly in enumerate(me.polygons):
        poly.material_index = loop_mats[i]
        poly.use_smooth = True

    if have_uv:
        layer = me.uv_layers.new(name="UVMap")
        for loop in me.loops:
            u, v = uvs_v[loop.vertex_index]
            layer.data[loop.index].uv = (u, 1.0 - v) if opt["flip_v"] else (u, v)

    if have_col and opt["vcol"]:
        try:
            attr = me.color_attributes.new(name="Col", type="FLOAT_COLOR",
                                           domain="POINT")
            for i, c in enumerate(cols_v):
                attr.data[i].color = c
        except Exception as exc:
            print("[SWBH] vertex colours failed: %s" % exc)

    if have_nrm and opt["normals"]:
        try:
            me.normals_split_custom_set_from_vertices(normals)
            if hasattr(me, "use_auto_smooth"):    # Blender < 4.1
                me.use_auto_smooth = True
        except Exception as exc:
            print("[SWBH] custom normals failed: %s" % exc)

    me.validate(clean_customdata=False)
    me.update()
    obj = bpy.data.objects.new(name, me)

    if nodes and any(x for x in skin_v):
        groups = {}
        for vi, pairs in enumerate(skin_v):
            if not pairs:
                continue
            for bi, w in pairs:
                bn = nodes[bi][0]
                g = groups.get(bn)
                if g is None:
                    g = groups[bn] = obj.vertex_groups.new(name=bn)
                g.add([vi], w, "REPLACE")

    return obj


# ============================================================================
#  .lvt FORMAT  -  per-area world table (self-describing ASCII, CRLF)
# ============================================================================
#  Not guessed at - read whole (b1.lvt, 2382 lines) and confirmed section by
#  section. Sections are NUM<X>: <count> ... END<X>, each a flat list of
#  fixed-shape records. The three sections this addon uses:
#
#    WORLD: ... SKY: <name>.bin ... ENDWORLD
#      <name> is a *prefix*, not one exact file - b1.lvt says "B1_skydome.bin"
#      but the shipped files are B1_skydome0/1/2.bin (parallax layers). Case
#      does not match the files on disk either ("B1_..." vs "b1_...") - always
#      match case-insensitively.
#
#    NUMGEOM: <n> ... GEO: <file> <flag> <area_count> / AREAS: <id>... / ENDGEO
#      This is NOT a reliable "whole level" file list: cross-checked b1's 38
#      GEO entries against the 68 .bin files actually shipped next to it -
#      every GEO entry exists on disk (good), but 28 real room files on disk
#      are absent from the list (pa_nightclub_a, pa_entertainment_*, the
#      pa_area_b_wh_* warehouses...). This importer does NOT use this section
#      to decide what to load - directory scanning already found every one of
#      these rooms and is strictly more complete. Kept here only for the SKY
#      cross-check and because it is genuinely useful diagnostic context.
#
#    NUMLIGHTS: <n> ...
#      LIGHT: <name> POS: x y z ROT: w x y z      <- (w,x,y,z), NOT (x,y,z,w)
#      AREAS: <id>...                                like the binary .abin
#      AMBIENT: r g b LIGHTCOLOR: r g b a             convention. Verified
#      ENDLIGHT                                       with a physical check:
#      fed Jango's a1.lvr ROT through acos(w)*2 and got a clean 80 degrees
#      around a pure Z axis, which only comes out clean under (w,x,y,z).
#      This is a REAL answer to "is the light baked or separate": both. The
#      geometry still carries baked vertex colour (existing SLOT_COL stream),
#      but on top of that a level like b1 (Coruscant nightlife) places up to
#      several dozen real, named, coloured point lights (neon signs, uplights)
#      that are not baked into any mesh. No radius/falloff/type field was
#      found in the record - only position, rotation, ambient and colour -
#      so this importer creates plain POINT lights and lets Blender's default
#      radius/power stand in; that is a recorded gap, not a measured fact.
#
#    NUMACTORS: <n> ... ACTOR: <instance> <class> POS: x y z ROT: w x y z
#      ID: <id> FLAGS: <n> / AREAS: <id>...  / ENDACTORS
#      Same (w,x,y,z) convention as LIGHT. <class> is the asset name to look
#      up (e.g. "O_B_HorizontalDoorC", "Jango") - unlike room geometry, prop/
#      character .bin models are in LOCAL space and need this POS/ROT applied
#      as the object's transform, exactly like positioning a light. b1.lvr
#      turned out to hold the identical 135 actors as a subset of what's
#      already in b1.lvt (same NUMACTORS count, same records) - so .lvr looks
#      redundant wherever .lvt exists, and this importer reads actors from
#      .lvt only. Left unparsed for now: the second ACTOR block (NUMOVR /
#      "ACTOR INSTANCE DATA", the Class/ConvertType/Set property-list form
#      seen at the tail of a1.lvr) - it carries gameplay behaviour, not
#      placement, so it doesn't affect where or whether something is drawn.
# ============================================================================

_LVT_LIGHT_RE = re.compile(
    r"LIGHT:\s*(\S+)\s*POS:\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*"
    r"ROT:\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")
_LVT_AMBIENT_RE = re.compile(
    r"AMBIENT:\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*"
    r"LIGHTCOLOR:\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")
_LVT_ACTOR_RE = re.compile(
    r"ACTOR:\s*(\S+)\s+(\S+)\s*POS:\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*"
    r"ROT:\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*"
    r"ID:\s*(\d+)\s+FLAGS:\s*(\d+)")


def parse_lvt(path):
    """Return {"sky": str|None, "geom": {filename.lower(): [area_ids]},
    "lights": [{"name","pos","rot","ambient","lightcolor"}],
    "actors": [{"name","cls","pos","rot","id","flags"}]}.
    Missing/malformed sections are returned empty rather than raising - this
    file is a bonus enrichment, never required for an import to succeed."""
    try:
        with open(path, "r", encoding="latin1", newline="") as f:
            text = f.read()
    except OSError:
        return {"sky": None, "geom": {}, "lights": [], "actors": []}

    lines = text.split("\r\n") if "\r\n" in text else text.splitlines()

    sky = None
    for l in lines:
        m = re.match(r"\s*SKY:\s*(\S+)", l)
        if m:
            sky = os.path.splitext(m.group(1))[0]
            break

    geom = {}
    try:
        gi = next(i for i, l in enumerate(lines) if l.strip().startswith("NUMGEOM"))
        ge = next(i for i in range(gi, len(lines)) if lines[i].strip() == "ENDGEO")
        i = gi + 1
        while i < ge:
            l = lines[i].strip()
            if l.startswith("GEO:"):
                parts = l.split()
                fn = parts[1].lower()
                areas = []
                if i + 1 < ge and lines[i + 1].strip().startswith("AREAS:"):
                    areas = [int(x) for x in lines[i + 1].split()[1:] if x.lstrip("-").isdigit()]
                    i += 1
                geom[fn] = areas
            i += 1
    except StopIteration:
        pass

    lights = []
    try:
        li = next(i for i, l in enumerate(lines) if l.strip().startswith("NUMLIGHTS"))
        le = next(i for i in range(li, len(lines)) if lines[i].strip() == "ENDLIGHT")
        for i in range(li + 1, le):
            m = _LVT_LIGHT_RE.search(lines[i])
            if not m:
                continue
            am = _LVT_AMBIENT_RE.search(lines[i + 2]) if i + 2 < le else None
            lights.append({
                "name": m.group(1),
                "pos": tuple(float(x) for x in m.groups()[1:4]),
                "rot": tuple(float(x) for x in m.groups()[4:8]),   # (w,x,y,z)
                "ambient": tuple(float(x) for x in am.groups()[0:3]) if am else (0, 0, 0),
                "lightcolor": tuple(float(x) for x in am.groups()[3:7]) if am else (1, 1, 1, 1),
            })
    except StopIteration:
        pass

    actors = []
    try:
        ai = next(i for i, l in enumerate(lines) if l.strip().startswith("NUMACTORS"))
        ae = next(i for i in range(ai, len(lines)) if lines[i].strip() == "ENDACTORS")
        for i in range(ai + 1, ae):
            m = _LVT_ACTOR_RE.search(lines[i])
            if not m:
                continue
            actors.append({
                "name": m.group(1),
                "cls": m.group(2),
                "pos": tuple(float(x) for x in m.groups()[2:5]),
                "rot": tuple(float(x) for x in m.groups()[5:9]),   # (w,x,y,z)
                "id": int(m.group(10)),
                "flags": int(m.group(11)),
            })
    except StopIteration:
        pass

    world = {}
    for l in lines:
        m = re.match(r"\s*(FOGDIST|FARCLIP|FOGCOL|AMBLITE|FILLCOL|FLAGS|SUNCOLOR|SUNDIR|DEFAULTSMT|OBJECTACQUIRERADIUS|OBJECTRELEASERADIUS|ENVATS|CHEWIE_LEVEL_CLASS):\s*(.*)$", l)
        if not m:
            continue
        key, value = m.groups()
        nums = value.split()
        if key in {"FOGDIST", "FOGCOL", "AMBLITE", "FILLCOL", "SUNCOLOR", "SUNDIR"}:
            try:
                world[key.lower()] = tuple(float(x) for x in nums if re.fullmatch(r"[-+0-9.eE]+", x))
            except ValueError:
                world[key.lower()] = value
        elif key in {"FARCLIP", "OBJECTACQUIRERADIUS", "OBJECTRELEASERADIUS"}:
            try: world[key.lower()] = float(nums[0])
            except (ValueError, IndexError): world[key.lower()] = value
        elif key == "FLAGS":
            try: world[key.lower()] = int(nums[0])
            except (ValueError, IndexError): world[key.lower()] = value
        else:
            world[key.lower()] = value.strip()
    return {"sky": sky, "geom": geom, "lights": lights, "actors": actors, "world": world, "path": path}


def get_collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def add_collision_visuals(col_path, target_obj=None, collection_name="SWBH Collision", scale=1.0):
    """Create simple hidden wire collision helpers from text .col data."""
    data = parse_col(col_path)
    if data.get("binary"):
        print("[SWBH] binary collision variant not decoded: %s" % col_path)
        return []
    col = get_collection(collection_name)
    created = []

    def _empty(name, location=(0, 0, 0), display="SPHERE", radius=0.25):
        obj = bpy.data.objects.new(name, None)
        obj.empty_display_type = display
        obj.empty_display_size = radius
        obj.empty_display_size = max(0.05, radius)
        obj.location = location
        col.objects.link(obj)
        obj.hide_render = True
        obj.hide_set(True)
        if target_obj is not None:
            obj["swbh_collision_source"] = col_path
            obj["swbh_collision_target"] = target_obj.name
        created.append(obj)
        return obj

    for i, box in enumerate(data.get("boxes", [])):
        mn = Vector(box["min"]); mx = Vector(box["max"])
        center = (mn + mx) * 0.5
        size = (mx - mn)
        ob = bpy.data.objects.new("COL_BBOX_%02d" % i, None)
        ob.empty_display_type = "CUBE"
        ob.empty_display_size = max(size.length * 0.5 * scale, 0.05)
        ob.location = tuple(float(v) * scale for v in center)
        col.objects.link(ob)
        ob.hide_render = True; ob.hide_set(True)
        ob["swbh_collision_source"] = col_path
        ob["swbh_collision_min"] = tuple(mn)
        ob["swbh_collision_max"] = tuple(mx)
        created.append(ob)

    for i, item in enumerate(data.get("dcs", [])):
        ob = _empty("COL_%s_%02d" % (item["bone"], i), item["offset"], "SPHERE", item["radius"])
        ob["swbh_collision_bone"] = item["bone"]
        ob["swbh_collision_radius"] = item["radius"]
    return created


def apply_asset_metadata(obj, ats_path, ats_info, related):
    if obj is None:
        return
    values = ats_info.get("values", {})
    obj["swbh_source_ats"] = ats_path
    obj["swbh_asset_label"] = str(values.get("label", obj.name))
    for key in ("chewieClass", "modelFile", "modelPath", "collisionFile", "animSetFile",
                "soundTable", "inventory", "propFlags", "controlFlags"):
        if key in values:
            obj["swbh_ats_" + key] = values[key]
    if related.get("col"):
        obj["swbh_collision_file"] = related["col"]
    if related.get("aset"):
        obj["swbh_animation_set"] = related["aset"]
    if related.get("bin"):
        obj["swbh_model_file"] = related["bin"]


def apply_lvt_world_settings(path, world_data, scale):
    scene = bpy.context.scene
    scene["swbh_source_lvt"] = path
    for key, value in world_data.items():
        try:
            scene["swbh_world_" + key] = value
        except Exception:
            pass
    w = bpy.data.worlds.get(os.path.splitext(os.path.basename(path))[0] + "_World")
    if w is None:
        w = bpy.data.worlds.new(os.path.splitext(os.path.basename(path))[0] + "_World")
    scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    if bg:
        amb = world_data.get("amblite")
        if isinstance(amb, (tuple, list)) and len(amb) >= 3:
            bg.inputs["Color"].default_value = (float(amb[0]), float(amb[1]), float(amb[2]), 1.0)
        strength = float(max(0.0, sum(amb) / 3.0)) if isinstance(amb, (tuple, list)) and amb else 0.3
        bg.inputs["Strength"].default_value = strength



# ============================================================================
#  .abin FORMAT  -  skeleton bind pose and animation
# ============================================================================
#
#  +0x00  u32  0xABADDEED
#  +0x08  f32  duration, seconds
#  +0x0C  u32  track count
#  +0x10  u32  -> track array (ABSOLUTE offsets here, unlike .bin)
#
#  TRACK, stride 64:
#    +0x00  u32       flags   0x80 = quantised to int16
#                             0x40 / 0x02 = translation
#                             0x20 / 0x01 = rotation
#    +0x04  char[32]  bone name
#    +0x24  f32       duration
#    +0x28  u16       key count
#    +0x2C  u32       -> times   (u16 each, unit = 1/128 s)
#    +0x34  u32       -> values
#    +0x3E  u16       fractional bits of the int16 quantisation
#                     13 for translation (/8192), 14 for rotation (/16384)
#
#  Values: float32[4] when unquantised, otherwise int16[3] for translation and
#  int16[4] for rotation (quaternion, x y z w). Verified: 9876 quaternions in
#  jango_standidle01 all come out unit length to within 1.1e-4.
#
#  A *_base.abin holds one key per track at t=0 in float32: that is the bind
#  pose. Ordinary animations only carry the bones that actually move.
# ============================================================================

ABIN_MAGIC = 0xABADDEED
TIME_UNIT = 128.0                 # ticks per second

# The quantisation bit is NOT a separate universal flag (0x80 was a guess that
# happened to survive on the samples that only ever used the "raw" bits). It is
# encoded per-channel, inside the very bits that mark the channel's type:
#   translation: 0x40 = quantised, 0x02 = raw
#   rotation:    0x20 = quantised, 0x01 = raw
# Confirmed on jango_crouchbreath's root_joint: pos track has flags=0x40 (no
# 0x80 at all) yet is genuinely quantised int16 data - reading it as raw f32
# produced garbage on the order of 1e38 and, on other clips, ran the reader
# past the end of the track's byte range ("data out of range").
F_TRANS_QUANT = 0x40
F_TRANS_RAW = 0x02
F_ROT_QUANT = 0x20
F_ROT_RAW = 0x01
F_TRANS = F_TRANS_QUANT | F_TRANS_RAW
F_ROT = F_ROT_QUANT | F_ROT_RAW

# When the file's fractional-bits field is 0, that means "use this channel's
# default", not "scale = 1". Defaults per the format notes above (§ .abin
# header): 13 bits for translation (/8192), 14 for rotation (/16384).
DEFAULT_BITS_TRANS = 13
DEFAULT_BITS_ROT = 14


def _parse_abin_modern(d):
    duration = struct.unpack_from("<f", d, 0x08)[0]
    count = _u32(d, 0x0C)
    arr = _u32(d, 0x10)
    if arr + count * 64 > len(d):
        raise BinParseError("track array runs past end of file")

    tracks = {}
    for i in range(count):
        o = arr + i * 64
        flags = _u32(d, o)
        name = d[o + 4:o + 0x24].split(b"\0")[0].decode("ascii", "replace")
        keys = _u16(d, o + 0x28)
        p_t = _u32(d, o + 0x2C)
        p_v = _u32(d, o + 0x34)
        bits = _u16(d, o + 0x3E)

        is_trans = bool(flags & F_TRANS)
        is_rot = bool(flags & F_ROT)
        if not (is_trans or is_rot):
            continue

        quant = bool(flags & (F_TRANS_QUANT if is_trans else F_ROT_QUANT))
        eff_bits = bits or (DEFAULT_BITS_TRANS if is_trans else DEFAULT_BITS_ROT)
        scale = float(1 << eff_bits) if quant else 1.0
        n_comp = 3 if is_trans else 4
        step = (2 * n_comp) if quant else 16

        if p_v + keys * step > len(d) or p_t + keys * 2 > len(d):
            raise BinParseError("track '%s' data out of range" % name)

        out = []
        for k in range(keys):
            t = _u16(d, p_t + k * 2) / TIME_UNIT
            if quant:
                raw = struct.unpack_from("<%dh" % n_comp, d, p_v + k * step)
                val = tuple(x / scale for x in raw)
            else:
                val = struct.unpack_from("<4f", d, p_v + k * step)[:n_comp]
            out.append((t, val))

        slot = tracks.setdefault(name, {})
        slot["pos" if is_trans else "rot"] = out

    return duration, tracks


# ----------------------------------------------------------------------------
# Legacy container: "ABin file Version 18Mar00.001" (an older exporter -
# jango_drawgun.abin and presumably a handful of similar prop-interaction
# clips use this instead of the 0xABADDEED format).
#
#   +0x00  char[64]  version string, null-padded ("ABin file Version ...")
#   +0x40  u32       unknown, seen as 0
#   +0x44  f32       duration, seconds
#   +0x48  u32       track count
#   +0x4C  u32       -> track array (absolute offset)
#
#   TRACK, stride 56 (8 bytes shorter than the modern 64):
#     +0x00  u32       flags        same bits as the modern format
#                                    (0x02/0x40 translation, 0x01/0x20 rotation)
#     +0x04  char[32]  bone name
#     +0x24  f32       duration (matches the clip duration)
#     +0x28  u16       key count
#     +0x2A  u16       unknown, always 1 in the one file seen so far
#     +0x2C  u32       -> times    float32 SECONDS directly (not u16 ticks!)
#     +0x30  u32       -> values   float32[3 or 4], tightly packed - no
#                                   quantised variant has been observed, so we
#                                   refuse rather than guess if that bit shows up
#     +0x34  u32       unknown, always 1 in the one file seen so far
#
#   Verified on jango_drawgun.abin: all 45 tracks decode with the time array
#   starting at 0.0, ending exactly at the track's own duration, and strictly
#   increasing; both "unknown" fields constant at 1 across every track.
# ----------------------------------------------------------------------------

LEGACY_MAGIC = b"ABin file Version"


def _parse_abin_legacy(d):
    duration = struct.unpack_from("<f", d, 0x44)[0]
    count = _u32(d, 0x48)
    arr = _u32(d, 0x4C)
    if arr + count * 56 > len(d):
        raise BinParseError("track array runs past end of file (legacy .abin)")

    tracks = {}
    for i in range(count):
        o = arr + i * 56
        flags = _u32(d, o)
        name = d[o + 4:o + 0x24].split(b"\0")[0].decode("ascii", "replace")
        keys = _u16(d, o + 0x28)
        p_t = _u32(d, o + 0x2C)
        p_v = _u32(d, o + 0x30)

        is_trans = bool(flags & F_TRANS)
        is_rot = bool(flags & F_ROT)
        if not (is_trans or is_rot):
            continue

        quant = bool(flags & (F_TRANS_QUANT if is_trans else F_ROT_QUANT))
        if quant:
            # No sample of a quantised legacy track has turned up yet, so the
            # bits/scale field position for this branch is unknown. Refuse
            # rather than silently decode it wrong.
            raise BinParseError(
                "track '%s' is a quantised legacy track - format not yet "
                "confirmed, refusing to guess" % name)

        n_comp = 3 if is_trans else 4
        if p_t + keys * 4 > len(d) or p_v + keys * n_comp * 4 > len(d):
            raise BinParseError("track '%s' data out of range" % name)

        times = struct.unpack_from("<%df" % keys, d, p_t)
        out = []
        for k in range(keys):
            val = struct.unpack_from("<%df" % n_comp, d, p_v + k * n_comp * 4)
            out.append((times[k], val))

        slot = tracks.setdefault(name, {})
        slot["pos" if is_trans else "rot"] = out

    return duration, tracks


def parse_abin(path):
    with open(path, "rb") as f:
        d = f.read()

    if len(d) >= 0x30 and _u32(d, 0) == ABIN_MAGIC:
        return _parse_abin_modern(d)
    if d[:len(LEGACY_MAGIC)] == LEGACY_MAGIC:
        return _parse_abin_legacy(d)
    raise BinParseError("not an .abin")


def sample_pos(keys, t):
    if not keys:
        return Vector((0.0, 0.0, 0.0))
    if len(keys) == 1 or t <= keys[0][0]:
        return Vector(keys[0][1])
    if t >= keys[-1][0]:
        return Vector(keys[-1][1])
    for i in range(len(keys) - 1):
        t0, v0 = keys[i]
        t1, v1 = keys[i + 1]
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return Vector(v0).lerp(Vector(v1), f)
    return Vector(keys[-1][1])


def _quat(v):
    # file stores (x, y, z, w); Blender wants (w, x, y, z)
    return Quaternion((v[3], v[0], v[1], v[2])).normalized()


def sample_rot(keys, t):
    if not keys:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    if len(keys) == 1 or t <= keys[0][0]:
        return _quat(keys[0][1])
    if t >= keys[-1][0]:
        return _quat(keys[-1][1])
    for i in range(len(keys) - 1):
        t0, v0 = keys[i]
        t1, v1 = keys[i + 1]
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return _quat(v0).slerp(_quat(v1), f)
    return _quat(keys[-1][1])


# ============================================================================
#  Armature
# ============================================================================
#  Bones run along local -X in this rig (every child sits at a negative X
#  offset from its parent), while Blender bones run along +Y. AXIS_FIX bridges
#  the two. It is applied consistently to both the rest pose and every animated
#  pose, so the result is exact no matter which convention we picked.
# ============================================================================

AXIS_FIX = Matrix.Rotation(radians(90), 4, "Z")


def local_matrix(pose, name):
    p = pose.get(name, {})
    loc = Vector(p.get("pos", (0.0, 0.0, 0.0))[:3])
    rot = _quat(p.get("rot", (0.0, 0.0, 0.0, 1.0)))
    return Matrix.Translation(loc) @ rot.to_matrix().to_4x4()


def bind_pose_from_abin(path):
    """*_base.abin -> {bone: {'pos': vec3, 'rot': quat4}} using the single key."""
    _dur, tracks = parse_abin(path)
    return rest_pose_from_tracks(tracks)


def rest_pose_from_tracks(tracks):
    """Same shape as bind_pose_from_abin(), but from an already-parsed (and
    possibly resolve_mechanism_tracks()-remapped) tracks dict instead of
    re-reading the file. Needed because re-reading would give back the
    ORIGINAL, unresolved track names - measured on o_a_fan: the animation
    uses the resolved name "Spin", but bind_pose_from_abin(path) would only
    ever have "switch1", so the rest pose lookup for "Spin" would silently
    default to identity instead of the real first frame."""
    out = {}
    for name, t in tracks.items():
        e = {}
        if t.get("pos"):
            e["pos"] = t["pos"][0][1]
        if t.get("rot"):
            e["rot"] = t["rot"][0][1]
        out[name] = e
    return out



def _mat4_from_file(d, off, transpose=False):
    """Read one 4x4 float matrix from the BIN node-matrix table.

    The engine dumps matrices in its native row-vector layout on the samples
    measured so far (translation in the last ROW).  Blender uses column-vector
    transforms (translation in the last COLUMN), therefore the useful form is
    normally the transpose.  rigid_bin_node_matrices() still tests both forms
    instead of hard-coding that assumption so an odd exporter variant does not
    silently corrupt a mechanism.
    """
    vals = struct.unpack_from("<16f", d, off)
    m = Matrix((vals[0:4], vals[4:8], vals[8:12], vals[12:16]))
    return m.transposed() if transpose else m


def _matrix_max_error(a, b):
    return max(abs(float(a[r][c]) - float(b[r][c]))
               for r in range(4) for c in range(4))


def _matrix_affine_error(m):
    # In Blender/column-vector form an affine matrix's bottom row is 0,0,0,1.
    return (abs(float(m[3][0])) + abs(float(m[3][1]))
            + abs(float(m[3][2])) + abs(float(m[3][3]) - 1.0))


def _quat_angle_error(a, b):
    try:
        qa = a.normalized()
        qb = b.normalized()
        dot = min(1.0, max(-1.0, abs(float(qa.dot(qb)))))
        # 1-dot is enough for candidate scoring; no acos/radians needed.
        return 1.0 - dot
    except Exception:
        return 1.0


def rigid_bin_node_matrices(d, nodes, tracks=None):
    """Decode the per-node matrix table in a modern .bin.

    hdr[0x14] points at a node-matrix array and hdr[0x1C] at the name/parent
    array.  The horizontal-door files measured by the new door probe have
    128-byte node records: two 4x4 matrices per node.  Other actors commonly
    use 256-byte records; their first two matrices are still tested here, but
    ONLY rigid_node mechanisms consume this result.

    We deliberately do not guess which matrix is local/world, whether the dump
    needs transposing, or whether an exporter stored the inverse pair.  Eight
    interpretations are scored against BOTH the hierarchy and the ABIN frame-0
    transform (when a matching track exists).  The lowest-error interpretation
    wins.  This is the missing source of static node orientation that the old
    importer ignored entirely - exactly the kind of omission that can turn one
    door leaf by 90 degrees while the LVT placement itself is correct.

    Returns (local_by_name, world_by_index, metadata), or ({}, {}, metadata)
    when the table is unavailable/implausible.  Matrices are in GAME UNITS;
    scale is applied later when building Blender data.
    """
    meta = {"available": False}
    if not nodes or len(d) < 0x30:
        return {}, {}, meta

    try:
        mat_base_raw = _u32(d, 0x14)
        name_base_raw = _u32(d, 0x1C)
        if name_base_raw <= mat_base_raw:
            return {}, {}, meta
        span = name_base_raw - mat_base_raw
        if span % len(nodes):
            return {}, {}, meta
        stride = span // len(nodes)
        meta["stride"] = stride
        if stride < 128:
            return {}, {}, meta

        mat_base = mat_base_raw + BASE
        if mat_base + (len(nodes) - 1) * stride + 128 > len(d):
            return {}, {}, meta

        raw_pairs = []
        for i in range(len(nodes)):
            off = mat_base + i * stride
            raw_pairs.append((off, off + 64))

        candidates = []
        for transpose in (False, True):
            pair = []
            failed = False
            for a_off, b_off in raw_pairs:
                try:
                    pair.append((_mat4_from_file(d, a_off, transpose),
                                 _mat4_from_file(d, b_off, transpose)))
                except (struct.error, ValueError):
                    failed = True
                    break
            if failed:
                continue

            for swap in (False, True):
                for invert in (False, True):
                    local = []
                    world = []
                    ok = True
                    for a, b in pair:
                        l, w = ((b, a) if swap else (a, b))
                        if invert:
                            try:
                                l = l.inverted()
                                w = w.inverted()
                            except Exception:
                                ok = False
                                break
                        local.append(l.copy())
                        world.append(w.copy())
                    if not ok:
                        continue

                    affine = sum(_matrix_affine_error(m) for m in local + world)
                    hierarchy = 0.0
                    for i, (_bn, par) in enumerate(nodes):
                        expected = local[i] if par < 0 else (world[par] @ local[i])
                        hierarchy += _matrix_max_error(world[i], expected)

                    track_error = 0.0
                    track_hits = 0
                    if tracks:
                        for i, (bn, _par) in enumerate(nodes):
                            tr = tracks.get(bn)
                            if not tr:
                                continue
                            if tr.get("pos"):
                                p0 = Vector(tr["pos"][0][1][:3])
                                track_error += (local[i].to_translation() - p0).length
                                track_hits += 1
                            if tr.get("rot"):
                                q0 = sample_rot(tr["rot"], 0.0)
                                track_error += 4.0 * _quat_angle_error(
                                    local[i].to_quaternion(), q0)
                                track_hits += 1

                    # Affine/layout mistakes should dominate; hierarchy is the
                    # structural authority; ABIN frame 0 disambiguates direct
                    # vs inverse/local-vs-world in otherwise symmetric cases.
                    score = affine * 1000.0 + hierarchy * 100.0 + track_error
                    candidates.append((score, affine, hierarchy, track_error,
                                       track_hits, transpose, swap, invert,
                                       local, world))

        if not candidates:
            return {}, {}, meta
        candidates.sort(key=lambda x: x[0])
        (score, affine, hierarchy, track_error, track_hits,
         transpose, swap, invert, local, world) = candidates[0]

        meta.update({
            "available": True,
            "score": float(score),
            "affine_error": float(affine),
            "hierarchy_error": float(hierarchy),
            "track_error": float(track_error),
            "track_hits": int(track_hits),
            "transpose": bool(transpose),
            "pair_swapped": bool(swap),
            "inverted": bool(invert),
        })

        # A wildly non-affine result is worse than the old ABIN-only fallback.
        # Keep this threshold deliberately generous: normal float noise is many
        # orders of magnitude below it.
        if affine > 0.1 or hierarchy > max(0.1, len(nodes) * 0.05):
            meta["rejected"] = True
            return {}, {}, meta

        return ({nodes[i][0]: local[i] for i in range(len(nodes))},
                {i: world[i] for i in range(len(nodes))}, meta)
    except (struct.error, ValueError, ZeroDivisionError) as exc:
        meta["error"] = str(exc)
        return {}, {}, meta


def _matrix_with_channels(base, track):
    """Merge an animation track's frame-0 channels into a BIN local matrix.

    The BIN matrix supplies static channels that an animation does NOT record.
    Example: a sliding door can animate translation only while its node keeps a
    permanent 90-degree rest rotation.  The old importer replaced the entire
    local transform from the ABIN track and therefore silently lost that static
    rotation.
    """
    try:
        loc, rot, scl = base.decompose()
    except Exception:
        loc = Vector((0.0, 0.0, 0.0))
        rot = Quaternion((1.0, 0.0, 0.0, 0.0))
        scl = Vector((1.0, 1.0, 1.0))

    if track:
        if track.get("pos"):
            loc = sample_pos(track["pos"], 0.0)
        if track.get("rot"):
            rot = sample_rot(track["rot"], 0.0)

    sm = Matrix.Diagonal((float(scl[0]), float(scl[1]), float(scl[2]), 1.0))
    return Matrix.Translation(loc) @ rot.to_matrix().to_4x4() @ sm


def rigid_rest_state(d, nodes, tracks):
    """Build the complete rest state for a rigid_node mechanism.

    BIN local matrices are the static authority; ABIN frame 0 overrides only
    the channels it actually contains.  If the matrix table cannot be decoded,
    this cleanly falls back to identity + ABIN frame 0, i.e. the useful part of
    the old rigid_rest_offsets() behaviour without discarding rotations.

    Returns (local_by_name, world_by_index, metadata).
    """
    tracks = tracks or {}
    bin_local, _bin_world, meta = rigid_bin_node_matrices(d, nodes, tracks)

    local = {}
    for bn, _par in nodes:
        base = bin_local.get(bn, Matrix.Identity(4))
        local[bn] = _matrix_with_channels(base, tracks.get(bn))

    world = {}
    for i, (bn, par) in enumerate(nodes):
        L = local[bn]
        world[i] = (world[par] @ L) if par >= 0 and par in world else L.copy()

    meta = dict(meta)
    meta["source"] = "BIN+ABIN" if bin_local else "ABIN_fallback"
    return local, world, meta


def _scaled_rigid_matrix(m, scale):
    """Scale only translation; keep a clean orthonormal bone basis."""
    loc, rot, _scl = m.decompose()
    return Matrix.Translation(loc * scale) @ rot.to_matrix().to_4x4()

def find_base_abin(model_dir, obj_stem):
    folder = os.path.join(model_dir, "animations")
    if not os.path.isdir(folder):
        return None
    want = (obj_stem + "_base.abin").lower()
    exact, any_base = None, None
    for fn in os.listdir(folder):
        low = fn.lower()
        if low == want:
            exact = os.path.join(folder, fn)
        elif low.endswith("_base.abin") and any_base is None:
            any_base = os.path.join(folder, fn)
    return exact or any_base


def find_any_abin(model_dir, obj_stem):
    """Fallback for rigid_rest_offsets() when no *_base.abin exists at all -
    measured case: o_c_smalldoora only ships o_c_smalldoora_open.abin, no
    _base. Prefers a clip whose name starts with the object's own stem."""
    folder = os.path.join(model_dir, "animations")
    if not os.path.isdir(folder):
        return None
    prefixed, any_clip = None, None
    for fn in os.listdir(folder):
        low = fn.lower()
        if not low.endswith(".abin"):
            continue
        if prefixed is None and low.startswith(obj_stem.lower()):
            prefixed = os.path.join(folder, fn)
        elif any_clip is None:
            any_clip = os.path.join(folder, fn)
    return prefixed or any_clip


def rigid_rest_offsets(nodes, tracks):
    """For non-skinned multi-part rigid mechanisms (doors etc.) whose
    submeshes carry a "rigid_node" id (measured: matches a leaf node's index
    exactly, e.g. o_c_smalldoora's OuterSmallDoorLeft) instead of real skin
    weights - the raw vertex data alone is NOT the closed/rest pose. Measured
    on o_c_smalldoora.bin + o_c_smalldoora_open.abin: at t=0 (the animation's
    own first frame - this door has no *_base.abin at all, only the "_open"
    clip) leftdoor1/rightdoor1/outleftDoor/outrightDoor all carry a non-zero
    translation. Skipping it is exactly what left the two leaves crossing
    past the centre instead of meeting there.

    Returns {node_index: (dx, dy, dz)}, the accumulated t=0 translation of
    every ancestor (incl. the node itself) that has a position track.

    Rotation tracks are read but NOT composed here yet - no sample seen so
    far has one on an ancestor of a rigid_node leaf, so there is nothing to
    verify a rotation-composition order against. Applying an unverified
    rotation would risk turning a measured fix into a guess; recorded gap,
    not a silent guess.
    """
    by_idx = {i: (nm, par) for i, (nm, par) in enumerate(nodes)}
    offsets = {}
    for i in by_idx:
        dx = dy = dz = 0.0
        j, seen = i, set()
        while j is not None and j != -1 and j not in seen:
            seen.add(j)
            nm, par = by_idx.get(j, (None, -1))
            if nm is None:
                break
            tr = tracks.get(nm)
            if tr and tr.get("pos"):
                p0 = tr["pos"][0][1]
                dx += p0[0]; dy += p0[1]; dz += p0[2]
            j = par
        if dx or dy or dz:
            offsets[i] = (dx, dy, dz)
    return offsets


def rigid_driving_node(nodes, tracks, node_index):
    """Like rigid_rest_offsets, but returns the NAME of the single ancestor
    (including the node itself) whose track should drive this node's
    animation over time, or None if none has one. Used to keyframe a plain
    object directly from that track - no armature, no bone-axis convention
    needed at all, which sidesteps entirely the AXIS_FIX regression found
    when a synthetic armature was tried for these.

    Matches EITHER a position or a rotation track. Measured two distinct
    families: sliding mechanisms (o_c_smalldoora, o_b_horizontaldoorc/d) -
    pure translation, no rotation key at all; and hinge/swing mechanisms
    (o_f_doorbig, o_f_doorsmall, o_f_gatebig) - pure Z-axis rotation, NO
    position key at all (confirmed: parse_abin on all three returns only a
    "rot" entry per track, nothing else). animate_rigid_object() below
    keyframes whichever of the two the track actually provides.

    Only the FIRST tracked ancestor is used; a chain of more than one
    tracked ancestor hasn't come up in real data yet - same recorded
    limitation as rigid_rest_offsets."""
    by_idx = {i: (nm, par) for i, (nm, par) in enumerate(nodes)}
    j, seen = node_index, set()
    while j is not None and j != -1 and j not in seen:
        seen.add(j)
        nm, par = by_idx.get(j, (None, -1))
        if nm is None:
            break
        tr = tracks.get(nm)
        if tr and (tr.get("pos") or tr.get("rot")):
            return nm
        j = par
    return None


def animate_rigid_object(obj, track, duration, fps, scale, frame_offset=1):
    """Keyframes a plain object straight from a track - Location if it has
    a "pos" list, rotation_quaternion if it has a "rot" list, both if it
    somehow has both. No armature, no AXIS_FIX, no bone semantics at all.
    Resamples uniformly at `fps` using the same sample_pos()/sample_rot()
    interpolation the bone-animation path uses, for consistent playback
    speed/feel. Returns the number of frames baked (excluding the start)."""
    has_pos = bool(track.get("pos"))
    has_rot = bool(track.get("rot"))
    if not has_pos and not has_rot:
        return 0
    if obj.animation_data is None:
        obj.animation_data_create()
    if has_rot:
        obj.rotation_mode = "QUATERNION"
    n_frames = max(1, int(round(duration * fps)))
    for f in range(n_frames + 1):
        t = f * duration / n_frames if n_frames else 0.0
        frame = frame_offset + f
        if has_pos:
            x, y, z = sample_pos(track["pos"], t)
            obj.location = (x * scale, y * scale, z * scale)
            obj.keyframe_insert("location", frame=frame)
        if has_rot:
            obj.rotation_quaternion = sample_rot(track["rot"], t)
            obj.keyframe_insert("rotation_quaternion", frame=frame)
    return n_frames


def has_riggable_data(nodes, subs):
    """True if building a REAL armature is worth it: real skin weights only.

    rigid_node-only mechanisms (doors etc, see rigid_rest_offsets) do NOT
    go through here even though they're technically riggable in principle -
    confirmed regression: AXIS_FIX (see the Armature section below) assumes
    the character-bone convention "every child sits at a negative X offset
    from its parent", verified only on real character skeletons like Jango.
    Applying it to an arbitrary rigid-mechanism node hierarchy introduced a
    spurious 90-degree twist - reported on O_B_HorizontalDoorD after this
    function briefly also triggered on rigid_node, with a screenshot showing
    the exact warped silhouette that rotation would produce, on doors that
    stood correctly before that change. The static rigid_offsets correction
    doesn't touch bone axes at all and is unaffected - it stays the working,
    always-on fix for these mechanisms. A real animatable armature for rigid
    mechanisms is a real feature still worth having, but it needs an axis
    convention actually verified against a non-character rig first, not
    reusing AXIS_FIX on the assumption it generalises."""
    if len(nodes) <= 1:
        return False
    return any(s["skin"] for s in subs)


def build_armature(name, nodes, pose, scale):
    """nodes: [(bone_name, parent_index)]; pose: bind transforms from *_base.abin"""
    arm_data = bpy.data.armatures.new(name + "_arm")
    arm = bpy.data.objects.new(name + "_skeleton", arm_data)
    bpy.context.collection.objects.link(arm)

    prev = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")

    # world rest matrices, parents first (the file already guarantees that order)
    world = []
    for i, (bn, par) in enumerate(nodes):
        L = local_matrix(pose, bn)
        world.append((world[par] @ L) if par >= 0 else L)

    S = Matrix.Scale(scale, 4)
    ebones = []
    for i, (bn, par) in enumerate(nodes):
        eb = arm_data.edit_bones.new(bn)
        eb.head = (0.0, 0.0, 0.0)
        eb.tail = (0.0, 1.0, 0.0)              # placeholder; matrix sets the rest
        eb.matrix = S @ world[i] @ AXIS_FIX
        ebones.append(eb)

    # length: reach the first child, else inherit something sane
    kids = {}
    for i, (bn, par) in enumerate(nodes):
        if par >= 0:
            kids.setdefault(par, []).append(i)
    for i, eb in enumerate(ebones):
        ch = kids.get(i, [])
        if ch:
            d = (ebones[ch[0]].head - eb.head).length
            eb.length = max(d, 1e-4)
        else:
            eb.length = max(ebones[nodes[i][1]].length * 0.5, 1e-3) \
                if nodes[i][1] >= 0 else 0.1
    for i, (bn, par) in enumerate(nodes):
        if par >= 0:
            ebones[i].parent = ebones[par]

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = prev

    arm["swbh_rest"] = name
    return arm


def resolve_mechanism_tracks(nodes, tracks):
    """If no track name matches any node name at all, and there is exactly
    one track, remap it onto the root node's name. Measured on o_a_fan.bin
    + o_a_fan_spin.abin: the only track is called "switch1", matching
    neither "Spin" (root) nor "O_A_Fan" (the mesh-holding child) - yet this
    is unambiguously a single-part spinning mechanism (2 nodes, 1 track, a
    clean continuous rotation curve). Returns a (possibly identical) tracks
    dict; never mutates the input."""
    node_names = set(nm for nm, _ in nodes)
    if any(nm in tracks for nm in node_names):
        return tracks
    if len(tracks) == 1:
        only_name = next(iter(tracks))
        root_name = nodes[0][0]
        remapped = dict(tracks)
        remapped[root_name] = remapped.pop(only_name)
        return remapped
    return tracks


def build_rigid_armature(name, nodes, rest_local, scale):
    """Build an armature for a rigid_node mechanism from COMPLETE local rest
    matrices (BIN static transform + ABIN frame-0 channels).

    No character AXIS_FIX is used.  Unlike the old implementation, bone rest
    orientation is not reconstructed from animation tracks alone: unanimated
    channels come from the BIN node-matrix table, so a translation-only door
    keeps its authored static rotation.  Bone matrices are kept orthonormal;
    only their translations are scaled to Blender units.
    """
    arm_data = bpy.data.armatures.new(name + "_arm")
    arm = bpy.data.objects.new(name + "_skeleton", arm_data)
    bpy.context.collection.objects.link(arm)
    try:
        arm.show_in_front = True
    except Exception:
        pass

    prev = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")

    world = []
    for i, (bn, par) in enumerate(nodes):
        L = rest_local.get(bn, Matrix.Identity(4))
        world.append((world[par] @ L) if par >= 0 else L.copy())

    ebones = []
    for i, (bn, par) in enumerate(nodes):
        eb = arm_data.edit_bones.new(bn)
        eb.head = (0.0, 0.0, 0.0)
        eb.tail = (0.0, 1.0, 0.0)
        eb.matrix = _scaled_rigid_matrix(world[i], scale)
        ebones.append(eb)

    kids = {}
    for i, (_bn, par) in enumerate(nodes):
        if par >= 0:
            kids.setdefault(par, []).append(i)
    for i, eb in enumerate(ebones):
        ch = kids.get(i, [])
        if ch:
            distances = [(ebones[c].head - eb.head).length for c in ch]
            distances = [d for d in distances if d > 1e-6]
            eb.length = max(min(distances) if distances else 0.05 * scale, 1e-4)
        else:
            par = nodes[i][1]
            eb.length = max(ebones[par].length * 0.5, 1e-3) if par >= 0 else 0.1

    for i, (_bn, par) in enumerate(nodes):
        if par >= 0:
            ebones[i].parent = ebones[par]

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = prev

    arm["swbh_rest"] = name
    arm["swbh_rigid_armature"] = True
    return arm

def apply_rigid_animation_onto(arm, nodes, rest_local, tracks, fps, scale,
                               action, frame_offset=1):
    """Animate a rigid mechanism while preserving BIN-authored rest channels.

    ABIN clips are sparse by channel: a sliding door may contain translation
    only; a hinge may contain rotation only.  Missing channels MUST retain the
    BIN/rest value.  The previous rigid path substituted zero translation or
    identity rotation, which could erase a static 90-degree node orientation.

    At frame 0, the merged rest matrix and sampled animation matrix are
    identical, therefore every pose bone's matrix_basis is identity.  Later
    frames are exact deltas from that rest state.  Translation deltas are
    scaled to Blender units; rotations are untouched.
    """
    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = action

    rest_scaled = {bn: _scaled_rigid_matrix(
        rest_local.get(bn, Matrix.Identity(4)), scale) for bn, _ in nodes}
    rest_inv = {bn: rest_scaled[bn].inverted() for bn, _ in nodes}

    rest_components = {}
    for bn, _ in nodes:
        try:
            loc, rot, _scl = rest_local.get(bn, Matrix.Identity(4)).decompose()
        except Exception:
            loc = Vector((0.0, 0.0, 0.0))
            rot = Quaternion((1.0, 0.0, 0.0, 0.0))
        rest_components[bn] = (loc, rot)

    duration = 0.0
    for tr in tracks.values():
        if not isinstance(tr, dict):
            continue
        if tr.get("pos"):
            duration = max(duration, tr["pos"][-1][0])
        if tr.get("rot"):
            duration = max(duration, tr["rot"][-1][0])

    n_frames = max(1, int(round(duration * fps)))
    animated = [bn for bn, _ in nodes
                if bn in tracks and isinstance(tracks[bn], dict)
                and (tracks[bn].get("pos") or tracks[bn].get("rot"))]

    for f in range(n_frames + 1):
        t = f * duration / n_frames if n_frames else 0.0
        frame = frame_offset + f
        for bn in animated:
            pb = arm.pose.bones.get(bn)
            if pb is None:
                continue
            tr = tracks[bn]
            rest_loc, rest_rot = rest_components[bn]
            loc = sample_pos(tr["pos"], t) if tr.get("pos") else rest_loc
            rot = sample_rot(tr["rot"], t) if tr.get("rot") else rest_rot
            anim_local = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
            anim_scaled = _scaled_rigid_matrix(anim_local, scale)
            pb.matrix_basis = rest_inv[bn] @ anim_scaled
            pb.rotation_mode = "QUATERNION"
            if tr.get("pos"):
                pb.keyframe_insert("location", frame=frame, group=bn)
            if tr.get("rot"):
                pb.keyframe_insert("rotation_quaternion", frame=frame, group=bn)

    return n_frames

def apply_animation_onto(arm, nodes, rest_pose, path, fps, scale, action,
                         frame_offset, zero_root_translation=False,
                         root_bone_name=""):
    """Bakes one .abin onto `action`, starting at `frame_offset`.
    Returns (n_frames, n_bones_animated)."""
    duration, tracks = parse_abin(path)
    if not tracks:
        raise BinParseError("no tracks")

    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = action

    n_frames = max(1, int(round(duration * fps)))
    Fi = AXIS_FIX.inverted()

    rest_local = {bn: local_matrix(rest_pose, bn) for bn, _ in nodes}
    rest_inv = {bn: rest_local[bn].inverted() for bn, _ in nodes}

    animated = [bn for bn, _ in nodes if bn in tracks]
    # Which bone carries the whole-body movement is not something we've
    # confirmed yet - it might be the parentless bone, or a named bone one
    # level below it (e.g. 'godnode' wrapping 'root_joint'). Default to
    # "no parent"; let the caller override with an explicit name once that's
    # confirmed against the actual rig.
    root_bones = ({root_bone_name} if root_bone_name
                  else {bn for bn, parent in nodes if parent == -1})

    for f in range(n_frames + 1):
        t = f * duration / n_frames if n_frames else 0.0
        frame = frame_offset + f
        for bn in animated:
            pb = arm.pose.bones.get(bn)
            if pb is None:
                continue
            tr = tracks[bn]
            if zero_root_translation and bn in root_bones:
                loc = Vector(rest_pose.get(bn, {}).get("pos", (0.0, 0.0, 0.0))[:3])
            else:
                loc = (sample_pos(tr["pos"], t) if tr.get("pos")
                       else Vector(rest_pose.get(bn, {}).get("pos", (0.0, 0.0, 0.0))[:3]))
            rot = (sample_rot(tr["rot"], t) if tr.get("rot")
                   else _quat(rest_pose.get(bn, {}).get("rot", (0.0, 0.0, 0.0, 1.0))))
            L_anim = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()

            # basis = F^-1 . L_rest^-1 . L_anim . F   (derivation in the notes)
            pb.matrix_basis = Fi @ rest_inv[bn] @ L_anim @ AXIS_FIX
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert("location", frame=frame, group=bn)
            pb.keyframe_insert("rotation_quaternion", frame=frame, group=bn)

    return n_frames, len(animated)


def import_animation_clips(arm, nodes, rest_pose, paths, fps, scale, gap,
                           action_name, report, context=None,
                           zero_root_translation=False, root_bone_name=""):
    """Bakes each path in `paths` into its OWN Action, then lays them
    back-to-back as separate strips on one NLA track. Not muted - scrub the
    timeline and it plays straight through. Scene markers are added at each
    clip's start (visible in Timeline/Dope Sheet/NLA without any setup).

    Why not one shared Action (an earlier version of this did that): a bone's
    position/rotation in one clip's data has no relation to the same bone's
    data in the next clip - they're independent recordings, not a continuous
    path. Sharing one Action means Blender's F-curve interpolates smoothly
    between the last key of clip A and the first key of clip B, which for a
    moving root bone produces a huge, meaningless swoop across the scene at
    every clip boundary. Separate NLA strips don't have that problem - each
    is only evaluated within its own frame range, so playback cuts cleanly
    from one clip to the next instead of sweeping between them.

    Returns (nla_track, n_ok, n_frames_total)."""
    if arm.animation_data is None:
        arm.animation_data_create()

    track = arm.animation_data.nla_tracks.new()
    track.name = action_name
    track.mute = False

    cursor = 1
    ok = 0
    total = len(paths)
    wm = context.window_manager if context else None
    if wm:
        wm.progress_begin(0, total)
    for i, p in enumerate(paths, 1):
        print("[SWBH] (%d/%d) baking %s ..." % (i, total, os.path.basename(p)),
              flush=True)
        if wm:
            wm.progress_update(i - 1)
        clip_name = os.path.splitext(os.path.basename(p))[0]
        clip_action = bpy.data.actions.new(clip_name)
        clip_action.use_fake_user = True
        try:
            n_frames, n_bones = apply_animation_onto(
                arm, nodes, rest_pose, p, fps, scale, clip_action, 1,
                zero_root_translation=zero_root_translation,
                root_bone_name=root_bone_name)
        except (BinParseError, struct.error) as exc:
            report({"WARNING"}, "%s: %s" % (os.path.basename(p), exc))
            bpy.data.actions.remove(clip_action)
            continue
        except Exception as exc:                     # noqa: BLE001 - see below
            # Anything other than the two expected parse-error types means we
            # hit a genuinely new case. Don't let it silently swallow the rest
            # of the batch or vanish without a trace - print the full
            # traceback to the console so it can be diagnosed, same spirit as
            # the two errors above (name the file, keep going).
            import traceback
            traceback.print_exc()
            report({"WARNING"}, "%s: unexpected error (%s) - see console"
                   % (os.path.basename(p), exc))
            bpy.data.actions.remove(clip_action)
            continue

        for fc in clip_action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

        strip = track.strips.new(clip_name, cursor, clip_action)
        strip.name = clip_name
        if context is not None:
            context.scene.timeline_markers.new(clip_name, frame=cursor)
        print("[SWBH]   -> frames %d-%d, %d bones" % (cursor, cursor + n_frames, n_bones),
              flush=True)
        cursor += n_frames + gap
        ok += 1
    if wm:
        wm.progress_end()

    # No active action on top - let the NLA track alone drive the pose, so
    # what you see is exactly the strips laid out above, nothing blended in.
    arm.animation_data.action = None
    bpy.context.scene.frame_end = max(bpy.context.scene.frame_end, cursor)
    return track, ok, cursor


# ============================================================================
#  Preferences
# ============================================================================
#  The data root lives HERE, not in the import dialog: Blender refuses to open
#  a second file browser on top of the import one, so a DIR_PATH field there is
#  unusable.
# ============================================================================

class SWBHPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    data_root: StringProperty(
        name="Unpacked data root",
        subtype="DIR_PATH",
        description="The _unpacked folder holding data/ and data1/. "
                    "Leave empty to auto-detect from the imported file",
    )

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "data_root")
        col.label(text="Used to resolve the .mat and texture files a mesh refers to.",
                  icon="INFO")


def prefs_root():
    try:
        return bpy.context.preferences.addons[__name__].preferences.data_root.strip()
    except (KeyError, AttributeError):
        return ""


# ============================================================================
#  Import operator
# ============================================================================

class IMPORT_OT_swbh_bin(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.swbh_bin"
    bl_label = "Import SWBH .bin"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".bin"
    filter_glob: StringProperty(default="*.bin", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={"HIDDEN"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN"})

    scale: FloatProperty(
        name="Scale", default=1.0, min=0.001, max=1000.0,
        description="1.0 keeps game units (Jango stands 1.91 tall)")
    whole_area: BoolProperty(
        name="Import whole area (ignore file selection)", default=False,
        options={"HIDDEN"},
        description="Every world area ships as loose parts - <area>__pa_*.bin "
                    "(portal-culled rooms), <area>_skydome*.bin, and a few "
                    "loose <area>__visgeo*/__polysurface*/__lgcgeo*.bin - that "
                    "already sit in one shared world-space coordinate system "
                    "(checked: on a1, neighbouring rooms' bounding boxes tile "
                    "exactly, no gaps or overlaps - no external assembly file "
                    "exists or is needed). When this is on, the file(s) you "
                    "picked are only used to find the folder; every matching "
                    "part in it is imported and grouped into one collection. "
                    "Parts that are not the modern .bin format are skipped by "
                    "their header, not their name - measured on b1: 12 "
                    "pa_area_*.bin and 4 __visgeo*.bin there turned out to be "
                    "older/unidentified containers (same family as "
                    "gs_world.bin) and are correctly absent from b1.lvt's own "
                    "GEO manifest for that reason - the game does not load "
                    "them either")
    skydome: BoolProperty(
        name="Include skydome", default=True,
        description="Only used with 'Import whole area'. No separate "
                    "visibility switch was found for it (the WORLD block in "
                    "<area>.lvt has a SKY: reference, but that only names "
                    "which file(s) to use, not when to show them) - it is "
                    "just a big mesh that Blender, like the game, will only "
                    "show through wherever the room geometry has an opening")
    import_lights: BoolProperty(
        name="Import lights (from <area>.lvt)", default=True,
        description="Only used with 'Import whole area'. If <area>.lvt sits "
                    "next to the parts, its NUMLIGHTS section is read and "
                    "each entry becomes a real Blender point light (name, "
                    "position, rotation and colour - no radius/falloff field "
                    "was found in the file, so Blender's default stands in). "
                    "This is on top of, not instead of, the baked vertex-"
                    "colour lighting already on the geometry itself - "
                    "confirmed both exist at once on b1 (Coruscant)")
    import_actors: BoolProperty(
        name="Import actors/props (from <area>.lvt, experimental)", default=True,
        description="Only used with 'Import whole area'. Reads NUMACTORS from "
                    "<area>.lvt and, for every entry whose class name matches "
                    "a .bin found anywhere under the data root (by filename, "
                    "e.g. 'O_B_HorizontalDoorC' -> O_B_HorizontalDoorC.bin), "
                    "imports that mesh and places it at the actor's position/"
                    "rotation - doors, crates, switches, etc. Classes with no "
                    "matching file are skipped and listed in the console, not "
                    "silently dropped")
    actor_skeleton: BoolProperty(
        name="Actor skeleton (no animations)", default=True,
        description="Only used with 'Import actors/props'. If a placed "
                    "actor's mesh has more than one node and real skin data, "
                    "build its armature in bind pose too (needs a matching "
                    "*_base.abin next to that actor's .bin - skipped with a "
                    "console note if it's missing, same as the plain mesh "
                    "importer). Animation clips are never loaded here - open "
                    "the placed actor's .bin separately with the mesh "
                    "importer's own animation options for that.\n"
                    "Rigid multi-part mechanisms (doors, etc.) are a "
                    "different, separate case handled elsewhere, always on, "
                    "not by this checkbox: their submeshes aren't skinned at "
                    "all - each one is rigidly tied to one node instead "
                    "(measured on o_c_smalldoora.bin - a submesh 'id' field "
                    "no one had decoded turned out to match its node's index "
                    "exactly), and their rest-pose translation gets corrected "
                    "on its own. An actual armature was briefly built for "
                    "them too, but that reused AXIS_FIX (see the Armature "
                    "section), which assumes a character-bone convention "
                    "that doesn't hold for an arbitrary rigid hierarchy - "
                    "confirmed on O_B_HorizontalDoorD, which stood correctly "
                    "before that change and visibly twisted after. Reverted; "
                    "this checkbox only ever covers real smooth-skinned "
                    "skeletons again")
    animate_mechanisms: BoolProperty(
        name="Animate rigid mechanisms (doors etc.)", default=True,
        description="Rigid multi-part mechanisms (doors, gates, fans, switches, "
                    "etc.) are imported with a dedicated mechanism armature. "
                    "Static node transforms come from the BIN node-matrix "
                    "table, while ABIN frame 0 overrides only the channels it "
                    "actually records. This preserves authored rest rotations "
                    "and pivots and keeps the mechanism convenient to edit "
                    "and animate in Blender. Translation- and rotation-driven "
                    "mechanisms are both supported.")
    flip_v: BoolProperty(
        name="Flip V", default=True,
        description="The game puts the UV origin at the top, Blender at the bottom")
    normals: BoolProperty(
        name="Import normals", default=True,
        description="Use the file's normals as custom split normals")
    split_parts: BoolProperty(
        name="Split into connected parts", default=False,
        description="A .bin can pack many unrelated pieces into one shared "
                    "vertex buffer purely for storage (measured on "
                    "b1__pa_area_b: all 91 visible submeshes - walls, floor, "
                    "every prop including a light fixture - share one "
                    "buffer, but literally NONE of them share an actual "
                    "vertex index with any other; every one is already its "
                    "own disconnected island). Splitting on real shared-"
                    "vertex topology (not just material) turns each island "
                    "into its own object with its own normals, which is why "
                    "the same light fixture prop could come out smoothed in "
                    "one room's import and faceted in an adjacent one - each "
                    "room's shared buffer was previously treated as a single "
                    "blob for every purpose, seam-smoothing included. Off "
                    "here by default (opening one .bin usually means "
                    "wanting one object); on by default when importing a "
                    "whole area, where this problem actually shows up")
    vcol: BoolProperty(
        name="Import vertex colours", default=True,
        description="Level geometry stores its baked lighting per vertex. "
                    "Without this a map renders flat")
    textures: BoolProperty(
        name="Find textures", default=True,
        description="Resolve .mat and texture files from the unpacked tree")
    skeleton: BoolProperty(
        name="Import skeleton", default=True,
        description="Build an armature from the node array. The bind pose comes "
                    "from the matching *_base.abin next to the model")
    import_animations: BoolProperty(
        name="Import animations", default=False,
        description="Also bake every *.abin found in the model's adjacent "
                    "'animations' folder onto the new armature, back-to-back "
                    "on one Action with a marker per clip")
    anim_fps: FloatProperty(
        name="Sample rate", default=30.0, min=1.0, max=120.0,
        description="Only used when 'Import animations' is checked")
    anim_gap: IntProperty(
        name="Gap between clips (frames)", default=10, min=0,
        description="Only used when 'Import animations' is checked")
    anim_start: IntProperty(
        name="Start at #", default=0, min=0,
        description="Skip this many clips before importing. Clips are sorted "
                    "alphabetically by filename, so this is a stable way to "
                    "import in batches")
    anim_count: IntProperty(
        name="How many (0 = all)", default=0, min=0,
        description="Import at most this many clips, starting from 'Start at #'. "
                    "0 imports every remaining clip")
    zero_root_translation: BoolProperty(
        name="Freeze root in place", default=False,
        description="Keep the root bone's position at the bind pose and only "
                    "bake its rotation. Use this if a clip's root motion (the "
                    "whole character sliding/rotating through the world, as "
                    "recorded for locomotion) is not what you want for this "
                    "particular preview/edit")
    root_bone_name: StringProperty(
        name="Root bone (blank = auto)", default="",
        description="Which bone 'Freeze root in place' targets. Leave blank to "
                    "use the parentless bone automatically - only set this if "
                    "that guess is wrong (e.g. the mover turns out to be a "
                    "named bone like 'root_joint' one level under a wrapper "
                    "node such as 'godnode')")
    lod: EnumProperty(
        name="LOD",
        items=[
            ("SEPARATE", "All, as separate objects (recommended)",
             "Import every LOD level found, each as its own object named "
             "'<name>_LOD0', '<name>_LOD1', ... (also tagged with a custom "
             "property 'swbh_lod'). Nothing is discarded"),
            ("AUTO", "Highest detail only",
             "Import only the most detailed LOD level found - what the game "
             "shows at close range. Discards the other levels"),
            ("MERGE", "All, merged into one object (old behaviour)",
             "Combine every LOD level into a single mesh - this is what "
             "earlier versions did before LOD was discovered. Keeps every "
             "triangle but loses the separation between levels; not "
             "recommended, kept for comparison/debugging"),
        ],
        default="SEPARATE",
        description="Some models carry multiple levels of detail as extra "
                    "submeshes tagged in the vertex-format field")
    collision: EnumProperty(
        name="Collision hulls",
        items=[
            ("SKIP", "Skip", "Do not import them"),
            ("SEPARATE", "Separate collection",
             "Import into a hidden 'SWBH Collision' collection"),
            ("MERGE", "Merge in", "Treat them as normal geometry (not recommended)"),
        ],
        default="SEPARATE",
        description="Submeshes with no material, no UVs and no normals are "
                    "never drawn by the game")

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "scale")
        col.prop(self, "flip_v")
        col.prop(self, "normals")
        col.prop(self, "vcol")
        col.prop(self, "split_parts")
        col.separator()
        col.prop(self, "textures")
        col.prop(self, "skeleton")
        if self.skeleton:
            col.prop(self, "import_animations")
            if self.import_animations:
                sub = col.column()
                sub.prop(self, "anim_fps")
                sub.prop(self, "anim_gap")
                sub.prop(self, "anim_start")
                sub.prop(self, "anim_count")
                sub.prop(self, "zero_root_translation")
                if self.zero_root_translation:
                    sub.prop(self, "root_bone_name")
        col.prop(self, "animate_mechanisms")
        col.separator()
        col.prop(self, "lod")
        col.prop(self, "collision")

        box = self.layout.box()
        root = prefs_root()
        if root:
            box.label(text="Data root: " + os.path.basename(root.rstrip("\\/")),
                      icon="CHECKMARK")
        else:
            box.label(text="Data root: auto-detect", icon="INFO")
            box.label(text="Set it in Preferences > Add-ons if lookup fails")

    def execute(self, context):
        paths = ([os.path.join(self.directory, f.name) for f in self.files]
                 if self.files else ([self.filepath] if self.filepath else []))
        paths = [p for p in paths if p.lower().endswith(".bin")]
        if not paths and not (self.whole_area and self.directory):
            self.report({"ERROR"}, "no .bin selected")
            return {"CANCELLED"}

        missing = set()
        imported = []
        failed = 0

        area_collection = None
        lvt = {"sky": None, "geom": {}, "lights": [], "actors": [], "world": {}}
        if self.whole_area:
            area_dir = self.directory or os.path.dirname(paths[0])
            area_name = os.path.basename(area_dir.rstrip("\\/")) or "SWBH Area"

            lvt_path = None
            for fn in os.listdir(area_dir):
                if fn.lower() == area_name.lower() + ".lvt":
                    lvt_path = os.path.join(area_dir, fn)
                    break
            if lvt_path:
                lvt = parse_lvt(lvt_path)
                apply_lvt_world_settings(lvt_path, lvt.get("world", {}), self.scale)
            sky_prefix = (lvt["sky"] or (area_name + "_skydome")).lower()

            found = []
            skipped_other_format = []
            for fn in sorted(os.listdir(area_dir)):
                low = fn.lower()
                if not low.endswith(".bin"):
                    continue
                is_part = ("__pa_" in low or low.startswith(sky_prefix)
                          or "__visgeo" in low or "__polysurface" in low
                          or "__lgcgeo" in low)
                if not is_part:
                    continue
                if low.startswith(sky_prefix) and not self.skydome:
                    continue
                full = os.path.join(area_dir, fn)
                if not is_modern_bin(full):
                    skipped_other_format.append(fn)  # legacy/unknown container
                    continue
                found.append(fn)
            if skipped_other_format:
                print("[SWBH]   skipped %d file(s) not in the modern .bin "
                      "format (legacy exporter or unidentified header): %s"
                      % (len(skipped_other_format), ", ".join(skipped_other_format)))
            if not found:
                self.report({"ERROR"},
                            "'%s' has no __pa_*.bin parts - not a level area folder?"
                            % area_dir)
                return {"CANCELLED"}
            paths = [os.path.join(area_dir, fn) for fn in found]
            area_collection = get_collection(area_name)
            if lvt_path:
                area_collection["swbh_source_lvt"] = lvt_path
                for _k, _v in lvt.get("world", {}).items():
                    try:
                        area_collection["swbh_world_" + _k] = _v
                    except Exception:
                        pass
            print("[SWBH] whole area '%s': %d part files found%s"
                  % (area_name, len(paths),
                     " (%s.lvt found, sky='%s', %d lights, %d actors, %d/%d "
                     "GEO entries on disk - manifest is informational only, "
                     "every part actually on disk is still imported)"
                     % (area_name, lvt["sky"], len(lvt["lights"]), len(lvt["actors"]),
                        sum(1 for f in lvt["geom"] if os.path.isfile(os.path.join(area_dir, f))),
                        len(lvt["geom"]))
                     if lvt_path else " (no .lvt found beside it)"))

        if not paths:
            self.report({"ERROR"}, "no .bin selected")
            return {"CANCELLED"}

        model_dir = os.path.dirname(paths[0])
        root = prefs_root() or find_data_root(model_dir) or ""
        index = (build_index(root)
                 if self.textures and root and os.path.isdir(root) else ({}, {}))
        if self.textures and not root:
            self.report({"WARNING"},
                        "data root not found - looking only beside the model")

        opt = {"scale": self.scale, "flip_v": self.flip_v, "normals": self.normals,
               "vcol": self.vcol, "textures": self.textures}

        if self.whole_area and self.import_lights and lvt["lights"]:
            sc = self.scale
            for lt in lvt["lights"]:
                light = bpy.data.lights.new(lt["name"], type="POINT")
                r, g, b, _a = lt["lightcolor"]
                light.color = (r, g, b)
                obj = bpy.data.objects.new(lt["name"], light)
                x, y, z = lt["pos"]
                obj.location = (x * sc, y * sc, z * sc)
                w, qx, qy, qz = lt["rot"]
                obj.rotation_mode = "QUATERNION"
                obj.rotation_quaternion = Quaternion((w, qx, qy, qz))
                obj["swbh_ambient"] = lt["ambient"]
                area_collection.objects.link(obj)
                imported.append(obj)
            print("[SWBH]   lights: %d imported from %s.lvt"
                  % (len(lvt["lights"]), area_name))

        if self.whole_area and self.import_actors and lvt["actors"]:
            bin_index = build_bin_index(root) if root else {}
            sc = self.scale
            n_placed = 0
            n_armatures = 0
            missing_classes = set()
            missing_base = set()
            for a in lvt["actors"]:
                bp = bin_index.get(a["cls"].lower())
                if not bp:
                    missing_classes.add(a["cls"])
                    continue
                try:
                    ad, a_internal, a_subs, a_nodes = parse_bin(bp)
                except (BinParseError, struct.error):
                    missing_classes.add(a["cls"] + "  (found but failed to parse)")
                    continue
                a_visible = [s for s in a_subs if not s["collision"]
                            and s["lod"] == min(s2["lod"] for s2 in a_subs)]
                if not a_visible:
                    continue

                a_arm = None
                a_tracks = {}
                a_dur = 0.0
                a_rp = None
                a_rest_local = {}
                a_rigid_matrices = {}
                a_rigid_meta = {}
                a_is_rigid = False
                a_has_rigid_nodes = (len(a_nodes) > 1 and any(
                    s["rigid_node"] > 0 and not s.get("skin") for s in a_visible))
                if a_has_rigid_nodes:
                    a_stem = os.path.splitext(os.path.basename(bp))[0]
                    a_anim_dir = os.path.join(os.path.dirname(bp), "animations")
                    a_rp = (find_base_abin(os.path.dirname(bp), a_stem)
                           or find_any_abin(os.path.dirname(bp), a_stem))
                    # A non-zero rigid_node is also used by some exported scene
                    # graphs.  Only opt into mechanism rest matrices when this
                    # asset actually has an animation context; otherwise leave
                    # ordinary static props exactly as previous versions did.
                    a_is_rigid = bool(a_rp or os.path.isdir(a_anim_dir))
                    if a_is_rigid and a_rp:
                        try:
                            a_dur, a_tracks = parse_abin(a_rp)
                            a_tracks = resolve_mechanism_tracks(a_nodes, a_tracks)
                        except (BinParseError, struct.error) as exc:
                            print("[SWBH]   %s: mechanism ABIN parse failed %s: %s"
                                  % (a["cls"], os.path.basename(a_rp), exc))
                            a_tracks = {}
                    if a_is_rigid:
                        a_rest_local, a_rigid_matrices, a_rigid_meta = rigid_rest_state(
                            ad, a_nodes, a_tracks)
                    if a_is_rigid and a_rigid_meta.get("available"):
                        print("[SWBH]   %s: rigid rest from BIN matrices "
                              "(stride=%s, mode=%s%s%s, hierarchy_err=%.6g)"
                              % (a["cls"], a_rigid_meta.get("stride"),
                                 "transpose" if a_rigid_meta.get("transpose") else "raw",
                                 "+swap" if a_rigid_meta.get("pair_swapped") else "",
                                 "+inverse" if a_rigid_meta.get("inverted") else "",
                                 a_rigid_meta.get("hierarchy_error", 0.0)))
                    elif a_is_rigid and a_rp:
                        print("[SWBH]   %s: BIN rigid matrix table unavailable; "
                              "using ABIN frame-0 fallback" % a["cls"])

                if self.actor_skeleton and has_riggable_data(a_nodes, a_visible):
                    a_stem = os.path.splitext(os.path.basename(bp))[0]
                    a_base = (find_base_abin(os.path.dirname(bp), a_stem)
                             or find_any_abin(os.path.dirname(bp), a_stem))
                    if a_base:
                        try:
                            a_pose = bind_pose_from_abin(a_base)
                            a_arm = build_armature(a["name"] + "_armature",
                                                   a_nodes, a_pose, sc)
                            a_arm["swbh_base_abin"] = a_base
                            x, y, z = a["pos"]
                            a_arm.location = (x * sc, y * sc, z * sc)
                            w, qx, qy, qz = a["rot"]
                            a_arm.rotation_mode = "QUATERNION"
                            a_arm.rotation_quaternion = Quaternion((w, qx, qy, qz))
                            area_collection.objects.link(a_arm)
                            imported.append(a_arm)
                            n_armatures += 1
                        except (BinParseError, struct.error):
                            a_arm = None
                    else:
                        missing_base.add(a["cls"])

                # Rigid mechanisms keep a real armature for convenient editing,
                # but their rest state comes from BIN node matrices merged with
                # ABIN frame 0.  This preserves static leaf orientation/pivots.
                animated = False
                if self.animate_mechanisms and a_tracks and a_is_rigid:
                    any_animated_node = any(
                        nm in a_tracks and (a_tracks[nm].get("pos") or a_tracks[nm].get("rot"))
                        for nm, _ in a_nodes)
                    if any_animated_node:
                        try:
                            mech_arm = build_rigid_armature(a["name"], a_nodes,
                                                            a_rest_local, sc)
                            if a_rp:
                                mech_arm["swbh_base_abin"] = a_rp
                            mech_arm["swbh_rigid_rest_source"] = a_rigid_meta.get(
                                "source", "unknown")
                            x, y, z = a["pos"]
                            mech_arm.location = (x * sc, y * sc, z * sc)
                            w, qx, qy, qz = a["rot"]
                            mech_arm.rotation_mode = "QUATERNION"
                            mech_arm.rotation_quaternion = Quaternion((w, qx, qy, qz))
                            mech_arm["swbh_actor_id"] = a["id"]
                            mech_arm["swbh_actor_class"] = a["cls"]
                            area_collection.objects.link(mech_arm)
                            imported.append(mech_arm)

                            act = bpy.data.actions.new(a["name"] + "_anim")
                            act.use_fake_user = True
                            apply_rigid_animation_onto(
                                mech_arm, a_nodes, a_rest_local, a_tracks,
                                30.0, sc, act)

                            mech_obj = build_mesh(
                                ad, a["name"], a_visible, os.path.dirname(bp),
                                index, opt, missing, a_nodes, None,
                                a_rigid_matrices)
                            if mech_obj and mech_obj.vertex_groups:
                                mech_obj.parent = mech_arm
                                mod = mech_obj.modifiers.new("Armature", "ARMATURE")
                                mod.object = mech_arm
                                mech_obj["swbh_actor_id"] = a["id"]
                                mech_obj["swbh_actor_class"] = a["cls"]
                                area_collection.objects.link(mech_obj)
                                imported.append(mech_obj)
                                animated = True
                                n_placed += 1
                        except (BinParseError, struct.error, ValueError) as exc:
                            print("[SWBH]   %s: mechanism armature animation failed "
                                  "(%s) - falling back to static corrected mesh"
                                  % (a["cls"], exc))

                if animated:
                    continue

                aobj = build_mesh(ad, a["name"], a_visible, os.path.dirname(bp),
                                  index, opt, missing, a_nodes, None,
                                  a_rigid_matrices)
                if aobj:
                    if a_arm is not None and aobj.vertex_groups:
                        # The armature already carries the actor's placement,
                        # so the mesh only needs to follow it, at local origin.
                        aobj.parent = a_arm
                        mod = aobj.modifiers.new("Armature", "ARMATURE")
                        mod.object = a_arm
                    else:
                        x, y, z = a["pos"]
                        aobj.location = (x * sc, y * sc, z * sc)
                        w, qx, qy, qz = a["rot"]
                        aobj.rotation_mode = "QUATERNION"
                        aobj.rotation_quaternion = Quaternion((w, qx, qy, qz))
                    aobj["swbh_actor_id"] = a["id"]
                    aobj["swbh_actor_class"] = a["cls"]
                    actor_ats = find_related(root, a["cls"], ".ats", preferred_dir=os.path.dirname(bp)) if root else None
                    if actor_ats:
                        aobj["swbh_source_ats"] = actor_ats
                        ainfo = parse_ats(actor_ats)
                        a_related = resolve_ats_asset(actor_ats, ainfo, root)
                        apply_asset_metadata(aobj, actor_ats, ainfo, a_related)
                    area_collection.objects.link(aobj)
                    imported.append(aobj)
                    n_placed += 1
            print("[SWBH]   actors: %d/%d placed, %d with a bind-pose armature"
                  % (n_placed, len(lvt["actors"]), n_armatures))
            if missing_classes:
                print("[SWBH]   actor classes with no .bin found (%d): %s"
                      % (len(missing_classes), ", ".join(sorted(missing_classes))))
            if missing_base:
                print("[SWBH]   actor classes with skin data but no *_base.abin "
                      "found (%d, imported as static mesh instead): %s"
                      % (len(missing_base), ", ".join(sorted(missing_base))))

        for p in paths:
            try:
                d, internal, subs, nodes = parse_bin(p)
            except (BinParseError, struct.error) as exc:
                self.report({"WARNING"}, "%s: %s" % (os.path.basename(p), exc))
                failed += 1
                continue

            stem = os.path.splitext(os.path.basename(p))[0]
            name = stem if (not internal or GENERIC_NAMES.match(internal)) else internal
            mdir = os.path.dirname(p)

            lods_present = sorted(set(s["lod"] for s in subs))
            if self.lod == "AUTO":
                keep = lods_present[0]
                lod_groups = {keep: [s for s in subs if s["lod"] == keep]}
            elif self.lod == "SEPARATE":
                lod_groups = {lv: [s for s in subs if s["lod"] == lv]
                              for lv in lods_present}
            else:                                    # MERGE - old behaviour
                lod_groups = {lods_present[0]: subs}
            multi_lod = len(lods_present) > 1

            # Decode the complete rigid mechanism rest state once.  The BIN
            # matrix table carries static local orientation/pivots; ABIN frame 0
            # overrides only animated channels.  This is independent of the
            # character-skeleton checkbox.
            rigid_tracks = {}
            rigid_rp = None
            rigid_rest_local = {}
            rigid_matrices = {}
            rigid_meta = {}
            has_rigid_nodes = (len(nodes) > 1 and any(
                s["rigid_node"] > 0 and not s.get("skin") for s in subs))
            is_rigid_mechanism = False
            if has_rigid_nodes:
                rigid_rp = find_base_abin(mdir, stem) or find_any_abin(mdir, stem)
                anim_dir_for_mech = os.path.join(mdir, "animations")
                # Whole-area room BINs can carry node ids too but have no
                # mechanism animation folder.  Do not reinterpret static level
                # scene graphs as articulated mechanisms.
                is_rigid_mechanism = bool(rigid_rp or os.path.isdir(anim_dir_for_mech))
                if is_rigid_mechanism and rigid_rp:
                    try:
                        _dur, rigid_tracks = parse_abin(rigid_rp)
                        rigid_tracks = resolve_mechanism_tracks(nodes, rigid_tracks)
                    except (BinParseError, struct.error) as exc:
                        print("[SWBH]   mechanism ABIN parse failed %s: %s"
                              % (os.path.basename(rigid_rp), exc))
                        rigid_tracks = {}
                if is_rigid_mechanism:
                    rigid_rest_local, rigid_matrices, rigid_meta = rigid_rest_state(
                        d, nodes, rigid_tracks)
                if is_rigid_mechanism and rigid_meta.get("available"):
                    print("[SWBH]   rigid rest: BIN node matrices stride=%s, "
                          "mode=%s%s%s, hierarchy_err=%.6g"
                          % (rigid_meta.get("stride"),
                             "transpose" if rigid_meta.get("transpose") else "raw",
                             "+swap" if rigid_meta.get("pair_swapped") else "",
                             "+inverse" if rigid_meta.get("inverted") else "",
                             rigid_meta.get("hierarchy_error", 0.0)))
                elif is_rigid_mechanism and rigid_rp:
                    print("[SWBH]   rigid rest: BIN matrix table unavailable; "
                          "using ABIN frame-0 fallback")

            # Build the armature first: the mesh gets parented to it.
            arm = None
            if self.skeleton and has_riggable_data(nodes, subs):
                base = find_base_abin(mdir, stem) or find_any_abin(mdir, stem)
                if base:
                    try:
                        pose = bind_pose_from_abin(base)
                        arm = build_armature(name, nodes, pose, self.scale)
                        arm["swbh_base_abin"] = base
                        imported.append(arm)
                        print("[SWBH]   armature: %d bones, bind pose from %s"
                              % (len(nodes), os.path.basename(base)))

                        if self.import_animations:
                            anim_dir = os.path.join(mdir, "animations")
                            clips = []
                            if os.path.isdir(anim_dir):
                                clips = sorted(
                                    os.path.join(anim_dir, fn)
                                    for fn in os.listdir(anim_dir)
                                    if fn.lower().endswith(".abin")
                                    and not fn.lower().endswith("_base.abin"))
                            n_found = len(clips)
                            if self.anim_start or self.anim_count:
                                end = (self.anim_start + self.anim_count
                                       if self.anim_count else None)
                                clips = clips[self.anim_start:end]
                            if clips:
                                _track, n_ok, total = import_animation_clips(
                                    arm, nodes, pose, clips, self.anim_fps,
                                    self.scale, self.anim_gap, name + "_anim",
                                    self.report, context=context,
                                    zero_root_translation=self.zero_root_translation,
                                    root_bone_name=self.root_bone_name)
                                if n_ok:
                                    context.scene.frame_start = 1
                                    context.scene.frame_end = max(
                                        context.scene.frame_end, total)
                                    print("[SWBH]   animations: %d/%d selected "
                                          "(%d found in folder), %d frames total"
                                          % (n_ok, len(clips), n_found, total))
                            else:
                                self.report({"WARNING"},
                                            "no animation .abin found beside %s "
                                            "(or start/count skipped them all)" % stem)
                    except (BinParseError, struct.error) as exc:
                        self.report({"WARNING"}, "skeleton: %s" % exc)
                        arm = None
                else:
                    self.report({"WARNING"},
                                "no *_base.abin or any other .abin beside %s "
                                "- skeleton skipped" % stem)

            # Rigid mechanism armature.  Unlike the old 1.0/1.1.1 path,
            # this does not reconstruct rest transforms from sparse ABIN tracks
            # alone; it uses BIN static matrices too, so missing rotation/position
            # channels cannot zero out an authored rest transform.
            mech_animated = False
            if (arm is None and self.animate_mechanisms and is_rigid_mechanism
                    and rigid_tracks):
                any_animated_node = any(
                    nm in rigid_tracks and (rigid_tracks[nm].get("pos")
                                            or rigid_tracks[nm].get("rot"))
                    for nm, _ in nodes)
                if any_animated_node:
                    try:
                        arm = build_rigid_armature(
                            name, nodes, rigid_rest_local, self.scale)
                        if rigid_rp:
                            arm["swbh_base_abin"] = rigid_rp
                        arm["swbh_rigid_rest_source"] = rigid_meta.get(
                            "source", "unknown")
                        imported.append(arm)
                        act = bpy.data.actions.new(name + "_anim")
                        act.use_fake_user = True
                        n_frames = apply_rigid_animation_onto(
                            arm, nodes, rigid_rest_local, rigid_tracks,
                            self.anim_fps, self.scale, act)

                        mech_visible = [s for s in subs if not s["collision"]]
                        mech_obj = build_mesh(
                            d, name, mech_visible, mdir, index, opt, missing,
                            nodes, None, rigid_matrices)
                        if mech_obj and mech_obj.vertex_groups:
                            mech_obj.parent = arm
                            mod = mech_obj.modifiers.new("Armature", "ARMATURE")
                            mod.object = arm
                            (area_collection or context.collection).objects.link(mech_obj)
                            imported.append(mech_obj)
                            context.scene.frame_start = 1
                            context.scene.frame_end = max(
                                context.scene.frame_end, n_frames + 1)
                            mech_animated = True
                    except (BinParseError, struct.error, ValueError) as exc:
                        self.report({"WARNING"}, "mechanism animation: %s" % exc)

            for lod_level, lod_subs in sorted(lod_groups.items()):
                if mech_animated:
                    continue
                suffix = "_LOD%d" % lod_level if (self.lod == "SEPARATE" and multi_lod) else ""
                lod_name = name + suffix

                if self.collision == "MERGE":
                    visible, hulls = lod_subs, []
                else:
                    visible = [s for s in lod_subs if not s["collision"]]
                    hulls = [s for s in lod_subs if s["collision"]]

                if visible:
                    groups = split_connected(visible) if self.split_parts else [visible]
                    multi_part = len(groups) > 1
                    for gi, part in enumerate(groups):
                        if multi_part:
                            tex_hint = os.path.splitext(part[0]["texture"])[0] or "part"
                            part_name = "%s_%02d_%s" % (lod_name, gi, tex_hint)
                        else:
                            part_name = lod_name
                        obj = build_mesh(d, part_name, part, mdir, index, opt,
                                         missing, nodes, None, rigid_matrices)
                        if obj:
                            obj["swbh_source"] = internal
                            obj["swbh_source_bin"] = p
                            obj["swbh_lod"] = lod_level
                            obj["swbh_is_skydome"] = bool(
                                self.whole_area and os.path.basename(p).lower().startswith(sky_prefix))
                            (area_collection or context.collection).objects.link(obj)
                            imported.append(obj)
                            if arm is not None and obj.vertex_groups:
                                obj.parent = arm
                                mod = obj.modifiers.new("Armature", "ARMATURE")
                                mod.object = arm

                if hulls and self.collision == "SEPARATE":
                    try:
                        hull_obj = build_mesh(d, lod_name + "_collision", hulls, mdir,
                                              index, opt, missing, nodes, None, rigid_matrices)
                    except (BinParseError, struct.error, ValueError, UnboundLocalError) as exc:
                        print("[SWBH]   collision hull skipped for %s: %s"
                              % (lod_name, exc))
                        hull_obj = None
                    if hull_obj:
                        hull_obj.display_type = "WIRE"
                        hull_obj.hide_render = True
                        get_collection("SWBH Collision").objects.link(hull_obj)
                        hull_obj.hide_set(True)

                if multi_lod:
                    if self.lod == "MERGE":
                        lod_note = " (levels found: %s, ALL merged)" % lods_present
                    elif self.lod == "SEPARATE":
                        lod_note = " (levels found: %s, importing all separately)" % lods_present
                    else:
                        lod_note = " (levels found: %s, keeping %d)" % (lods_present, lod_level)
                else:
                    lod_note = ""

                print("[SWBH] %s%s: '%s', %d nodes, %d visible + %d collision "
                      "submeshes, %d tris%s%s"
                      % (os.path.basename(p),
                         " [LOD%d]" % lod_level if multi_lod else "",
                         internal, len(nodes), len(visible), len(hulls),
                         sum(len(s["indices"]) // 3 for s in visible),
                         ", skinned" if any(s["skinned"] for s in visible) else "",
                         lod_note))

        if missing:
            print("[SWBH] TEXTURES NOT FOUND: %d" % len(missing))
            for m in sorted(missing):
                print("[SWBH]   %s" % m)
            self.report({"WARNING"},
                        "%d textures not found (see System Console)" % len(missing))

        if not imported:
            self.report({"ERROR"}, "nothing imported")
            return {"CANCELLED"}

        for o in context.selected_objects:
            o.select_set(False)
        for o in imported:
            o.select_set(True)
        context.view_layer.objects.active = imported[0]

        self.report({"INFO"}, "imported %d mesh(es)%s"
                    % (len(imported), ", %d failed" % failed if failed else ""))
        return {"FINISHED"}


class IMPORT_OT_swbh_lvt(bpy.types.Operator, ImportHelper):
    """Import an entire level area in one step, starting from its .lvt"""
    bl_idname = "import_scene.swbh_lvt"
    bl_label = "Import SWBH Level Area (.lvt)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".lvt"
    filter_glob: StringProperty(default="*.lvt", options={"HIDDEN"})

    scale: FloatProperty(name="Scale", default=1.0, min=0.001, max=1000.0)
    flip_v: BoolProperty(name="Flip V", default=True)
    textures: BoolProperty(
        name="Find textures", default=True,
        description="Resolve .mat and texture files from the unpacked tree")
    skydome: BoolProperty(name="Include skydome", default=True)
    import_lights: BoolProperty(name="Import lights", default=True)
    import_actors: BoolProperty(
        name="Import actors/props", default=True,
        description="Places every actor whose class matches a .bin found "
                    "anywhere under the data root")
    actor_skeleton: BoolProperty(
        name="Actor skeleton (no animations)", default=True,
        description="Build the armature (bind pose only, no clips) for "
                    "placed actors that have real skin data and a matching "
                    "*_base.abin next to their .bin")
    animate_mechanisms: BoolProperty(
        name="Animate rigid mechanisms (doors etc.)", default=True,
        description="Rigid multi-part mechanisms (doors, gates, fans, switches, "
                    "etc.) are imported with a dedicated mechanism armature. "
                    "Static node transforms come from the BIN node-matrix "
                    "table, while ABIN frame 0 overrides only the channels it "
                    "actually records. This preserves authored rest rotations "
                    "and pivots and keeps the mechanism convenient to edit "
                    "and animate in Blender. Translation- and rotation-driven "
                    "mechanisms are both supported.")
    normals: BoolProperty(
        name="Import normals", default=True,
        description="Use the file's normals as custom split normals. Was "
                    "hardcoded off for every whole-area import until now - "
                    "real bug, not a deliberate default: without this "
                    "Blender falls back to auto-smoothing, which is exactly "
                    "what put a visible seam down the middle of symmetric "
                    "models instead of the smooth authored normals hiding it")
    split_parts: BoolProperty(
        name="Split into connected parts", default=False,
        description="A room .bin packs many unrelated pieces (walls, floor, "
                    "every prop) into one shared vertex buffer purely for "
                    "storage. Splitting on real shared-vertex topology (not "
                    "just material) turns each into its own object - useful "
                    "if you want individual props as separate objects to "
                    "work with. Was briefly defaulted on here while chasing "
                    "a reported per-room normals inconsistency; that turned "
                    "out to not be a real bug (a Blender Solid-shading-mode "
                    "display difference, not the underlying data or this "
                    "addon), so back to off by default - splitting a whole "
                    "area into hundreds of small objects per room is a "
                    "deliberate opt-in, not a default")
    vcol: BoolProperty(
        name="Import vertex colours", default=True,
        description="Level geometry stores its baked lighting per vertex. "
                    "Was ALSO hardcoded off for every whole-area import - "
                    "same bug as normals above, not a deliberate default. "
                    "Without this a map renders flat")
    lod: EnumProperty(
        name="LOD", items=[
            ("AUTO", "Highest detail only (recommended for a whole area)", ""),
            ("SEPARATE", "All, as separate objects", ""),
            ("MERGE", "All, merged into one object (old behaviour)", ""),
        ], default="AUTO",
        description="Room LOD variants occupy the SAME space (unlike actor "
                    "LODs) - measured on c1__pa_warehouseb.bin: LOD0 (21 "
                    "submeshes) and LOD5 (1 submesh, a crude single-mesh "
                    "impostor) share almost the exact same bounding box. "
                    "Importing every level at once for a whole area means "
                    "every one of those overlapping impostors renders on top "
                    "of the real room - that IS the 'fan of huge triangles' "
                    "artifact. AUTO avoids it; only pick SEPARATE here to "
                    "inspect the LOD levels themselves, not for a normal import")
    collision: EnumProperty(
        name="Collision hulls", items=[
            ("SKIP", "Skip", ""),
            ("SEPARATE", "Separate collection", ""),
            ("MERGE", "Merge in", ""),
        ], default="SEPARATE")

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "scale")
        col.prop(self, "flip_v")
        col.prop(self, "normals")
        col.prop(self, "vcol")
        col.prop(self, "split_parts")
        col.prop(self, "textures")
        col.prop(self, "skydome")
        col.prop(self, "import_lights")
        col.prop(self, "import_actors")
        if self.import_actors:
            col.prop(self, "actor_skeleton")
            col.prop(self, "animate_mechanisms")
        col.prop(self, "lod")
        col.prop(self, "collision")

    def execute(self, context):
        # This file's whole job is picking a .lvt and being one click - every
        # actual behaviour (directory scan, .lvt reading, LOD/collision
        # handling, light/actor placement) lives in one place: the whole-area
        # path of import_scene.swbh_bin. Calling it here instead of copying
        # ~150 lines keeps there being exactly one implementation to keep
        # correct.
        area_dir = os.path.dirname(self.filepath)
        return bpy.ops.import_scene.swbh_bin(
            "EXEC_DEFAULT",
            directory=area_dir, files=[], filepath="",
            whole_area=True, skydome=self.skydome,
            import_lights=self.import_lights, import_actors=self.import_actors,
            actor_skeleton=self.actor_skeleton,
            animate_mechanisms=self.animate_mechanisms,
            scale=self.scale, flip_v=self.flip_v, textures=self.textures,
            lod=self.lod, collision=self.collision,
            normals=self.normals, vcol=self.vcol,
            split_parts=self.split_parts,
            skeleton=False, import_animations=False)



class IMPORT_OT_swbh_abin(bpy.types.Operator, ImportHelper):
    """Bake .abin animations onto the selected SWBH armature"""
    bl_idname = "import_scene.swbh_abin"
    bl_label = "Import SWBH .abin"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".abin"
    filter_glob: StringProperty(default="*.abin", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={"HIDDEN"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN"})

    fps: FloatProperty(
        name="Sample rate", default=30.0, min=1.0, max=120.0,
        description="Keys are resampled onto this frame rate. The originals "
                    "were authored at about 30 fps")
    scale: FloatProperty(
        name="Scale", default=1.0, min=0.001, max=1000.0,
        description="Must match the scale the skeleton was imported with")
    gap: IntProperty(
        name="Gap between clips (frames)", default=10, min=0,
        description="Blank frames inserted between clips when several are "
                    "imported at once")
    anim_start: IntProperty(
        name="Start at #", default=0, min=0,
        description="Skip this many clips before importing. Selected files "
                    "are sorted alphabetically first, so this is stable")
    anim_count: IntProperty(
        name="How many (0 = all)", default=0, min=0,
        description="Import at most this many clips, starting from 'Start at #'. "
                    "0 imports every remaining selected clip")
    zero_root_translation: BoolProperty(
        name="Freeze root in place", default=False,
        description="Keep the root bone's position at the bind pose and only "
                    "bake its rotation - use this if a clip's root motion isn't "
                    "wanted for this preview/edit")
    root_bone_name: StringProperty(
        name="Root bone (blank = auto)", default="",
        description="Leave blank to auto-use the parentless bone; set this if "
                    "that guess is wrong for this rig")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE"

    def execute(self, context):
        arm = context.active_object
        if arm is None or arm.type != "ARMATURE":
            self.report({"ERROR"}, "select the SWBH armature first")
            return {"CANCELLED"}

        base = arm.get("swbh_base_abin", "")
        if not base or not os.path.isfile(base):
            self.report({"ERROR"},
                        "this armature has no linked *_base.abin - re-import the "
                        "model with 'Import skeleton' enabled")
            return {"CANCELLED"}

        try:
            rest_pose = bind_pose_from_abin(base)
        except (BinParseError, struct.error) as exc:
            self.report({"ERROR"}, "bind pose: %s" % exc)
            return {"CANCELLED"}

        # rebuild the node order from the armature itself
        nodes = []
        index = {b.name: i for i, b in enumerate(arm.data.bones)}
        for b in arm.data.bones:
            nodes.append((b.name, index[b.parent.name] if b.parent else -1))

        paths = ([os.path.join(self.directory, f.name) for f in self.files]
                 if self.files else [self.filepath])
        paths = sorted(p for p in paths if p.lower().endswith(".abin")
                       and not p.lower().endswith("_base.abin"))
        if not paths:
            self.report({"ERROR"}, "no animation .abin selected")
            return {"CANCELLED"}

        n_found = len(paths)
        if self.anim_start or self.anim_count:
            end = self.anim_start + self.anim_count if self.anim_count else None
            paths = paths[self.anim_start:end]
        if not paths:
            self.report({"ERROR"}, "start/count skipped every selected clip")
            return {"CANCELLED"}

        _track, ok, total = import_animation_clips(
            arm, nodes, rest_pose, paths, self.fps, self.scale, self.gap,
            arm.name + "_anim", self.report, context=context,
            zero_root_translation=self.zero_root_translation,
            root_bone_name=self.root_bone_name)

        if not ok:
            self.report({"ERROR"}, "no animation imported")
            return {"CANCELLED"}

        context.scene.frame_start = 1
        context.scene.frame_end = total

        self.report({"INFO"}, "imported %d/%d clip(s), %d frames total - "
                              "not muted, cuts cleanly between clips - "
                              "markers on the Timeline mark each clip's start"
                              % (ok, n_found, total))
        return {"FINISHED"}



class IMPORT_OT_swbh_ats(bpy.types.Operator, ImportHelper):
    """Import a complete ATS-described actor/prop using the existing BIN core."""
    bl_idname = "import_scene.swbh_ats"
    bl_label = "Import SWBH Asset (.ats)"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".ats"
    filter_glob: StringProperty(default="*.ats", options={"HIDDEN"})

    scale: FloatProperty(name="Scale", default=1.0, min=0.001, max=1000.0)
    textures: BoolProperty(name="Find textures", default=True)
    skeleton: BoolProperty(name="Import skeleton", default=True)
    animations: BoolProperty(name="Import animations", default=False)
    collision: BoolProperty(name="Import collision helpers", default=True)
    vcol: BoolProperty(name="Import vertex colours", default=True)
    normals: BoolProperty(name="Import normals", default=True)

    def draw(self, context):
        c = self.layout.column()
        c.prop(self, "scale")
        c.prop(self, "textures")
        c.prop(self, "skeleton")
        if self.skeleton:
            c.prop(self, "animations")
        c.prop(self, "vcol")
        c.prop(self, "normals")
        c.prop(self, "collision")

    def execute(self, context):
        ats_path = self.filepath
        info = parse_ats(ats_path)
        root = prefs_root() or find_data_root(os.path.dirname(ats_path)) or ""
        rel = resolve_ats_asset(ats_path, info, root)
        bp = rel.get("bin")
        if not bp:
            model = info.get("values", {}).get("modelFile", "")
            self.report({"ERROR"}, "ATS model not found: %s" % model)
            return {"CANCELLED"}

        before = set(bpy.context.scene.objects)
        result = bpy.ops.import_scene.swbh_bin(
            "EXEC_DEFAULT", filepath=bp, files=[], directory=os.path.dirname(bp),
            whole_area=False, scale=self.scale, flip_v=True, normals=self.normals,
            vcol=self.vcol, split_parts=False, textures=self.textures,
            skeleton=self.skeleton, import_animations=self.animations,
            animate_mechanisms=True, lod="SEPARATE", collision="SEPARATE")
        if "FINISHED" not in result:
            return result
        created = [o for o in bpy.context.scene.objects if o not in before]
        mesh_objs = [o for o in created if o.type == "MESH"]
        if not mesh_objs:
            self.report({"WARNING"}, "BIN imported but no mesh object was created")
            return {"FINISHED"}

        for obj in created:
            if obj.type in {"MESH", "ARMATURE"}:
                apply_asset_metadata(obj, ats_path, info, rel)
        if rel.get("col") and self.collision:
            try:
                helpers = add_collision_visuals(rel["col"], mesh_objs[0], scale=self.scale)
                print("[SWBH] ATS collision helpers: %d" % len(helpers))
            except Exception as exc:
                print("[SWBH] ATS collision import failed: %s" % exc)
        self.report({"INFO"}, "Imported ATS asset: %s" % info.get("values", {}).get("label", os.path.basename(ats_path)))
        return {"FINISHED"}


class IMPORT_OT_swbh_mat(bpy.types.Operator, ImportHelper):
    """Create a Blender material directly from a SWBH .mat descriptor."""
    bl_idname = "import_scene.swbh_mat"
    bl_label = "Import SWBH Material (.mat)"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".mat"
    filter_glob: StringProperty(default="*.mat", options={"HIDDEN"})

    def execute(self, context):
        path = self.filepath
        info = parse_mat(path)
        root = prefs_root() or find_data_root(os.path.dirname(path)) or ""
        tex = info.get("diffusemap", "")
        tex_index = build_index(root)[0] if root and os.path.isdir(root) else {}
        tex_path = (locate(os.path.splitext(tex)[0], os.path.dirname(path), "textures", TEX_EXTS, tex_index)
                    if tex else None)
        mat = make_material(os.path.splitext(os.path.basename(path))[0], tex_path,
                            info.get("shadertype", ""), False,
                            metadata=info)
        mat["swbh_source_mat"] = path
        bpy.context.view_layer.objects.active = context.active_object if context.active_object else None
        if context.active_object and context.active_object.type == "MESH":
            context.active_object.data.materials.append(mat)
        self.report({"INFO"}, "Created material %s" % mat.name)
        return {"FINISHED"}


class IMPORT_OT_swbh_col(bpy.types.Operator, ImportHelper):
    """Import a text .col as hidden collision helper empties."""
    bl_idname = "import_scene.swbh_col"
    bl_label = "Import SWBH Collision (.col)"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".col"
    filter_glob: StringProperty(default="*.col", options={"HIDDEN"})

    scale: FloatProperty(name="Scale", default=1.0, min=0.001, max=1000.0)

    def execute(self, context):
        data = parse_col(self.filepath)
        if data.get("binary"):
            self.report({"WARNING"}, "Binary .col variant is preserved but not decoded yet")
            return {"FINISHED"}
        created = add_collision_visuals(self.filepath, scale=self.scale)
        self.report({"INFO"}, "Imported %d collision helper(s)" % len(created))
        return {"FINISHED"}


class IMPORT_OT_swbh_aset(bpy.types.Operator, ImportHelper):
    """Load an ASET and import its referenced ABIN clips onto the active armature."""
    bl_idname = "import_scene.swbh_aset"
    bl_label = "Import SWBH Animation Set (.aset)"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".aset"
    filter_glob: StringProperty(default="*.aset", options={"HIDDEN"})
    fps: FloatProperty(name="Sample rate", default=30.0, min=1.0, max=120.0)
    scale: FloatProperty(name="Scale", default=1.0, min=0.001, max=1000.0)
    gap: IntProperty(name="Gap between clips", default=10, min=0)

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "ARMATURE"

    def execute(self, context):
        arm = context.active_object
        if arm is None or arm.type != "ARMATURE":
            self.report({"ERROR"}, "select an SWBH armature first")
            return {"CANCELLED"}
        aset = parse_aset(self.filepath)
        base_dir = os.path.dirname(self.filepath)
        root = prefs_root() or find_data_root(base_dir) or ""
        paths = []
        for entry in aset["entries"]:
            ref = entry["path"].replace("\\", os.sep).replace("/", os.sep)
            candidate = ref if os.path.isabs(ref) else os.path.join(root, ref.lstrip(os.sep)) if root else os.path.join(base_dir, ref)
            if os.path.isfile(candidate):
                paths.append(candidate)
                continue
            found = find_related(root, os.path.basename(ref), ".abin", preferred_dir=base_dir)
            if found:
                paths.append(found)
        # Include the base clip first when it was explicitly present.
        unique = []
        seen = set()
        for p in paths:
            key = os.path.abspath(p).lower()
            if key not in seen:
                seen.add(key); unique.append(p)
        if not unique:
            self.report({"ERROR"}, "no .abin files resolved from ASET")
            return {"CANCELLED"}

        base = arm.get("swbh_base_abin", "")
        if not base or not os.path.isfile(base):
            self.report({"WARNING"}, "armature has no swbh_base_abin; ASET can only apply onto rigs with a known bind pose")
            return {"CANCELLED"}
        try:
            rest_pose = bind_pose_from_abin(base)
        except (BinParseError, struct.error) as exc:
            self.report({"ERROR"}, "bind pose: %s" % exc)
            return {"CANCELLED"}
        nodes = []
        idx = {b.name: i for i, b in enumerate(arm.data.bones)}
        for b in arm.data.bones:
            nodes.append((b.name, idx[b.parent.name] if b.parent else -1))
        _track, n_ok, total = import_animation_clips(
            arm, nodes, rest_pose, unique, self.fps, self.scale, self.gap,
            arm.name + "_aset", self.report, context=context)
        if not n_ok:
            self.report({"ERROR"}, "no ASET animations imported")
            return {"CANCELLED"}
        context.scene.frame_start = 1
        context.scene.frame_end = total
        arm["swbh_source_aset"] = self.filepath
        arm["swbh_aset_entries"] = len(aset["entries"])
        self.report({"INFO"}, "Imported %d/%d ASET animation clips" % (n_ok, len(unique)))
        return {"FINISHED"}


class IMPORT_OT_swbh_lvr(bpy.types.Operator, ImportHelper):
    """Import actor placement markers from an LVR without rebuilding geometry."""
    bl_idname = "import_scene.swbh_lvr"
    bl_label = "Import SWBH Actor Placements (.lvr)"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".lvr"
    filter_glob: StringProperty(default="*.lvr", options={"HIDDEN"})
    scale: FloatProperty(name="Scale", default=1.0, min=0.001, max=1000.0)

    def execute(self, context):
        data = parse_lvr(self.filepath)
        col = get_collection(os.path.splitext(os.path.basename(self.filepath))[0] + "_Actors")
        made = 0
        for a in data["actors"]:
            e = bpy.data.objects.new(a["name"], None)
            e.empty_display_type = "ARROWS"
            e.empty_display_size = 0.5
            x, y, z = a["pos"]
            e.location = (x * self.scale, y * self.scale, z * self.scale)
            e.rotation_mode = "QUATERNION"
            w, qx, qy, qz = a["rot"]
            e.rotation_quaternion = Quaternion((w, qx, qy, qz))
            e["swbh_actor_id"] = a["id"]
            e["swbh_actor_class"] = a["cls"]
            e["swbh_source_lvr"] = self.filepath
            e["swbh_areas"] = a.get("areas", [])
            col.objects.link(e)
            made += 1
        self.report({"INFO"}, "Imported %d LVR actor placement marker(s)" % made)
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_swbh_bin.bl_idname,
                         text="SW: Bounty Hunter Model (.bin)")
    self.layout.operator(IMPORT_OT_swbh_lvt.bl_idname,
                         text="SW: Bounty Hunter Level (.lvt)")
    self.layout.operator(IMPORT_OT_swbh_abin.bl_idname,
                         text="SW: Bounty Hunter Animation (.abin)")


CLASSES = (SWBHPreferences, IMPORT_OT_swbh_ats, IMPORT_OT_swbh_bin, IMPORT_OT_swbh_lvt,
          IMPORT_OT_swbh_lvr, IMPORT_OT_swbh_abin, IMPORT_OT_swbh_aset,
          IMPORT_OT_swbh_mat, IMPORT_OT_swbh_col)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
