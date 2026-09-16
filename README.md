# Star Wars: Bounty Hunter — TVG Modding Toolkit 1.1.2

Blender add-on for importing **models, animations and level areas** from extracted Star Wars: Bounty Hunter game data.

The public workflow is intentionally limited to three importers. Supporting files are resolved automatically from the unpacked data tree and are not meant to be imported manually.

## Public importers

### Model — `.bin`

Imports a game model and automatically resolves supporting data when available:

- textures;
- `.mat` material descriptors;
- shader type information;
- alpha / cutout materials;
- additive / glow-style materials;
- reflection-style materials;
- normal maps using the game's naming convention;
- vertex-colour lighting;
- skeleton / skinning;
- LODs;
- rigid mechanism data for doors, gates and similar objects.

### Animation — `.abin`

Imports animation clips onto the selected SWBH armature. Existing clip/NLA behavior is preserved.

### Level — `.lvt`

Imports the whole extracted level area and automatically resolves:

- level geometry `.bin` parts;
- textures and `.mat` materials;
- baked vertex lighting;
- level lights and world settings;
- placed actors and props;
- actor models through their `.bin` files;
- actor materials and textures;
- actor skeletons where supported;
- rigid mechanisms such as doors.

The `.lvr`, `.ats`, `.aset`, `.mat` and `.col` files are internal supporting data for the level/model workflow and are not separate public import targets.

## Door / rigid mechanism handling

Rigid mechanisms are treated differently from character skeletons, but they **keep a dedicated Blender armature** so doors, gates, fans, switches and similar objects remain convenient to edit and animate.

Version 1.1.2 fixes the mechanism rest pose at its actual source instead of guessing from sparse animation tracks:

- the `.bin` per-node matrix table supplies static node position, orientation and hierarchy;
- frame 0 of the matching `.abin` overrides only the channels that the animation actually records;
- missing animation channels keep their authored `.bin` value instead of being replaced by zero translation or identity rotation;
- rigid submeshes sharing one vertex buffer are separated by their rigid-node id before synthetic armature weights are assigned;
- the level actor's `.lvt` position/rotation stays on the mechanism armature object, so one mechanism asset works at every authored level angle.

This matters for sliding doors such as `O_B_HorizontalDoorC/D`, whose animation contains translation only: a static rest rotation must not be destroyed just because the `.abin` has no rotation track. The same path also supports rotation-driven hinge/spin mechanisms.

The importer still falls back to ABIN frame-0 transforms when a BIN node-matrix table cannot be decoded.

## 1.1.2 mechanism-rest fix

The door probe confirmed that `LVT` and `LVR` placement agree for tested B1/B1b/B3 horizontal doors and that the C/D door clips animate only the local translations of `left_panel` and `right_panel`. The remaining rest orientation therefore has to come from the model hierarchy, not from a fabricated global +90° correction.

1.1.2 decodes and validates the BIN node-matrix pair dynamically (local/world ordering, storage transpose and inverse form are selected by hierarchy consistency and ABIN frame-0 agreement). This is intentionally general rather than hard-coded to one door class or one level.

## Materials

The importer reads the material descriptor associated with each mesh material and builds the Blender shader automatically.

Supported shader behavior includes:

- lit materials;
- unlit materials;
- alpha cutout;
- alpha blend / transparency;
- additive / glow-style materials;
- reflection-style materials using Blender's environment reflections as the representation;
- baked vertex-colour lighting;
- normal-map textures when the matching texture exists.

The goal is to keep the extracted game asset as close as practical to the source appearance without requiring manual material setup.

## Level workflow

1. Extract the game's data first.
2. Keep the extracted `Data` / `Data1` directory structure intact.
3. In Blender choose **File → Import → SW: Bounty Hunter Model (.bin)**, **Level (.lvt)** or **Animation (.abin)**.
4. The importer searches the extracted tree for supporting files automatically.

A data root can also be configured in **Preferences → Add-ons → SWBH Toolkit** when automatic detection is not sufficient.

## Notes

- The importer is designed around the extracted game-data layout.
- Supporting descriptors do not need to be imported separately.
- Level LOD handling defaults to the highest-detail room geometry so overlapping lower-detail room variants do not render on top of the normal room.
- Collision data is treated as supporting information and can be kept in the hidden SWBH Collision collection.

## Files

- `io_import_swbh.py` — Blender importer.
- `buny_extract.py` — archive extraction utility.
