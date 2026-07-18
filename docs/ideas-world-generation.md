# Idea — Claude-driven world generation on Pyrite64 (recorded 2026-07-18)

Future-step material, not scheduled against any milestone — goes with
`docs/ideas.md` and `docs/ideas-m7-living-npcs.md`, not the M-series plan.
Prompted by: "could you build a version of pyrite64 you could hook into,
so I could describe worlds/games and you'd just do it — hand you a 2D map,
you build a 3D world?" This doc records the feasibility research done in
response, not a committed design.

**"World gen" here means design-time and permanent, not runtime.** Per
Luke's clarification: the idea is Claude authoring a world that gets
*baked into* the Pyrite64 project and compiled into the ROM — the same
way `trainer/make_canned_blob.py` bakes the NPC brain's weights in — not
a roguelike-style generator running live on the N64. This matters beyond
phrasing: N64 has no writable filesystem for user content and no network
asset streaming, so *everything* the console ever shows has to exist as
compiled cart data already. That constraint is exactly what a
design-time Python pipeline naturally produces — it isn't a workaround,
it's the native shape of the problem.

**Verdict: feasible today, and nobody appears to be doing this yet** for
Pyrite64 specifically (checked upstream repo/docs/forums — see
References). General prior art for LLM-driven 3D scene generation exists
(SceneCraft, LLplace) and validates the same overall shape of pipeline.
It has never been pointed at an N64 target.

## What's actually there (verified against the local SDK + this repo)

- `game/project.p64proj` and `game/data/scenes/*/scene.json` are **plain
  JSON**, not a proprietary binary. Confirmed by reading this project's
  own scene file directly:

  ```json
  // project.p64proj
  {
    "name": "64gpt",
    "pathEmu": "ares",
    "romName": "64gpt",
    "sceneIdLastOpened": 1,
    "sceneIdOnBoot": 1,
    "sceneIdOnReset": 1
  }
  ```

  Scene nodes carry `pos` / `rot` / `scale`, a `components` list (Camera,
  Light, Model, custom "Code" script components), and models are
  referenced by UUID pointing at an asset's `.conf` file:

  ```json
  // data/scenes/1/scene.json — one node, trimmed
  {
    "name": "Camera",
    "pos": [-280.9, 315.0, 130.9],
    "rot": [-0.286, -0.498, -0.178, 0.799],
    "scale": [0.9999, 0.9999, 0.9999],
    "components": [
      { "name": "Camera", "data": { "fov": 65.0, "near": 100.0, "far": 5000.0, "vpSize": [320, 240] } }
    ]
  }
  ```

  A `Model (Static)` component references an asset by UUID plus inline
  material overrides (color, lighting flags, env/fresnel) — everything a
  scene needs to place and shade an object is legible, plain-text data.

- **The asset pipeline is already CLI-only**, no GUI required:
  `gltf_to_t3d`, `mkmodel`, `mkmaterial`, `mkasset` (all in
  `$HOME/pyrite64-sdk/bin/`) take flags and files, glTF in → N64-ready
  `t3dm`/`model64` out. E.g.:

  ```
  gltf_to_t3d <in.gltf> <out.t3dm> [--bvh] [--base-scale=64] [--asset-path=assets]
  mkmodel [-o dir] [-c level] <input files...>
  mkmaterial [-o path] [-t texdb path] <file.mat>...
  ```

- **The Pyrite64 app's own CLI exposes exactly one verb**: `--cli --cmd
  build` (confirmed via `pyrite64 --help` and the upstream CLI docs —
  no `--cmd` value beyond `build` exists). There's no "add node" or
  "import scene" command. This doesn't block anything, though: the scene
  format is just data, so nothing stops writing the JSON directly and
  treating that as the API, then invoking `--cli --cmd build` to compile
  and booting in Ares to verify — the exact same headless-build +
  screenshot loop this project already uses for ROM milestones.

- **Caveat**: this project's own `CLAUDE.md` currently marks
  `project.p64proj` and `game/data/` as editor-owned / hands-off, to keep
  the NPC-brain milestone scope tight. That's a 64gpt-specific
  convention, not a technical wall in Pyrite64 itself — a world-gen
  experiment would need that rule deliberately relaxed, and probably
  belongs in its own project rather than layered onto 64gpt.

## The real gap: content, not tooling

Tooling is not the hard part; sourcing N64-appropriate 3D content is.
Three strategies considered:

1. **Procedural blockout from a 2D tile grid** — generate wall/floor/
   ceiling geometry algorithmically from tile types (à la SM64-style
   greybox levels). No asset database required at all; fits N64's tiny
   poly/texture budget by construction, since the generator can be
   written to respect the budget directly. **Fastest path to a working
   "describe it → boots in Ares" loop**, and the recommended starting
   point.
2. **Curated low-poly CC0 prop packs** (Kenney.nl-style asset sets) for
   decoration, selected by tile semantics (a "tree" tile pulls from a
   forest pack, etc.). Pre-optimized, glTF-native, safe. Layer in after
   (1) is proven.
3. **AI text-to-3D generation** (Meshy, Hyper3D, Sloyd) — considered and
   **not recommended** for this use case. These target general
   real-time engines, not N64's actual constraints (tiny TMEM, low tri
   counts, this codebase's integer-only pipeline elsewhere). Output
   would need heavy manual retopology and palette reduction to fit,
   undermining the "just build it" premise of the whole idea. Revisit
   only if (1)+(2) prove the loop and more polygon/texture headroom is
   later found.

## Proposed pipeline (not built, not scheduled)

```
2D tile map + description
        │
        ▼
procedural blockout generator (new, off this project)
  → geometry for walls/floor/ceiling per tile
  → prop placement from curated pack, by tile tag
        │
        ▼
emit scene.json + project.p64proj directly (bypass GUI editor)
        │
        ▼
existing CLI asset pipeline (gltf_to_t3d / mkmodel / mkmaterial)
        │
        ▼
pyrite64 --cli --cmd build   →  .z64
        │
        ▼
boot in Ares, screenshot, verify   (same loop as ROM milestones)
```

## Object generation & import pipeline

Luke's follow-up ask: work out a *simple, cheap* pipeline for generating
individual objects at N64-right size/quality, importing them, and
placing them in the scene JSON — and figure out whether Claude-written
Python helpers (and the local opencoder worker) can do this
automatically. Design, not yet built:

**Hybrid sourcing, chosen per object's repetition count:**

| Tier | Source | Why |
|---|---|---|
| Background props placed many times (trees, rocks, crates, fences, pillars) | **Procedural parametric primitives**, generated straight to glTF | Zero asset dependency, dimensions are literal function parameters so "right-sized for N64" is true by construction, fully deterministic and free to regenerate |
| Hero / unique set-pieces (a specific building, a statue, a named landmark) | **Curated CC0 low-poly pack** (Kenney.nl-style), imported and normalized | More visual variety than primitives can cheaply give; these are placed once so the per-instance budget cost is affordable |

**A budget table drives both paths** instead of hoping — e.g. background
props capped around 50–150 triangles and a shared/atlased small texture,
hero objects allowed a few hundred triangles. The table is a frozen
design decision (mine to set, not the generator's to guess), and both
the primitive generator and the pack importer are required to enforce it
before a model is allowed into the scene.

**Materials — verified, not guessed.** Inspected this repo's own
`game/assets/box.glb` binary directly (parsed its glTF JSON chunk) rather
than trusting prose about the format. Finding: its material is not
standard glTF PBR (no `baseColorFactor`/metallic-roughness) — it carries
two custom glTF extensions, `FAST64_materials_n64` and
`FAST64_materials_f3d`, encoding the N64 RDP's fixed-function combiner
stages, geometry mode flags, and lights directly. This is what Blender's
Fast64 plugin emits and what `mkmodel`/`gltf_to_t3d` expect **if** you
want per-mesh material control. Reverse-engineering that extension
schema by hand would make the "cheap simple" primitive generator not
cheap or simple. The actual escape hatch is already sitting in the CLI,
found earlier when reading `gltf_to_t3d --help`:
`--ignore-materials` — *"Ignore F3D materials and write dummy data,
useful for custom material systems."* Procedural primitives should use
this: emit plain geometry (positions/normals/vertex-colors only, no
FAST64 extensions) and drive appearance entirely through the scene.json
`Model` component's inline `material` override — confirmed present in
this project's own `scene.json` (`prim` color, `lighting`, `env`,
`fresnel`, `setDepth`, etc.) — rather than baking material state into
each generated mesh. Curated hero assets, by contrast, should keep their
Fast64-authored materials intact (don't strip what's already correct).

**Tooling shape** (sketch, not written):

```
worldgen/primitives.py     parametric mesh generators (box, cylinder, cone,
                            prism, simple deform) → glTF, sized in the
                            project's real world/tile units
worldgen/asset_import.py   takes a curated-pack glTF, checks it against the
                            budget table, auto-scales to a target bounding
                            box, runs gltf_to_t3d / mkmodel / mkmaterial,
                            writes the asset .conf + registers its UUID
worldgen/scene_writer.py   takes a 2D tile grid + a tile→prop-tag manifest,
                            emits scene.json nodes (pos/rot/scale + Model
                            component referencing the right asset UUID)
worldgen/build.py          orchestrates: generate/import assets → write
                            scene → `pyrite64 --cli --cmd build` → boot
                            check in Ares
```

**Where the local opencoder worker fits**: this project already runs
Engineering Lead Mode for exactly this shape of work (`.ai/tasks/`,
`qwen-worker`, DeepSeek-Coder as the draft author). The architecture
stays mine to set — the glTF/scene JSON schema, the budget table, and
the verification gate (stays in budget *and* boots in Ares) are frozen
interfaces, not delegatable. But writing N parametric primitive
generator functions against one fixed glTF-writing helper, or writing
the tile→prop-tag manifest boilerplate, is exactly the large,
mechanically-verifiable, self-contained work the lead-mode contract flow
exists for — a natural `qwen-worker` dispatch once this is actually
being built, following the memory-audit ritual and the reference_files
lesson from past tasks (point it at one hand-written example, don't make
it guess a schema from prose).

This is design only — nothing above is built or scheduled yet.

## Cross-checked against a second write-up (Grok, 2026-07-18)

Luke shared an independent technical doc (from Grok) covering the same
territory via the "human" pipeline — Blender + the **Fast64** plugin
exporting glTF, imported through Pyrite64's editor. Worth folding in:
Fast64 (`github.com/Fast-64/fast64`) is the canonical tool artists use
to author N64-correct materials in Blender, and it's *exactly* what
produced the `FAST64_materials_n64`/`FAST64_materials_f3d` extensions
found in `box.glb` above — so that doc and this repo's own asset agree
on the same ground truth. Confirmed matches:

- **Textures are external PNG files**, not embedded in the glTF/glb —
  matches this repo's own `assets/p64/font.ia4.png` sitting next to its
  `.conf` as a sibling file, not packed inside a binary asset.
- **Ares (recent version) is the accurate emulator**, Project64 is not
  reliable for this engine — matches the workflow already established
  in this repo's `CLAUDE.md`.
- Low-poly / TMEM-aware texture budgets — matches the budget-table
  approach above.

Two claims in that doc contradict direct evidence and should go back to
Grok as questions rather than be taken on faith:

1. **"No built-in 2D support (WIP)."** This repo's own
   `game/src/user/DialogueDemo.cpp` calls `DrawLayer::use2D()` today, and
   `scene.json` has a first-class `layers2D` render layer — 2D rendering
   is already in active use in a shipped milestone of this project. Ask
   Grok: is this claim about a missing *2D editor mode* (e.g. no 2D-only
   scene/tilemap authoring UI) rather than the renderer itself lacking
   2D drawing? The doc doesn't cite a source for this line.
2. **"HDR + Bloom" as a world-building tip.** No corroboration found —
   this repo's `scene.json` render-layer config (`fbFormat`, `blender`
   int codes) shows no HDR framebuffer or bloom-pass fields, and HDR
   framebuffers are unusual for real N64 output hardware (no floating
   point, 8-bit-per-channel color out). Ask Grok for a source/citation;
   until then, treat as unverified rather than folding it into the
   budget/quality plan above.

A follow-up worth asking Grok directly: does `mkmodel`/`gltf_to_t3d`
accept a *plain* PBR glTF material (no `FAST64_materials_*` extensions)
when **not** passing `--ignore-materials` — i.e. does it degrade
gracefully, or does the Fast64 extension effectively become mandatory
the moment you want material data honored at all? That determines
whether the primitive generator's "skip materials, drive appearance via
scene.json only" plan above is the *only* cheap path, or merely the
simplest one.

## Recommendation

Start with procedural-blockout-only, as a standalone experiment in its
own project/branch — it sidesteps the asset-library problem entirely and
proves the "describe it, get a booting ROM" loop fastest. Layer in
curated prop packs for decoration second. Skip AI mesh generation for
now; it's the weakest link in the chain given N64's actual hardware
budget.

**Explicit goal from Luke: this should look beautiful, not just boot.**
"Correct blockout" is the proof-of-loop milestone, not the bar for the
idea itself — the actual target is worlds that look good on real N64
hardware. Pyrite64's renderer is tiny3d (the `t3dm` format everything
gets converted to via `gltf_to_t3d`), which exposes real lighting —
the scene/material JSON already inspected above has fields for ambient +
directional lights, fog (color/min/max/mode per render layer), and
per-material env/fresnel/lighting flags, not just flat-shaded polygons.
Getting to "beautiful" on N64's budget is a real design problem in its
own right (careful light/fog use, a cohesive palette across the curated
prop packs, texture reuse to stay inside TMEM) — worth its own pass once
the blockout loop proves out, not something to bolt on at the end.

## References

- [Pyrite64 GitHub (HailToDodongo/pyrite64)](https://github.com/HailToDodongo/pyrite64) — upstream engine/editor
- [Pyrite64 documentation](https://hailtododongo.github.io/pyrite64/)
- [Pyrite64 CLI manual](https://hailtododongo.github.io/pyrite64/docs/manual/cli.html) — confirms `build` is the only documented `--cmd` verb
- [SceneCraft — LLM agent synthesizing 3D scenes as Blender code](https://arxiv.org/html/2403.01248v1)
- [LLplace — 3D indoor scene layout/editing via LLM, JSON-based](https://arxiv.org/pdf/2406.03866)
- [Meshy Low Poly Mode (text-to-3D, game-ready export)](https://www.meshy.ai/tutorials/make-low-poly-3d-models)
- Local SDK inspected directly: `$HOME/pyrite64-sdk/bin/{mkmodel,mkmaterial,gltf_to_t3d,mkasset}`,
  this repo's `game/project.p64proj` and `game/data/scenes/1/scene.json`
