#!/usr/bin/env python3
"""
Block Texture Comparison Tool for arnis
========================================
Compares Minecraft block textures against their mapped Luanti equivalents
(minetest_game and Mineclonia) using CNN feature extraction.

Generates HTML reports showing block pairs sorted by texture distance,
and suggests better matches for poorly-matched blocks.

Prerequisites:
    ./setup.sh          # clones game repos + downloads MC client.jar
    # Then generate registered_nodes.txt dumps via the Luanti dump mod
    # (see luanti_dump_mod/ — install as worldmod, start each game once)

Usage:
    python compare_textures.py [--game mtg|mcl|both]
"""

import argparse
import base64
import io
import os
import re
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration — paths relative to this script (populated by setup.sh)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ARNIS_ROOT = HERE.parent
BLOCK_MAP_RS = Path(os.environ.get("BLOCK_MAP_RS", ARNIS_ROOT / "src" / "luanti_block_map.rs"))

MC_TEXTURES_DIR = HERE / "minecraft" / "extracted" / "assets" / "minecraft" / "textures" / "block"
MC_ENTITY_DIR = HERE / "minecraft" / "extracted" / "assets" / "minecraft" / "textures" / "entity"
# Fallback to legacy directory name
if not MC_TEXTURES_DIR.exists():
    MC_TEXTURES_DIR = HERE / "Minecraft 1.21.8 Block Textures" / "extracted" / "assets" / "minecraft" / "textures" / "block"
    MC_ENTITY_DIR = HERE / "Minecraft 1.21.8 Block Textures" / "extracted" / "assets" / "minecraft" / "textures" / "entity"

# Game mod directories (setup.sh clones these)
MTG_MODS = HERE / "minetest_game" / "mods"
MCL_MODS = HERE / "mineclonia" / "mods"
# Fallback to system-wide installs
if not MTG_MODS.exists():
    MTG_MODS = Path("/home/linu/.minetest/games/minetest_game/mods")
if not MCL_MODS.exists():
    MCL_MODS = Path("/home/linu/.minetest/games/mineclonia/mods")

REPORT_DIR = HERE / "reports"

# Minimum alpha value to consider a pixel opaque
ALPHA_THRESHOLD = 128
# Grid size for sub-region feature extraction on transparent textures
GRID_SIZE = 4
# Minimum fraction of opaque pixels in a grid cell to include it
MIN_CELL_COVERAGE = 0.25

# ---------------------------------------------------------------------------
# Feature extraction model (MobileNetV2 — tiny & fast)
# ---------------------------------------------------------------------------
_model = None
_transform = None


def get_model():
    global _model, _transform
    if _model is None:
        print("Loading MobileNetV2 feature extractor...")
        weights = models.MobileNet_V2_Weights.DEFAULT
        full_model = models.mobilenet_v2(weights=weights)
        full_model.eval()
        # Use only the feature layers (no classifier) — output is 1280-dim
        _model = torch.nn.Sequential(
            full_model.features,
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        if torch.cuda.is_available():
            _model = _model.cuda()
        _model.eval()
        _transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _model, _transform


# ---------------------------------------------------------------------------
# Word embedding model (tiny BERT for block name similarity)
# ---------------------------------------------------------------------------
_embed_model = None
_embed_tokenizer = None
_embed_cache: dict[str, torch.Tensor] = {}


def get_embed_model():
    """Load a small sentence embedding model (all-MiniLM-L6-v2 via transformers)."""
    global _embed_model, _embed_tokenizer
    if _embed_model is None:
        try:
            from transformers import AutoModel, AutoTokenizer
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"Loading word embedding model: {model_name}...")
            _embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _embed_model = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                _embed_model = _embed_model.cuda()
            _embed_model.eval()
        except Exception as e:
            print(f"Warning: could not load embedding model: {e}")
            return None, None
    return _embed_model, _embed_tokenizer


def _block_name_to_words(name: str) -> str:
    """Convert a block name like 'mcl_core:dark_oak_planks' to readable words."""
    # Strip mod prefix
    if ":" in name:
        name = name.split(":", 1)[1]
    # Replace underscores with spaces
    return name.replace("_", " ")


def embed_name(name: str) -> torch.Tensor | None:
    """Get sentence embedding for a block name."""
    if name in _embed_cache:
        return _embed_cache[name]
    model, tokenizer = get_embed_model()
    if model is None:
        return None
    words = _block_name_to_words(name)
    inputs = tokenizer(words, return_tensors="pt", truncation=True, padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling over token embeddings
    token_embs = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    embedding = (token_embs * mask).sum(1) / mask.sum(1)
    embedding = embedding.squeeze(0).cpu()
    _embed_cache[name] = embedding
    return embedding


def name_distance(name1: str, name2: str) -> float:
    """Cosine distance between two block name embeddings."""
    e1 = embed_name(name1)
    e2 = embed_name(name2)
    if e1 is None or e2 is None:
        return 0.5  # neutral fallback
    sim = torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
    return 1.0 - sim.item()
    return _model, _transform


def average_color(img: Image.Image):
    """Compute average RGB color, excluding transparent pixels."""
    img_rgba = img.convert("RGBA")
    data = np.array(img_rgba)
    mask = data[:, :, 3] >= ALPHA_THRESHOLD
    if not mask.any():
        return (128.0, 128.0, 128.0)  # neutral gray for fully transparent
    opaque = data[mask][:, :3].astype(np.float64)
    return tuple(opaque.mean(axis=0))


def color_distance(c1, c2):
    """Euclidean distance between two RGB colors."""
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2) ** 0.5


def cosine_distance(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Cosine distance (1 - cosine_similarity) between two feature vectors."""
    sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    return 1.0 - sim.item()


def _opaque_ratio(img_rgba: Image.Image) -> float:
    """Fraction of pixels that are opaque (alpha >= threshold)."""
    alpha = np.array(img_rgba.split()[3])
    return (alpha >= ALPHA_THRESHOLD).sum() / alpha.size


def _composite_on_avg(img_rgba: Image.Image) -> Image.Image:
    """Composite RGBA image onto a background of its own average opaque color."""
    avg = average_color(img_rgba)
    bg = Image.new("RGB", img_rgba.size, tuple(int(c) for c in avg))
    bg.paste(img_rgba, mask=img_rgba.split()[3])
    return bg


def _find_opaque_grid_cells(img_rgba: Image.Image):
    """Split image into a grid, return bounding boxes of cells with enough opaque content."""
    w, h = img_rgba.size
    alpha = np.array(img_rgba.split()[3])
    cell_w = max(1, w // GRID_SIZE)
    cell_h = max(1, h // GRID_SIZE)
    cells = []
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            x0, y0 = gx * cell_w, gy * cell_h
            x1, y1 = min(x0 + cell_w, w), min(y0 + cell_h, h)
            cell_alpha = alpha[y0:y1, x0:x1]
            coverage = (cell_alpha >= ALPHA_THRESHOLD).sum() / cell_alpha.size
            if coverage >= MIN_CELL_COVERAGE:
                cells.append((x0, y0, x1, y1))
    return cells


def extract_features(img: Image.Image) -> torch.Tensor:
    """Extract a 1280-dim feature vector from a PIL image.

    For textures with significant transparency (< 90% opaque), splits the
    image into a grid and extracts features only from cells that contain
    enough non-transparent pixels. Features are area-weighted averaged.
    Transparent pixels within selected cells are composited onto the cell's
    average opaque color to avoid CNN artifacts from black/white backgrounds.
    """
    model, transform = get_model()
    use_cuda = torch.cuda.is_available()
    img_rgba = img.convert("RGBA")

    if _opaque_ratio(img_rgba) >= 0.9:
        tensor = transform(img_rgba.convert("RGB")).unsqueeze(0)
        if use_cuda:
            tensor = tensor.cuda()
        with torch.no_grad():
            return model(tensor).squeeze(0).cpu()

    cells = _find_opaque_grid_cells(img_rgba)
    if not cells:
        # Fully transparent — fall back to full-image RGB
        tensor = transform(img_rgba.convert("RGB")).unsqueeze(0)
        if use_cuda:
            tensor = tensor.cuda()
        with torch.no_grad():
            return model(tensor).squeeze(0).cpu()

    features, weights = [], []
    for x0, y0, x1, y1 in cells:
        crop_rgba = img_rgba.crop((x0, y0, x1, y1))
        crop_rgb = _composite_on_avg(crop_rgba)
        tensor = transform(crop_rgb).unsqueeze(0)
        if use_cuda:
            tensor = tensor.cuda()
        with torch.no_grad():
            feat = model(tensor).squeeze(0).cpu()
        area = (x1 - x0) * (y1 - y0)
        features.append(feat * area)
        weights.append(area)

    return sum(features) / sum(weights)


# ---------------------------------------------------------------------------
# Texture loading — Minecraft (multi-face)
# ---------------------------------------------------------------------------

# Face suffixes to try for MC blocks (in priority order)
MC_FACE_SUFFIXES = ["", "_top", "_side", "_front", "_bottom", "_back"]

# Special entity textures (model-based blocks like chests)
MC_ENTITY_MAP = {
    "chest": "chest/normal.png",
    "ender_chest": "chest/ender.png",
    "trapped_chest": "chest/trapped.png",
}


def _mc_base_name(name: str) -> str:
    """Strip shape suffixes to get the base material name."""
    for suffix in ("_wall", "_fence", "_fence_gate", "_slab", "_stairs",
                   "_pressure_plate", "_button", "_sign",
                   "_door", "_trapdoor"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _mc_texture_candidates(name: str) -> list[str]:
    """Generate candidate texture base names for a Minecraft block name."""
    base = _mc_base_name(name)
    candidates = [name]
    if base != name:
        candidates += [base, f"{base}_planks", f"{base}_block", f"{base}s"]

    # Carpet → wool
    if name.endswith("_carpet"):
        candidates.append(name[: -len("_carpet")] + "_wool")

    # Potted plants → flower name
    if name.startswith("potted_"):
        candidates.append(name[len("potted_"):])

    # smooth_X variants (applied to both original name and stripped base)
    for n in {name, base}:
        if n.startswith("smooth_"):
            inner = n[len("smooth_"):]
            candidates += [f"{inner}_top", f"{inner}_block_bottom", inner]

    # Water / lava
    if name in ("water", "lava"):
        candidates += [f"{name}_still", f"{name}_flow"]

    # Crops: try stage variants (highest stage = mature)
    if name in ("carrots", "potatoes", "wheat", "beetroots", "sweet_berries"):
        for stage in range(7, -1, -1):
            candidates.append(f"{name}_stage{stage}")

    # X_block → X (e.g., snow_block → snow)
    if name.endswith("_block"):
        candidates.append(name[: -len("_block")])

    # Quartz special cases
    if "quartz" in name and base == "quartz":
        candidates += ["quartz_block_side", "quartz_block_top"]

    # Also try planks / block / front / side suffixes on original name
    candidates += [f"{name}_planks", f"{name}_block", f"{name}_front", f"{name}_side"]
    return list(dict.fromkeys(candidates))  # deduplicate preserving order


def load_mc_textures(mc_block_name: str) -> list[Image.Image]:
    """Load all face textures for a Minecraft block (multi-face).
    Returns a list of PIL images, one per unique face texture found."""
    # Try entity textures first (model-based blocks like chests)
    if mc_block_name in MC_ENTITY_MAP:
        entity_path = MC_ENTITY_DIR / MC_ENTITY_MAP[mc_block_name]
        if entity_path.exists():
            return [Image.open(entity_path)]

    images = []
    seen_paths = set()

    for base in _mc_texture_candidates(mc_block_name):
        for suffix in MC_FACE_SUFFIXES:
            path = MC_TEXTURES_DIR / f"{base}{suffix}.png"
            if path.exists() and path not in seen_paths:
                seen_paths.add(path)
                images.append(Image.open(path))
        if images:
            break  # found at least one face for this base name

    return images


# ---------------------------------------------------------------------------
# Texture loading — Luanti (from game source via Lua parsing + fallback)
# ---------------------------------------------------------------------------

def build_texture_index(mods_dir: Path) -> dict[str, Path]:
    """Build filename → full_path index of all textures in a game's mods."""
    index = {}
    for tex_path in mods_dir.rglob("*.png"):
        if "/textures/" in str(tex_path) or "\\textures\\" in str(tex_path):
            index[tex_path.name] = tex_path
    return index


def parse_lua_tiles(mods_dir: Path) -> dict[str, list[str]]:
    """Parse Lua source files to extract node_name → [tile_texture_names].

    Best-effort regex-based parser covering common registration patterns.
    Returns a dict mapping node names to lists of texture filenames.
    """
    node_tiles = {}

    for lua_file in mods_dir.rglob("*.lua"):
        try:
            content = lua_file.read_text(errors="replace")
        except Exception:
            continue

        # --- Pattern 1: Direct register_node calls ---
        for m in re.finditer(
            r'(?:minetest|core)\.register_node\(\s*"([^"]+)"',
            content,
        ):
            node_name = m.group(1).lstrip(":")
            rest = content[m.end():]
            tiles = _extract_tiles_block(rest)
            if tiles:
                node_tiles[node_name] = tiles

        # --- Pattern 2: mcl_trees.register_wood("name", { tree={tiles=...}, ... }) ---
        for m in re.finditer(r'mcl_trees\.register_wood\(\s*"([^"]+)"', content):
            tree_name = m.group(1)
            rest = content[m.end():]
            block = _extract_brace_block(rest)
            if not block:
                continue

            # Determine calling mod name from file path
            calling_mod = _mod_name_from_path(lua_file, mods_dir)

            # Extract sub-components: tree, wood, leaves, sapling
            found_components = set()
            for component, prefix in [("tree", "tree"), ("wood", "wood"),
                                       ("leaves", "leaves"), ("sapling", "sapling")]:
                comp_match = re.search(rf'\b{component}\s*=\s*\{{', block)
                if comp_match:
                    comp_block = _extract_brace_block(block[comp_match.start():])
                    if comp_block:
                        tiles = _extract_png_strings(comp_block)
                        if tiles:
                            node_tiles[f"mcl_trees:{prefix}_{tree_name}"] = tiles
                            found_components.add(component)

            # Generate defaults for components without explicit tile overrides
            if calling_mod:
                if "tree" not in found_components:
                    node_tiles[f"mcl_trees:tree_{tree_name}"] = [
                        f"{calling_mod}_log_{tree_name}_top.png",
                        f"{calling_mod}_log_{tree_name}.png",
                    ]
                if "wood" not in found_components:
                    node_tiles[f"mcl_trees:wood_{tree_name}"] = [
                        f"{calling_mod}_planks_{tree_name}.png",
                    ]
                if "leaves" not in found_components:
                    node_tiles[f"mcl_trees:leaves_{tree_name}"] = [
                        f"{calling_mod}_leaves_{tree_name}.png",
                    ]

        # --- Pattern 3: xpanes.register_pane("name", { textures = {...} }) ---
        for m in re.finditer(r'xpanes\.register_pane\(\s*"([^"]+)"', content):
            pane_name = m.group(1)
            rest = content[m.end():]
            # Find textures field
            tex_match = re.search(r'\btextures\s*=\s*\{', rest)
            if tex_match:
                block = rest[tex_match.end():]
                close = block.find("}")
                if close > 0:
                    tiles = _extract_png_strings(block[:close])
                    if tiles:
                        node_tiles[f"xpanes:{pane_name}_flat"] = tiles
                        node_tiles[f"xpanes:{pane_name}"] = tiles

        # --- Pattern 4: Concatenation in loops with register_node ---
        # Pattern: for VAR in/= ... register_node("mod:"..VAR, { tiles = {"prefix_"..VAR..".png"} })
        # We detect the template and expand with common color/material names
        _parse_loop_registrations(content, node_tiles)

    return node_tiles


def _extract_brace_block(text: str) -> str | None:
    """Extract the content within the first { ... } block, handling nesting."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, min(len(text), start + 10000)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None


def _mod_name_from_path(lua_file: Path, mods_dir: Path) -> str | None:
    """Determine the mod name from a Lua file's path.
    Walks up from the file looking for a directory containing mod.conf or init.lua."""
    current = lua_file.parent
    while current != mods_dir and current != current.parent:
        if (current / "mod.conf").exists() or (current / "init.lua").exists():
            return current.name
        current = current.parent
    return None


def _extract_png_strings(text: str) -> list[str]:
    """Extract all .png filename strings from a Lua code block."""
    textures = []
    for tex_m in re.finditer(r'"([^"]*\.png[^"]*)"', text):
        tex = tex_m.group(1).split("^")[0].strip()
        if tex:
            textures.append(tex)
    return textures


def _extract_tiles_block(text: str) -> list[str]:
    """Find tiles = { ... } in text and extract texture filenames."""
    tiles_match = re.search(r'\btiles\s*=\s*\{', text)
    if not tiles_match:
        return []
    # Check tiles is within this registration (before next register_node)
    next_reg = re.search(r'(?:minetest|core)\.register_node\(', text)
    if next_reg and tiles_match.start() > next_reg.start():
        return []
    block = _extract_brace_block(text[tiles_match.start():])
    return _extract_png_strings(block) if block else []


# Common color/material names used in loop registrations
_COMMON_COLORS = [
    "white", "grey", "silver", "light_grey", "dark_grey", "black",
    "red", "orange", "yellow", "green", "dark_green", "lime",
    "blue", "light_blue", "cyan", "magenta", "pink", "purple", "brown",
]

_COMMON_MATERIALS = [
    "wood", "stone", "cobble", "brick", "sandstone", "desert_stone",
    "desert_cobble", "desert_sandstone", "obsidian", "stonebrick",
    "stone_brick", "red_sandstone",
]


def _parse_loop_registrations(content: str, node_tiles: dict):
    """Detect loop-based node registrations with string concatenation.

    Handles patterns like:
        for _, color in ipairs(colors) do
            register_node("mcl_wool:" .. color, { tiles = {"wool_" .. color .. ".png"} })
    """
    # Find for-loop blocks that contain register_node with concatenation
    for m in re.finditer(
        r'(?:minetest|core)\.register_node\(\s*"([^"]*?)"\s*\.\.\s*(\w+)',
        content,
    ):
        node_prefix = m.group(1).lstrip(":")
        var_name = m.group(2)
        rest = content[m.end():]

        # Find tiles pattern with same variable
        tiles_match = re.search(r'\btiles\s*=\s*\{', rest[:2000])
        if not tiles_match:
            continue
        tiles_block = rest[tiles_match.end():tiles_match.end() + 500]
        close = tiles_block.find("}")
        if close < 0:
            continue
        tiles_str = tiles_block[:close]

        # Check if tile uses the loop variable
        if f".. {var_name}" in tiles_str or f"..{var_name}" in tiles_str:
            # Extract the template: "prefix_" .. var .. ".png"  →  "prefix_{}.png"
            tile_template = re.search(
                r'"([^"]*?)"\s*\.\.\s*' + re.escape(var_name) + r'\s*\.\.\s*"([^"]*)"',
                tiles_str,
            )
            if tile_template:
                prefix = tile_template.group(1)
                suffix = tile_template.group(2)
                # Expand with common values
                for val in _COMMON_COLORS + _COMMON_MATERIALS:
                    node_name = f"{node_prefix}{val}"
                    tex_name = f"{prefix}{val}{suffix}".split("^")[0].strip()
                    if tex_name:
                        node_tiles.setdefault(node_name, []).append(tex_name)

    # Also handle "mod:" .. var .. "_suffix" pattern
    for m in re.finditer(
        r'(?:minetest|core)\.register_node\(\s*"([^"]*?)"\s*\.\.\s*(\w+)\s*\.\.\s*"([^"]*?)"',
        content,
    ):
        node_prefix = m.group(1).lstrip(":")
        var_name = m.group(2)
        node_suffix = m.group(3)
        rest = content[m.end():]

        tiles_match = re.search(r'\btiles\s*=\s*\{', rest[:2000])
        if not tiles_match:
            continue
        tiles_block = rest[tiles_match.end():tiles_match.end() + 500]
        close = tiles_block.find("}")
        if close < 0:
            continue
        tiles_str = tiles_block[:close]

        if f".. {var_name}" in tiles_str or f"..{var_name}" in tiles_str:
            tile_template = re.search(
                r'"([^"]*?)"\s*\.\.\s*' + re.escape(var_name) + r'\s*\.\.\s*"([^"]*)"',
                tiles_str,
            )
            if tile_template:
                prefix = tile_template.group(1)
                suffix = tile_template.group(2)
                for val in _COMMON_COLORS + _COMMON_MATERIALS:
                    node_name = f"{node_prefix}{val}{node_suffix}"
                    tex_name = f"{prefix}{val}{suffix}".split("^")[0].strip()
                    if tex_name:
                        node_tiles.setdefault(node_name, []).append(tex_name)


def load_luanti_textures(node_name: str, lua_tiles: dict, tex_index: dict) -> list[Image.Image]:
    """Load all face textures for a Luanti node.
    Uses Lua-parsed tiles first, falls back to naming conventions."""
    images = []

    # Strategy 1: Use Lua-parsed tiles
    tile_names = lua_tiles.get(node_name, [])
    for tex_name in tile_names:
        if tex_name in tex_index:
            try:
                images.append(Image.open(tex_index[tex_name]))
            except Exception:
                pass

    if images:
        return images

    # Strategy 2: Smart naming convention fallbacks
    candidates = []
    if ":" in node_name:
        mod, name = node_name.split(":", 1)
        # Direct: mod_name.png
        candidates.append(f"{mod}_{name}.png")

        # grey/gray normalization (MCL uses gray in textures, grey in node names)
        if "grey" in name:
            candidates.append(f"{mod}_{name.replace('grey', 'gray')}.png")
        if "gray" in name:
            candidates.append(f"{mod}_{name.replace('gray', 'grey')}.png")

        # Stairs/slabs use the base material texture
        for prefix in ("stair_", "slab_"):
            if name.startswith(prefix):
                base = name[len(prefix):]
                candidates.append(f"default_{base}.png")
                candidates.append(f"{base}.png")
                candidates.append(f"mcl_core_{base}.png")
                # Plural form: mud_brick → mcl_mud_bricks.png
                candidates.append(f"mcl_mud_{base}s.png")
                candidates.append(f"mcl_{base}s.png")
                # Try broader mod search with plural
                for mod_prefix in ["mcl_mud", "mcl_nether", "mcl_core", "mcl_blackstone"]:
                    candidates.append(f"{mod_prefix}_{base}.png")
                    candidates.append(f"{mod_prefix}_{base}s.png")
                # quartzblock → mcl_nether_quartz_block_side.png
                candidates.append(f"mcl_nether_{base}_side.png")
                candidates.append(f"mcl_nether_quartz_{base.replace('quartz', '')}_side.png")
                # blackstone_brick_polished → mcl_blackstone_polished_bricks.png
                if "polished" in base:
                    rearranged = base.replace("_polished", "").replace("_brick", "")
                    candidates.append(f"mcl_blackstone_polished_{rearranged}s.png")
                    candidates.append(f"mcl_blackstone_polished_bricks.png")
                # slab_stone_double → default_stone.png
                if base.endswith("_double"):
                    real_base = base[:-7]
                    candidates.append(f"default_{real_base}.png")
                    candidates.append(f"mcl_core_{real_base}.png")

        # Doors: door_wood_b/a → doors_door_wood.png
        if name.endswith(("_a", "_b", "_b_1", "_b_2", "_t_1", "_t_2")):
            base = re.sub(r'(_[abt](?:_[12])?)$', '', name)
            candidates.append(f"{mod}_{base}.png")
            # MCL door textures: mcl_doors_door_X_lower.png
            candidates.append(f"{mod}_door_{base.replace('_door', '')}_lower.png")
            candidates.append(f"{mod}_door_{base.replace('_door', '')}_upper.png")
            # wooden_door → door_wood
            if "wooden" in base:
                candidates.append(f"{mod}_door_wood_lower.png")

        # Trapdoors: mcl_doors:X_trapdoor → mcl_doors_trapdoor_X.png
        if name.endswith("_trapdoor"):
            wood = name[:-9]  # strip _trapdoor
            candidates.append(f"{mod}_trapdoor_{wood}.png")
            candidates.append(f"{mod}_{name}.png")

        # Carpet: mcl_wool:X_carpet → wool_X.png
        if name.endswith("_carpet"):
            color = name[:-7]
            candidates.append(f"wool_{color}.png")
            candidates.append(f"mcl_wool_{color}.png")

        # Walls: walls:cobble → default_cobble.png
        if mod == "walls":
            candidates.append(f"default_{name}.png")

        # Xpanes: bar_flat → xpanes_bar.png
        if name.endswith("_flat"):
            candidates.append(f"{mod}_{name[:-5]}.png")
            candidates.append(f"xpanes_top_{name[:-5].replace('pane_', 'glass_')}.png")

        # Copper blocks: block_oxidized → mcl_copper_oxidized.png (loop-registered)
        if mod == "mcl_copper" and name.startswith("block_"):
            suffix = name[6:]  # strip "block_"
            candidates.append(f"mcl_copper_{suffix}.png")

        # Lanterns: lantern_floor → mcl_lanterns_lantern.png
        if name.startswith("lantern_"):
            candidates.append(f"{mod}_lantern.png")

        # Anvils: anvil_damage_1 → mcl_anvils_anvil_top_damaged_1.png
        if "anvil_damage" in name:
            dmg = name.split("_")[-1]
            candidates.append(f"{mod}_anvil_top_damaged_{dmg}.png")

        # Brewing stand: just use the burner texture
        if "stand_" in name and mod == "mcl_brewing":
            candidates.append(f"{mod}_burner.png")

        # Double flowers: double_fern → mcl_flowers_double_plant_fern_bottom.png
        if name.startswith("double_") and not name.endswith("_top"):
            plant = name[7:]
            candidates.append(f"{mod}_double_plant_{plant}_bottom.png")
        if name.endswith("_top") and "double_" in name:
            plant = name[7:-4]
            candidates.append(f"{mod}_double_plant_{plant}_top.png")

        # Farming crops: carrot_7 → farming_carrot_4.png (MCL renumbers stages)
        if mod == "mcl_farming":
            base_match = re.match(r"(\w+?)_(\d+)$", name)
            if base_match:
                crop = base_match.group(1)
                stage = int(base_match.group(2))
                # Modern MCL format: mcl_farming_{crop}_stage_{N}.png
                candidates.append(f"mcl_farming_{crop}_stage_{stage}.png")
                # Plural form (potatoes)
                candidates.append(f"mcl_farming_{crop}s_stage_{stage}.png")
                candidates.append(f"mcl_farming_{crop}es_stage_{stage}.png")
                # Try nearby stages too
                for stage_offset in [0, -1, -2, -3, 1]:
                    s = stage + stage_offset
                    if s >= 0:
                        candidates.append(f"farming_{crop}_{s}.png")
                        candidates.append(f"{mod}_{crop}_{s}.png")
                        candidates.append(f"mcl_farming_{crop}_stage_{s}.png")
                        candidates.append(f"mcl_farming_{crop}s_stage_{s}.png")
                        candidates.append(f"mcl_farming_{crop}es_stage_{s}.png")

        # Chest: default:chest → default_chest_front.png
        candidates.append(f"{mod}_{name}_front.png")
        candidates.append(f"{mod}_{name}_top.png")

        # Carts/rail: carts:rail → carts_rail_straight.png
        candidates.append(f"{mod}_{name}_straight.png")

    for cand in candidates:
        if cand in tex_index:
            try:
                images.append(Image.open(tex_index[cand]))
                return images  # one hit is enough for fallback
            except Exception:
                pass

    # Strategy 3: Fuzzy search — find texture files containing the meaningful part
    if ":" in node_name:
        _, name = node_name.split(":", 1)
        # Strip common prefixes to get the material name
        for prefix in ("stair_", "slab_", "door_", "wall_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        # Strip suffixes
        name = re.sub(r'(_[abt](?:_[12])?)$', '', name)
        name = re.sub(r'_flat$', '', name)

        # Try compound word splitting (e.g., "stonebrick" → "stone_brick")
        name_variants = [name]
        # Insert underscore at common word boundaries
        for split_word in ("stone", "sand", "brick", "block", "glass", "wood", "iron", "gold"):
            if split_word in name and f"{split_word}_" not in name and name != split_word:
                idx = name.index(split_word) + len(split_word)
                if idx < len(name):
                    name_variants.append(name[:idx] + "_" + name[idx:])

        for variant in name_variants:
            for tex_name, tex_path in tex_index.items():
                if f"_{variant}." in f"_{tex_name}" or f"_{variant}_" in tex_name:
                    try:
                        images.append(Image.open(tex_path))
                        return images
                    except Exception:
                        pass

    return images


# ---------------------------------------------------------------------------
# Parse luanti_block_map.rs
# ---------------------------------------------------------------------------

def parse_block_map(func_name: str) -> list[dict]:
    """Parse a mapping function from luanti_block_map.rs.
    Returns list of {id, mc_name, luanti_node}."""
    with open(BLOCK_MAP_RS) as f:
        content = f.read()

    start = content.find(f"fn {func_name}(")
    if start == -1:
        return []
    func_body = content[start:]
    next_fn = func_body.find("\nfn ", 1)
    if next_fn > 0:
        func_body = func_body[:next_fn]

    mappings = []
    for line in func_body.split("\n"):
        stripped = line.strip()

        # Extract MC block name from inline comment: // mc_block_name
        mc_name = None
        comment_match = re.search(r"//\s*(.+?)(?:\s*\(.*\))?\s*$", stripped)
        if comment_match:
            mc_name = comment_match.group(1).strip()

        # Extract block ID and Luanti node name
        # Pattern: N => "node:name",
        # Pattern: N => return conv_*(props, "node:name", ...),
        id_match = re.match(r"(\d+)\s*(?:\|[^=]*)?\s*=>", stripped)
        if not id_match:
            continue

        block_id = int(id_match.group(1))

        # Find all "mod:node" strings on this line
        nodes = re.findall(r'"([a-z][a-z_]*:[a-z][a-z0-9_]*)"', stripped)
        if not nodes:
            continue

        # For conv_trapdoor, first string is the closed variant
        # For conv_stair, first string is the node name
        luanti_node = nodes[0]

        if mc_name:
            mappings.append({
                "id": block_id,
                "mc_name": mc_name,
                "luanti_node": luanti_node,
            })

    return mappings


# ---------------------------------------------------------------------------
# Image → base64 for HTML embedding
# ---------------------------------------------------------------------------

def img_to_b64(img: Image.Image, size: int = 64) -> str:
    """Convert PIL image to base64 data URI (PNG)."""
    img_resized = img.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img_resized.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def placeholder_b64(size: int = 64) -> str:
    """Generate a gray placeholder image as base64."""
    img = Image.new("RGB", (size, size), (128, 128, 128))
    return img_to_b64(img, size)


# ---------------------------------------------------------------------------
# Multi-face comparison
# ---------------------------------------------------------------------------

def compute_face_features(images: list[Image.Image]) -> tuple:
    """Compute average color and CNN features across a list of face images.
    Returns (avg_color, avg_features, primary_image)."""
    if not images:
        return None, None, None

    colors = [average_color(img) for img in images]
    feats = [extract_features(img) for img in images]

    avg_color = tuple(sum(c[i] for c in colors) / len(colors) for i in range(3))
    avg_feat = sum(feats) / len(feats)
    return avg_color, avg_feat, images[0]


def optimal_face_distance(mc_imgs: list[Image.Image], lt_imgs: list[Image.Image]) -> tuple:
    """Compute distance between MC and Luanti blocks using optimal face matching.

    For each MC face texture, find the best matching Luanti face. The final
    distance is the average of these best-match distances.

    Returns (color_dist, cosine_dist, combined_dist).
    """
    mc_data = [(average_color(img), extract_features(img)) for img in mc_imgs]
    lt_data = [(average_color(img), extract_features(img)) for img in lt_imgs]

    total_cdist = 0.0
    total_fdist = 0.0
    for mc_color, mc_feat in mc_data:
        best_cdist = float("inf")
        best_fdist = float("inf")
        best_combined = float("inf")
        for lt_color, lt_feat in lt_data:
            cd = color_distance(mc_color, lt_color)
            fd = cosine_distance(mc_feat, lt_feat)
            combined = 0.3 * (cd / 441.7) + 0.7 * fd
            if combined < best_combined:
                best_cdist = cd
                best_fdist = fd
                best_combined = combined
        total_cdist += best_cdist
        total_fdist += best_fdist

    n = len(mc_data)
    return total_cdist / n, total_fdist / n, 0.3 * (total_cdist / n / 441.7) + 0.7 * (total_fdist / n)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_game(game_label: str, func_name: str, mods_dir: Path):
    """Analyze texture similarity for one game target."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {game_label}")
    print(f"{'=' * 60}")

    mappings = parse_block_map(func_name)
    print(f"Parsed {len(mappings)} block mappings from {func_name}")

    print("Parsing Lua source for node→texture mappings...")
    lua_tiles = parse_lua_tiles(mods_dir)
    print(f"  Parsed {len(lua_tiles)} node tile definitions from Lua source")

    print("Building texture file index...")
    tex_index = build_texture_index(mods_dir)
    print(f"  Indexed {len(tex_index)} texture files")

    # Precompute all Luanti texture data for auto-match
    print("Loading Luanti textures for auto-match candidates...")
    luanti_textures = {}
    for node_name in lua_tiles:
        imgs = load_luanti_textures(node_name, lua_tiles, tex_index)
        if imgs:
            color, feat, primary = compute_face_features(imgs)
            if color and feat is not None:
                luanti_textures[node_name] = {
                    "img": primary,
                    "avg_color": color,
                    "features": feat,
                }

    # Also try naming convention for nodes not found via Lua parsing
    for tex_name, tex_path in tex_index.items():
        # Reverse: mod_name.png → mod:name
        base = tex_name.rsplit(".", 1)[0]
        parts = base.split("_", 1)
        if len(parts) == 2:
            node_name = f"{parts[0]}:{parts[1]}"
            if node_name not in luanti_textures:
                try:
                    img = Image.open(tex_path)
                    color = average_color(img)
                    feat = extract_features(img)
                    luanti_textures[node_name] = {
                        "img": img,
                        "avg_color": color,
                        "features": feat,
                    }
                except Exception:
                    pass

    print(f"  {len(luanti_textures)} Luanti nodes with textures available for matching")

    # Pre-compute name embeddings for all candidates
    print("Computing name embeddings for Luanti nodes...")
    for cand_node in luanti_textures:
        embed_name(cand_node)
    print(f"  Cached {len(_embed_cache)} name embeddings")

    # Analyze each mapping
    results = []
    print("Comparing block pairs...")
    for m in mappings:
        mc_name = m["mc_name"]
        luanti_node = m["luanti_node"]

        mc_imgs = load_mc_textures(mc_name)
        lt_imgs = load_luanti_textures(luanti_node, lua_tiles, tex_index)

        # Compute name embedding distance for current pair
        mc_embed_name = mc_name or ""
        ndist = name_distance(mc_embed_name, luanti_node) if mc_embed_name else 0.5

        if not mc_imgs or not lt_imgs:
            lt_data = luanti_textures.get(luanti_node)
            results.append({
                **m,
                "mc_img": mc_imgs[0] if mc_imgs else None,
                "lt_img": lt_data["img"] if lt_data else (lt_imgs[0] if lt_imgs else None),
                "color_dist": float("inf"),
                "cosine_dist": float("inf"),
                "name_dist": ndist,
                "combined_dist": float("inf"),
                "top5": [],
            })
            continue

        cdist, fdist, combined = optimal_face_distance(mc_imgs, lt_imgs)

        # Compute MC aggregate for auto-match comparison
        mc_color, mc_feat, _ = compute_face_features(mc_imgs)

        # Find top-5 best matching Luanti textures
        candidates = []
        for cand_node, cand_data in luanti_textures.items():
            c_cdist = color_distance(mc_color, cand_data["avg_color"])
            c_fdist = cosine_distance(mc_feat, cand_data["features"])
            c_ndist = name_distance(mc_embed_name, cand_node) if mc_embed_name else 0.5
            # Combined: 25% color + 55% CNN + 20% name embedding
            c_combined = 0.25 * (c_cdist / 441.7) + 0.55 * c_fdist + 0.20 * c_ndist
            candidates.append({
                "node": cand_node,
                "dist": c_combined,
                "color_dist": c_cdist,
                "cosine_dist": c_fdist,
                "name_dist": c_ndist,
                "img": cand_data["img"],
            })

        candidates.sort(key=lambda c: c["dist"])
        top5 = candidates[:5]

        results.append({
            **m,
            "mc_img": mc_imgs[0],
            "lt_img": lt_imgs[0],
            "color_dist": cdist,
            "cosine_dist": fdist,
            "name_dist": ndist,
            "combined_dist": 0.25 * (cdist / 441.7) + 0.55 * fdist + 0.20 * ndist,
            "top5": top5,
        })

    # Sort by combined distance (worst first)
    results.sort(key=lambda r: -r["combined_dist"])

    return results


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_html_report(results: list, game_label: str, output_path: Path):
    """Generate an HTML report with inline texture images and expandable top-5."""
    placeholder = placeholder_b64()

    rows_html = []
    for i, r in enumerate(results):
        mc_b64 = img_to_b64(r["mc_img"]) if r["mc_img"] else placeholder
        lt_b64 = img_to_b64(r["lt_img"]) if r["lt_img"] else placeholder

        # Color-code distance
        dist = r["combined_dist"]
        if dist == float("inf"):
            color = "#999"
            dist_str = "N/A"
        elif dist < 0.15:
            color = "#2d8a2d"  # green — good match
            dist_str = f"{dist:.3f}"
        elif dist < 0.30:
            color = "#b8860b"  # amber — ok match
            dist_str = f"{dist:.3f}"
        else:
            color = "#cc3333"  # red — bad match
            dist_str = f"{dist:.3f}"

        ndist_str = f"{r.get('name_dist', 0.5):.3f}"

        # Build top-5 expandable section
        top5 = r.get("top5", [])
        top5_html = ""
        if top5:
            top5_rows = []
            for j, cand in enumerate(top5):
                cand_b64 = img_to_b64(cand["img"]) if cand.get("img") else placeholder
                improvement = dist - cand["dist"] if dist != float("inf") else 0
                imp_str = f" (↓{improvement:.3f})" if improvement > 0.01 else ""
                top5_rows.append(f"""
                    <tr>
                        <td>{j+1}</td>
                        <td><img src="{cand_b64}" width="40" height="40" style="image-rendering:pixelated"></td>
                        <td><code>{cand["node"]}</code></td>
                        <td>{cand["dist"]:.3f}{imp_str}</td>
                        <td>{cand["color_dist"]:.1f}</td>
                        <td>{cand["cosine_dist"]:.3f}</td>
                        <td>{cand.get("name_dist", 0.5):.3f}</td>
                    </tr>""")
            top5_html = f"""
            <tr class="top5-row" id="top5-{i}" style="display:none">
                <td colspan="10" style="padding:8px 24px;background:#222">
                    <table style="width:auto;margin:0">
                        <tr><th>#</th><th>Tex</th><th>Node</th><th>Dist</th>
                            <th>Color Δ</th><th>CNN Δ</th><th>Name Δ</th></tr>
                        {"".join(top5_rows)}
                    </table>
                </td>
            </tr>"""

        best_node = top5[0]["node"] if top5 else ""
        btn_html = f"""<td><button onclick="toggleTop5({i})" class="expand-btn"
                          id="btn-{i}">▶ 5</button></td>""" if top5 else "<td></td>"

        rows_html.append(f"""
        <tr>
            <td>{r["id"]}</td>
            <td><img src="{mc_b64}" width="48" height="48" style="image-rendering:pixelated"></td>
            <td><code>{r["mc_name"]}</code></td>
            <td><img src="{lt_b64}" width="48" height="48" style="image-rendering:pixelated"></td>
            <td><code>{r["luanti_node"]}</code></td>
            <td style="color:{color}; font-weight:bold">{dist_str}</td>
            <td>{r["color_dist"]:.1f}</td>
            <td>{r["cosine_dist"]:.3f}</td>
            <td>{ndist_str}</td>
            {btn_html}
        </tr>{top5_html}""")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Block Texture Comparison — {game_label}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               margin: 20px; background: #1a1a1a; color: #ddd; }}
        h1 {{ color: #fff; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th {{ background: #333; color: #fff; padding: 8px 12px; text-align: left;
              position: sticky; top: 0; z-index: 1; }}
        td {{ padding: 6px 12px; border-bottom: 1px solid #333; vertical-align: middle; }}
        tr:hover {{ background: #2a2a2a; }}
        code {{ background: #333; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; }}
        img {{ border: 1px solid #555; border-radius: 2px; }}
        .legend {{ margin: 10px 0 20px; }}
        .legend span {{ margin-right: 20px; }}
        .stats {{ background: #222; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .expand-btn {{
            background: #444; color: #ddd; border: 1px solid #666; border-radius: 4px;
            cursor: pointer; padding: 2px 8px; font-size: 0.8em;
        }}
        .expand-btn:hover {{ background: #555; }}
        .top5-row td {{ border-bottom: 2px solid #555; }}
        .top5-row table {{ border-collapse: collapse; }}
        .top5-row th {{ background: #2a2a2a; font-size: 0.85em; padding: 4px 8px; }}
        .top5-row td {{ padding: 4px 8px; font-size: 0.9em; border-bottom: 1px solid #333; }}
    </style>
    <script>
    function toggleTop5(idx) {{
        var row = document.getElementById('top5-' + idx);
        var btn = document.getElementById('btn-' + idx);
        if (row.style.display === 'none') {{
            row.style.display = '';
            btn.textContent = '▼ 5';
        }} else {{
            row.style.display = 'none';
            btn.textContent = '▶ 5';
        }}
    }}
    </script>
</head>
<body>
    <h1>🧱 Block Texture Comparison — {game_label}</h1>
    <div class="legend">
        <span style="color:#2d8a2d">■ Good (&lt;0.15)</span>
        <span style="color:#b8860b">■ Okay (0.15–0.30)</span>
        <span style="color:#cc3333">■ Poor (&gt;0.30)</span>
    </div>
    <div class="stats">
        <strong>Total pairs:</strong> {len(results)} |
        <strong>Good:</strong> {sum(1 for r in results if r["combined_dist"] < 0.15)} |
        <strong>Okay:</strong> {sum(1 for r in results if 0.15 <= r["combined_dist"] < 0.30)} |
        <strong>Poor:</strong> {sum(1 for r in results if r["combined_dist"] >= 0.30 and r["combined_dist"] != float("inf"))} |
        <strong>Missing texture:</strong> {sum(1 for r in results if r["combined_dist"] == float("inf"))}
    </div>
    <p><em>Sorted by distance (worst match first). Combined = 25% color + 55% CNN + 20% name embedding.
    Click "▶ 5" to see top-5 alternative matches.</em></p>
    <table>
        <tr>
            <th>ID</th>
            <th colspan="2">Minecraft Block</th>
            <th colspan="2">Current Luanti Node</th>
            <th>Distance</th>
            <th>Color Δ</th>
            <th>CNN Δ</th>
            <th>Name Δ</th>
            <th>Top 5</th>
        </tr>
        {"".join(rows_html)}
    </table>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nReport saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Block texture comparison tool")
    parser.add_argument("--game", choices=["mtg", "mcl", "both"], default="both",
                        help="Which game to analyze")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.game in ("mtg", "both"):
        results = analyze_game(
            "minetest_game",
            "to_minetest_game_node",
            MTG_MODS,
        )
        generate_html_report(results, "minetest_game", REPORT_DIR / "mtg_comparison.html")

    if args.game in ("mcl", "both"):
        results = analyze_game(
            "Mineclonia",
            "to_mineclonia_node",
            MCL_MODS,
        )
        generate_html_report(results, "Mineclonia", REPORT_DIR / "mcl_comparison.html")


if __name__ == "__main__":
    main()
