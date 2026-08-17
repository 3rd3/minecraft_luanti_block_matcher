#!/bin/bash
# ============================================================================
# Block Matcher — Data Setup
# ============================================================================
# Downloads / clones all required data sources for texture comparison:
#   1. Minecraft client.jar + extracted block textures
#   2. minetest_game (Luanti default game)
#   3. Mineclonia (Minecraft-like Luanti game)
#
# Usage:  ./setup.sh
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ---- Minecraft block textures ---------------------------------------------
MC_DIR="minecraft"
MC_JAR_URL="https://piston-data.mojang.com/v1/objects/191771837687b766537a8c4607cb6fad79c533a1/client.jar"

if [ ! -f "$MC_DIR/client.jar" ]; then
    echo "📦 Downloading Minecraft client.jar ..."
    mkdir -p "$MC_DIR"
    curl -L -o "$MC_DIR/client.jar" "$MC_JAR_URL"
else
    echo "✓ Minecraft client.jar already present"
fi

if [ ! -d "$MC_DIR/extracted/assets/minecraft/textures/block" ]; then
    echo "📦 Extracting block textures from client.jar ..."
    cd "$MC_DIR"
    unzip -q -o client.jar 'assets/minecraft/textures/block/*' -d extracted
    unzip -q -o client.jar 'assets/minecraft/textures/entity/*' -d extracted
    cd ..
    echo "  Extracted $(ls "$MC_DIR/extracted/assets/minecraft/textures/block/" | wc -l) block textures"
    echo "  Extracted entity textures (chest, etc.)"
else
    echo "✓ Minecraft block textures already extracted"
fi

# ---- minetest_game --------------------------------------------------------
if [ ! -d "minetest_game" ]; then
    echo "📦 Cloning minetest_game ..."
    git clone --depth 1 https://github.com/luanti-org/minetest_game.git
else
    echo "✓ minetest_game already cloned"
fi

# ---- Mineclonia -----------------------------------------------------------
if [ ! -d "mineclonia" ]; then
    echo "📦 Cloning Mineclonia ..."
    git clone --depth 1 https://codeberg.org/mineclonia/mineclonia.git
else
    echo "✓ Mineclonia already cloned"
fi

echo ""
echo "✅ Setup complete. Data directories:"
echo "   Minecraft textures: $MC_DIR/extracted/assets/minecraft/textures/block/"
echo "   minetest_game mods: minetest_game/mods/"
echo "   Mineclonia mods:    mineclonia/mods/"
echo ""
echo "Next steps:"
echo "  1. Install the Luanti dump mod (see luanti_dump_mod/README)"
echo "  2. Run: python compare_textures.py --game both"
