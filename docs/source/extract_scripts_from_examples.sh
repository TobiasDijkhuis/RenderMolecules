#!/usr/bin/env bash

for example in examples/*/*.blend; do
    directory=$(dirname $example)
    echo $example
    blender --background $example --python-expr 'import bpy; text = bpy.data.texts; keys = text.keys(); print(text[keys[0]].as_string())' | head -n -4 > $directory/script.py
done
