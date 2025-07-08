for file in *.png; do
    if [[ "$file" == *_white.png ]]; then
        echo "Skipping $file"      
        continue
    fi
    convert "$file" -background white -alpha remove -alpha off "${file%.png}"_white.png
done 


ffmpeg -framerate 16 -pattern_type glob -i '*_white.png' -c:v libx264 CH3OH_orbit_camera.mp4
