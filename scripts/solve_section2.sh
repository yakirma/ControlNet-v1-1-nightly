# depth map generation
# install depth-pro from https://github.com/apple/ml-depth-pro
# and run the following command to generate the depth image
# depth-pro-run -i ./data/example.jpg

# run diffusion synchronisation process between FG and BG
python lineart.py \
--input_image test_imgs/girl.png \
--input_depth test_imgs/girl.npz \
--prompt_fg "mermaid underwater, blue clothes, corals and fishes background" \
--prompt_bg "underwater, corals and fishes background" \
--do_diffusion_sync \
--output_dir "./outputs/section2" \
--seed 1