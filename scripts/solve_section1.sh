# depth map generation
# install depth-pro from https://github.com/apple/ml-depth-pro
# and run the following command to generate the depth image
# depth-pro-run -i ./data/example.jpg

python lineart.py \
--input_image test_imgs/girl.png \
--input_depth test_imgs/girl.npz \
--prompt_fg "mermaid underwater, blue clothes, corals and fishes background" \
--prompt_bg "mermaid underwater, blue clothes, corals and fishes background" \
--output_dir "./outputs/section1" \
--seed 1