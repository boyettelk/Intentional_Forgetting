#!/bin/bash

# path to your blender executable
blender=/usr/bin/blender-2.78c-linux-glibc219-x86_64/blender

#----------------------------------------------------------#
NUM_TRAIN_SAMPLES=750
NUM_VAL_SAMPLES=100
NUM_TEST_SAMPLES=150

NUM_PARALLEL_THREADS=10
NUM_THREADS=4
MIN_OBJECTS=1
MAX_OBJECTS=4
MAX_RETRIES=30

#----------------------------------------------------------#
# generate training images
for CLASS_ID in 0 1 2; do
    time $blender \
            --threads $NUM_THREADS \
            --background -noaudio \
            --python render_images_IF.py \
            -- --output_image_dir ../output/train/confounder4_F/images/ \
                --output_scene_dir ../output/train/confounder4_F/scenes/ \
                --output_scene_file ../output/train/confounder4_F/thesis_scenes_train.json \
                --filename_prefix IF_confounder4_F \
                --max_retries $MAX_RETRIES \
                --num_images $NUM_TRAIN_SAMPLES \
                --min_objects $MIN_OBJECTS \
                --max_objects $MAX_OBJECTS \
                --num_parallel_threads $NUM_PARALLEL_THREADS \
                --width 224 --height 224 \
                --properties_json data/properties.json \
                --conf_class_combos_json data/IF-conf.json \
                --gt_class_combos_json data/IF-gt.json \
                --img_class_id $CLASS_ID \
                --IF_type 'False' \
                --validation 'False' \

done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ./output/train/confounder4_F/

#----------------------------------------------------------#

# generate test images
for CLASS_ID in 0 1 2; do
    time $blender \
            --threads $NUM_THREADS \
            --background -noaudio \
            --python render_images_IF.py \
            -- --output_image_dir ../output/test/confounder4_F/images/ \
                --output_scene_dir ../output/test/confounder4_F/scenes/ \
                --output_scene_file ../output/test/confounder4_F/thesis_scenes_test.json \
                --filename_prefix IF_confounder4_F \
                --max_retries $MAX_RETRIES \
                --num_images $NUM_TEST_SAMPLES \
                --min_objects $MIN_OBJECTS \
                --max_objects $MAX_OBJECTS \
                --num_parallel_threads $NUM_PARALLEL_THREADS \
                --width 224 --height 224 \
                --properties_json data/properties.json \
                --conf_class_combos_json data/IF-conf.json \
                --gt_class_combos_json data/IF-gt.json \
                --img_class_id $CLASS_ID \
                --IF_type 'False' \
                --validation 'False' \

done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ./output/test/confounder4_F/

#----------------------------------------------------------#

# generate confounded val images
for CLASS_ID in 0 1 2; do
    time $blender \
            --threads $NUM_THREADS \
            --background -noaudio \
            --python render_images_IF.py \
            -- --output_image_dir ../output/val/confounder4_F/images/ \
                --output_scene_dir ../output/val/confounder4_F/scenes/ \
                --output_scene_file ../output/val/confounder4_F/thesis_scenes_val.json \
                --filename_prefix IF_confounder4_F \
                --max_retries $MAX_RETRIES \
                --num_images $NUM_VAL_SAMPLES \
                --min_objects $MIN_OBJECTS \
                --max_objects $MAX_OBJECTS \
                --num_parallel_threads $NUM_PARALLEL_THREADS \
                --width 224 --height 224 \
                --properties_json data/properties.json \
                --conf_class_combos_json data/IF-conf.json \
                --gt_class_combos_json data/IF-gt.json \
                --img_class_id $CLASS_ID \
                --IF_type 'False' \
                --validation 'False' \

done

# merge all classes join files to one json file
python merge_json_files.py --json_dir ./output/val/confounder4_F/
