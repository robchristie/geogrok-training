#!/usr/bin/env bash

#  uv run python src/read_metadata.py \
#    --output-path artifacts/s3-spacenet-dataset-images-ds \
#    --include WV \
#    --include P1BS \
#    --ext .NTF \
#    --ext .TIF

  #If you want to keep the search smaller at first, use the prefix as a staging choice, not a correctness claim:

#uv run python src/read_metadata.py \
#    --output-path artifacts/s3-spacenet-dataset-images-ds \
#    --path-prefix Hosted-Datasets/CORE3D-Public-Data/Satellite-Images \
#    --include WV \
#    --include P1BS \
#    --ext .NTF \
#    --ext .TIF
#
#uv run python src/read_metadata.py \
#    --output-path artifacts/s3-spacenet-dataset-images-ds \
#    --path-prefix Hosted-Datasets/CORE3D-Public-Data/Satellite-Images \
#    --include WV \
#    --include PAN \
#    --ext .NTF

#  If you also want the .tif variants:

#uv run python src/read_metadata.py \
#    --output-path artifacts/s3-spacenet-dataset-images-ds \
#    --path-prefix Hosted-Datasets/CORE3D-Public-Data/Satellite-Images \
#    --include WV \
#    --include PAN \
#    --ext .NTF \
#    --ext .TIF

#  Then overlap analysis on deduped scenes:

#uv run python src/find_overlap.py \
#    --metadata-path artifacts/s3-spacenet-dataset-images-ds \
#    --contains WV \
#    --contains P1BS \
#    --ext .NTF \
#    --ext .TIF \
#    --dedupe-scene

  uv run python src/download_overlap_images.py \
    --metadata-path artifacts/s3-spacenet-dataset-images-ds \
    --download-root spacenet.ai \
    --dedupe-scene

