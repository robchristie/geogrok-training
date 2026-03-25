25-03-2026 2:28 PM

I want to build a SOTA image intelligence system. The initial target is high resolution panchromatic satellite images. Target is non-orthorectified images (image analysis, not mapping). WV2 and WV3 wil make up the bulk of the initial imagery.
I can use the spacenet.ai data as my main datasource datasets/spacenet.ai/ (I haven't downloaded everything, but can grab more data as required). I started playing around with some tooling tools/spacenet/ to explore what's in the spacenet s3 bucket. I'm open to pulling in additional data if required. There's also datasets/srtm/ datasets/planet-osm/ which may be of use in this project.

Additional resources:
* RTX 3090 on this node for lightweight tranining / inference
* Can deploy to runpod.io if more GPU resources are required
* Frontier models (ChatGPT 5.4, Gemini 3 etc.) can be used for caption generation, taxonomy expansion, hard-negative mining, pseudo-label triage, QA and possibly even synthetic data generation
 
The strongest current approach looks to be a multi-part stack: a text-aligned vision-language encoder for embeddings and image/text retrieval, plus a dense geospatial vision backbone for classification, detection, and segmentation. Current geospatial VLM benchmarks still show that no single model dominates all tasks, and even strong generic VLMs lag on geospatial understanding.

For panchromatic specifically, my read of the current literature is that there is not yet a clearly dominant PAN-only, text-aligned foundation model. PAN’s core challenge is that it is high-spatial but single-band, so the real problem is the modality gap relative to RGB/MS-trained VLMs. I'm thinking of starting from a PAN-tolerant dense encoder such as Panopticon or AOM / DOFA, then distill or align it into text space using a stronger RS teacher such as RemoteCLIP or DOFA-CLIP on co-registered PAN↔RGB/MS crops or PAN↔captions. That direction is consistent with work showing that CLIP can be extended to non-RGB RS modalities through cross-modal alignment without training from scratch, and with CLIPPan, which explicitly adapts CLIP to recognize multispectral and panchromatic imagery in a PAN-sensitive setting.

Regarding training, A recent geospatial PEFT study found that PEFT can match or exceed full fine-tuning, improve generalization to unseen regions, and reduce training cost; the same study recommends UNet decoders as a good default for dense tasks. DEFLECT is also interesting for modality adaptation: it gets competitive classification/segmentation results with <1% tuned parameters and 5–10× fewer parameters than low-rank PEFT baselines. https://arxiv.org/abs/2504.17397

For the data I've got, I'm thinking PAN-first multi-model stack where RGB/MSI are used as privileged training-time teachers, and the deployed system runs on PAN only. That matters even more for this case because SpaceNet’s off-nadir benchmark shows current detectors/segmenters still degrade badly on off-nadir and unseen views, and geospatial VLM benchmarks show that general-purpose VLMs are still too weak to be the sole perception backbone for remote-sensing work.

The stack I'm thinking about is:

* **Semantic teacher ensemble for retrieval and text search:** **RemoteCLIP** as the low-risk baseline, plus **DOFA-CLIP** as the high-upside teacher. RemoteCLIP was built specifically for remote sensing and reports strong gains on zero-shot classification and image-text retrieval. DOFA-CLIP is newer and more ambitious: one backbone across EO modalities, trained on **GeoLangBind-2M**, with strong zero-shot results even on unseen modalities and varying spectral inputs. Add **LRSCLIP** for long, relational text queries rather than short labels. ([arXiv][2])

* **PAN-native deployment backbone:** **Panopticon** would be my first choice, with **AnySat** as the second baseline. Panopticon is explicitly an **any-sensor** EO foundation model, uses same-footprint cross-sensor views as natural augmentations, and is designed to handle arbitrary channel combinations. AnySat is also looks strong for one model across many resolutions, scales, and modalities. For later lower-compute transfer, I would watch **RoMA** and **FlexiMo**: RoMA is aimed at scaling efficiently on high-resolution RS imagery, while FlexiMo is designed to adapt to arbitrary spatial resolutions and varying channel counts. ([arXiv][3])

* **Open-vocabulary discovery layer:** use **LAE-DINO** for open-vocabulary detection and **SegEarth-OV** for open-vocabulary segmentation. LAE-DINO is one of the strongest remote-sensing-specific open-vocabulary detectors right now, and SegEarth-OV is a strong training-free segmentation option for buildings, roads, flood, and broader semantic classes. ([arXiv][4])

* **Agentic / analyst-facing layer:** use something like **Falcon** on top of the retrieval and dense stack, not instead of it. Falcon is one of the more interesting remote-sensing VLMs for image-, region-, and pixel-level prompting, while recent agentic work in geospatial AI argues for separating orchestration from task-solving rather than forcing one model to do everything end to end. ([arXiv][5])

That gives a clean separation of roles:

1. **PAN retrieval encoder** for tile embeddings, chip-to-chip search, and text-to-chip search.
2. **PAN dense encoder** for classifiers, detectors, and segmentation.
3. **Open-vocab detector/segmenter** for discovery and pseudo-labeling.
4. **MLLM/agent** for query planning, evidence fusion, and human-facing explanation. ([arXiv][2])

For SpaceNet/WV2/WV3 bootstrap, I'm thinking.

**First, stay in image space.** Maxar’s own product docs distinguish **Basic 1B** as neither geo-referenced nor cartographically projected, and **Standard 2A** as terrain-normalized but still **not orthorectified**. Their product metadata also exposes ancillary information such as collection time and **off-nadir**. Since the deployment target is non-orthorectified image analysis, I think it's best to train and evaluate in that same geometry rather than “cleaning it up” into mapping space. ([csda-maxar-pdfs.s3.amazonaws.com][6])

**Second, treat RGB/MSI as teachers, not runtime dependencies.** Build matched **PAN / RGB / MSI** crops from the same footprint. Run RemoteCLIP and DOFA-CLIP on the RGB/MSI side to create a shared text-aligned embedding target, then train a **PAN student**—preferably on a Panopticon-style backbone—to reproduce those semantics from PAN alone. Panopticon’s use of same-location multi-sensor views as augmentations matches this strategy very well, and DOFA-CLIP is currently one of the strongest language-aligned teachers for heterogeneous EO inputs. ([arXiv][3])

**Third, keep two indices, not one.** A single whole-tile embedding index is fine for scene retrieval, but it is not enough for “find me scenes containing this object” when the object occupies a small fraction of a 40 cm tile. I would keep:

* a **scene/tile index** from the PAN retrieval encoder, and
* a **region/object index** built from detector proposals or segmentation regions.
  That enables support of both “find me more scenes like this chip” and “find me anything containing this object,” which are different retrieval problems.

**Fourth, for dense tasks, start with PEFT rather than full fine-tuning.** A recent geospatial PEFT study found that PEFT can match or exceed full fine-tuning, improve generalization to unseen geographic regions, and reduce memory/time cost; it also recommends **UNet decoders** as a strong default. **TerraTorch** is a good scaffold here because it is built specifically for fine-tuning and benchmarking geospatial foundation models with configurable backbones and decoder heads. ([arXiv][7])

**Fifth, be careful with off-nadir labels.** Off-nadir imagery creates large **roof–footprint offsets**, and the literature explicitly calls out that naive footprint supervision breaks down as view angle, building height, and resolution increase. For building-like classes, I would prefer **roof masks / roof boxes**, or a **joint roof + roof-to-footprint offset head**, instead of blindly rasterizing OSM footprints into image space. ([arXiv][8])

For weak supervision sources:

* **OSM is absolutely worth using**, especially for tags, captions, and weak scene/object supervision. SkyScript was built by linking imagery to OSM semantics at scale and shows that this route is productive for remote-sensing vision-language training. ([arXiv][9])
* **Human produced SpaceNet annotations** should be the high-trust core for geometry-sensitive tasks.
* **Frontier models** are best used for caption generation, taxonomy expansion, hard-negative mining, pseudo-label triage, and QA—not as the final source of precise geometry without review.
* **Generated imagery** is useful as semantic augmentation. Text2Earth is the strongest current example I found for text-driven remote-sensing image generation, but I would use generated images mainly for retrieval robustness or curriculum augmentation, not as the primary source of box/mask truth. ([arXiv][10])

A good option may be:
**RemoteCLIP + DOFA-CLIP teachers → Panopticon PAN student → LAE-DINO + SegEarth-OV → Falcon agent layer.** ([arXiv][2])

[1]: https://openaccess.thecvf.com/content_ICCV_2019/html/Weir_SpaceNet_MVOI_A_Multi-View_Overhead_Imagery_Dataset_ICCV_2019_paper.html "https://openaccess.thecvf.com/content_ICCV_2019/html/Weir_SpaceNet_MVOI_A_Multi-View_Overhead_Imagery_Dataset_ICCV_2019_paper.html"
[2]: https://arxiv.org/abs/2306.11029 "https://arxiv.org/abs/2306.11029"
[3]: https://arxiv.org/html/2503.10845v1 "https://arxiv.org/html/2503.10845v1"
[4]: https://arxiv.org/abs/2408.09110 "https://arxiv.org/abs/2408.09110"
[5]: https://arxiv.org/abs/2503.11070 "https://arxiv.org/abs/2503.11070"
[6]: https://csda-maxar-pdfs.s3.amazonaws.com/DigitalGlobe-Base-Product.pdf "https://csda-maxar-pdfs.s3.amazonaws.com/DigitalGlobe-Base-Product.pdf"
[7]: https://arxiv.org/abs/2504.17397 "https://arxiv.org/abs/2504.17397"
[8]: https://arxiv.org/abs/2204.13637 "https://arxiv.org/abs/2204.13637"
[9]: https://arxiv.org/abs/2312.12856 "https://arxiv.org/abs/2312.12856"
[10]: https://arxiv.org/abs/2501.00895 "https://arxiv.org/abs/2501.00895"
[11]: https://arxiv.org/abs/2503.10392 "https://arxiv.org/abs/2503.10392"


For image reading, I've added third_party/gdal/ third_party/kakadu/
We need to build GDAL with the Kakadu driver to enable fast reading of JP2K encoded NITF files. Use the python binding this build generates to import gdal and open NITF and GeoTIFF image files.

Please help me distil the above into an implementation plan so that I can:
 * get the dataset in place
 * evaluate models
 * assemble models into a framework
 * evaluate framework
 * do additional training to improve performance

Please let me know if I'm missing anything or if anything is unclear.

---


