# HSM: Hierarchical Scene Motifs for Multi-Scale Indoor Scene Generation

[![Project Page](https://img.shields.io/badge/Project-Website-5B7493?logo=googlechrome&logoColor=5B7493)](https://3dlg-hcvc.github.io/hsm/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2503.16848)

[Hou In Derek Pun](https://houip.github.io/), [Hou In Ivan Tam](https://iv-t.github.io/), [Austin T. Wang](https://atwang16.github.io/), [Xiaoliang Huo](), [Angel X. Chang](https://angelxuanchang.github.io/), [Manolis Savva](https://msavva.github.io/)

3DV 2026

![HSM Overview](docs/static/images/teaser.png)

This repo contains the official implementation of Hierarchical Scene Motifs (HSM), a hierarchical framework for generating realistic indoor environments in a unified manner across scales using scene motifs.

## Environment Setup
The repo is tested on Ubuntu 22.04 LTS with `Python 3.11` and (optional) `CUDA 12.1` for faster object retrieval using CLIP.

To set up the environment, you need to have the following tools installed:
   - `git` and `git-lfs` for downloading HSSD models
   - `conda` or `mamba` for environment setup

### Automated Setup (Recommended)

1. **Acknowledge license to access HSSD on [Hugging Face](https://huggingface.co/datasets/hssd/hssd-models)**

2. **Set up environment variables:**
 
   You need to add the following API keys to the `.env` file:
   - [Your OpenAI or OpenAI-compatible API key](https://platform.openai.com/api-keys)
   - [Your Hugging Face access token](https://huggingface.co/settings/tokens)

   ```bash
   # Copy the template and edit it
   cp .env.example .env

   # Edit the .env file with your API keys
   vim .env  # or use your preferred editor
   ```

   Example `.env` fields:

   ```bash
   OPENAI_API_KEY=your_api_key
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_MODEL_NAME=gpt-4o-2024-08-06
   HF_TOKEN=your_huggingface_token
   ```

3. **Run the automated setup script:**
   ```bash
   ./setup.sh
   ```

This setup script handles all remaining setup steps including:
- Conda environment creation
- HSSD models downloads from Hugging Face
- Preprocessed data downloads from GitHub
- Verify file structure

**Note**: If downloads fail or are interrupted, you can run the setup script again to continue from where it left off.

### Manual Setup
You can follow the instructions below to manually setup the environment.
<details>
<summary>Click to expand</summary>

#### 0. Prepare environment variables
1. Following steps 1 and 2 in the [Automated Setup](#automated-setup-recommended) section.

2. Setup the `conda` environment with the following command:
    ```bash
    mamba env install -f environment.yml
    ```

#### 1. Preprocessed Data

1. **Download Preprocessed Data:**
   1. Visit the [HSM releases page](https://github.com/3dlg-hcvc/hsm/releases)
   2. Download `data.zip` from the latest release
   3. Unzip it at root directory, it should create a `data/` directory at root directory

2. **Download Support Surface Data:**
   1. Visit the [HSM releases page](https://github.com/3dlg-hcvc/hsm/releases)
   2. Download `support-surfaces.zip` from the latest release
   3. Unzip `support-surfaces.zip` and move it under `data/hssd-models/`

#### 2. Assets for Retrieval
We retrieve 3D models from the [Habitat Synthetic Scenes Dataset (HSSD)](https://3dlg-hcvc.github.io/hssd/).     
Accept the terms and conditions for access on [Hugging Face](https://huggingface.co/datasets/hssd/hssd-models).
Get your API token from [Hugging Face settings](https://huggingface.co/settings/tokens).

1.  **Download HSSD Models:**
    
    Activate the environment and login to Hugging Face:
    ```bash
    conda activate hsm
    hf auth login
    ```

    Then, clone the dataset repository (~72GB) under `data`:
    ```bash
    cd data
    git lfs install
    git clone [https://huggingface.co/datasets/hssd/hssd-models](https://huggingface.co/datasets/hssd/hssd-models)
    ```

2.  **Download Decomposed Models:**
    We also use decomposed models from HSSD, download it with the command below:
    ```bash
    hf download hssd/hssd-hab \
        --repo-type=dataset \
        --include "objects/decomposed/**/*_part_*.glb" \
        --exclude "objects/decomposed/**/*_part.*.glb" \
        --local-dir "data/hssd-models"
    ```

#### Directory Structure

Run `./setup.sh --verify` to verify the file structure after setup manually, or

Verify file structure manually, you should have the following file structure at the end:
```
hsm/
|── data/
    ├── hssd-models/
    │   ├── objects/
            ├── decomposed/
    │   ├── support-surfaces/
    │   ├── ...
    |── motif_library/
        ├── meta_programs/
            ├── in_front_of.json
            ├── ...
    ├── preprocessed/
        ├── clip_hssd_embeddings_index.yaml
        ├── hssd_wnsynsetkey_index.json
        ├── clip_hssd_embeddings.npy
        ├── object_categories.json
```
</details>

## Usage

To generate a scene with a description, run the script below:

```bash
conda activate hsm
python main.py [options]
```

**Arguments:**

* `--help`: Show help message and all available arguments
* `-d <desc>`: Description to generate the room.
* `--output <dir>`: Directory to save the generated scenes (default: `results/single_run`)

**Example:**

```bash
python main.py -d "A small living room with a desk and a chair. The desk has a monitor and keyboard on top."
```

To change the parameters, you can edit the `configs/scene/scene_config.yaml` file.

You can also override the model in `configs/scene/scene_config.yaml`:

```yaml
llm:
  model_type: gpt
  model_name: gpt-4.1
  base_url: https://your-openai-compatible-endpoint/v1
```

Note: Command line arguments will override the config file.

## Performance & Cost Estimates

- **Cost**: Approximately **USD $0.80** per scene
- **Time**: Approximately **10 minutes** per scene

These estimates are based on average scene generation runs using the default settings and may vary depending on input description, scene complexity and API response times.

## Output

The default result folder has the following structure:
```
results/
|── <timestamp_roomtype>/
    ├── scene_motifs/           # Scene motifs
    ├── visualizations/         # Visualizations
    ├── room_scene.glb          # GLB file for debugging
    ├── scene.log               # Log for debugging
    ├── stk_scene_state.json    # SceneEval evaluation input
```

**Note**: Do not use the GLB file for evaluation, as the origin of the room geometry is misaligned.


## Evaluation

We use [SceneEval](https://3dlg-hcvc.github.io/SceneEval/) to evaluate the scene generation quality and generate visuals in the paper.

By default, `stk_scene_state.json` will be generated in the output folder and can be used for evaluation.

For more details, please refer to the [official SceneEval repo](https://github.com/3dlg-hcvc/SceneEval).

We also provide the HSM-generated SceneEval-500 scenes and the support region dataset on [HuggingFace](https://huggingface.co/datasets/3dlg-hcvc/hsm).

## Adding New Motif Types

To add new motif types, you need to:
1. Learn a new motif type following the [**Learn Meta-Program from Example**](https://github.com/3dlg-hcvc/smc?tab=readme-ov-file#learn-meta-program-from-example) instructions in the SMC repo.
2. Add the learned meta-program JSON file from SMC to the `data/motif_library/meta_programs/` directory.
3. Update `configs/prompts/motif_types.yaml` and add the new motif type following the format in the `motifs` and `constraints` sections.


## Credits

This project would not be possible without the amazing projects below:
* [SceneMotifCoder](https://github.com/3dlg-hcvc/smc)
* [SceneEval](https://3dlg-hcvc.github.io/SceneEval/)
* [Libsg](https://github.com/smartscenes/libsg)
* [SSTK](https://github.com/smartscenes/sstk)
* [HSSD](https://3dlg-hcvc.github.io/hssd/)

If you use the HSM data or code, please cite:
```
@inproceedings{pun2025hsm,
  title={{HSM: Hierarchical Scene Motifs for Multi-Scale Indoor Scene Generation}},
  author={Pun, Hou In Derek and Tam, Hou In Ivan and Wang, Austin T and Huo, Xiaoliang and Chang, Angel X and Savva, Manolis},
  booktitle = {Proceedings of the IEEE Conference on 3D Vision (3DV)},
  year={2026}
}
```

## Acknowledgements
This work was funded in part by the Sony Research Award Program, a CIFAR AI Chair, a Canada Research Chair, NSERC Discovery Grants, and enabled by support from the [Digital Research Alliance of Canada](https://alliancecan.ca/).
We also thank Jiayi Liu, Weikun Peng, and Qirui Wu for helpful discussions.
