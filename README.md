# ComfyUI H3 One-Click Swap

ComfyUI nodes used by Openjutsu for MiniMax H3 character swaps, object edits, tracked-person matching, and face-refinement chunking.

This pack prepares prompts, masks, reference frames and clip plans. Video generation is performed by MiniMax H3 through the surrounding ComfyUI workflow.

## Installation

From your `ComfyUI/custom_nodes` directory:

```sh
git clone https://github.com/D3voz/comfyui-h3-oneclick-swap.git
```

Restart ComfyUI. The nodes appear under `conditioning/minimax/context_loop`.

Use ComfyUI **0.41.2 or newer**, as required by Openjutsu, with native MiniMax H3, SAM 3.1 and `TextGenerate` support. This pack imports ComfyUI's SAM 3 tracker helpers directly. PyTorch, NumPy and Pillow are already supplied by ComfyUI; the pack has no additional Python package requirements.

## Companion packs

For Openjutsu's Simple, Advanced, Object and Montage generation workflows, install:

| Pack | Purpose |
| --- | --- |
| [ComfyUI-MiniMaxH3-Context-Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop) **0.7.3+** | Clip chaining, checkpoints, masked targets and final assembly. |
| [ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder) | `MiniMaxH3FantasticVideoEditLatent` for masked editing. |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Mask growth and mask preview. |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | Video loading and video output. |

Openjutsu's **Fix faces** workflow additionally uses [ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine) and [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack). This pack supplies its `H3FaceChunks` and `H3FaceJoin` nodes; it does not supply face detection or SAM model loading.

## Included nodes

The workflow IDs below are stable identifiers; the names displayed in ComfyUI are more descriptive.

| Workflow ID | Purpose |
| --- | --- |
| `H3OneClickSwapPlan` | Plan overlapping clips and character-swap prompts for up to four reference pictures. |
| `H3AdvancedSwapPlan` | Add role labels and per-subject prompts for Advanced swaps. |
| `H3ObjectSwapPlan` | Plan Change, Remove, Background and Add edits. |
| `H3ObjectPlacementMask` | Build placement masks from normalized box keyframes for adding an object. |
| `H3PaintedEditArea` | Move and scale painted edit keyframes with the tracked part's bounding box. |
| `H3PeopleMask` | Build masks with per-person headroom and temporal holding. |
| `H3GreyMannequins` | Make grey motion references that hide the original person's appearance. |
| `H3ClipShots` | Find hard cuts and prepare shot-description frames. |
| `H3FillShots` | Insert the shot description into a clip prompt. |
| `H3TrackCrops` | Prepare tracked appearances for visual role matching. |
| `H3TrackRoles` | Use the supplied vision-language model to match appearances to subjects. |
| `H3RoleMask` | Build masks for selected labelled subjects. |
| `H3SubjectPrompt` | Select the prompt for one subject's generation pass. |
| `H3MontagePeople` | Save appearance previews and tracks for a Find people review. |
| `H3MontageTracks` | Reload reviewed tracks for the same source frames. |
| `H3MontagePicks` | Choose default replacements and report appearances left unchanged. |
| `H3MontagePlan` | Plan Montage clips and the subjects replaced in each clip. |
| `H3MontagePass` | Select the subject, photo and prompt for the current Montage pass. |
| `H3FaceChunks` | Split face crops into independent overlapping H3 refinement chunks. |
| `H3FaceJoin` | Crossfade refined face chunks before stitching them back. |

## Workflow setup

Use these nodes with Openjutsu's exported API workflows or wire them into a compatible H3 workflow. This repository contains the node implementation, not a complete generation workflow or model weights.

- Supply video frames at **24 fps**. Clip plans use H3's `17k + 5` frame grid.
- Character-swap plans accept reference pictures and optional descriptions. For roles, use a name followed by an optional appearance cue, such as `the drummer - person in a red cap`; the appearance cue helps matching and is kept out of the generated person's identity prompt.
- `short_edge` overrides the resolution preset for custom-size workflows.
- Object edits use SAM tracks for the things to change, remove or preserve. `keep_outline` controls the Object plan's prompt; workflows also need to connect or remove the grey video reference accordingly.
- Find people saves track data in ComfyUI's temporary directory. Run it again after a restart or a change to the source video, selected section or custom size.
- Reusing the same source and plan settings produces the same run name for checkpoint-based resume in the surrounding workflow.

## Adding an object

In Openjutsu, choose **Object swap → Add**, describe the object and its location, and draw a box covering its full silhouette and contact shadow. For example, use `a chocolate cake on a plate` and `on the table`. An optional reference picture controls the new object's appearance. For something that moves or changes, such as a person, say what it does in `action` (`she dances the same routine in sync with the others`); empty means a still object.

One placement keyframe keeps the editable area fixed on screen. For a moving camera, scrub to later frames and draw additional boxes; their positions and sizes are interpolated. This is manual placement animation, not automatic tracking. Draw new positions around cuts as needed. The model generates motion, perspective, lighting and occlusion within that editable area; results still depend on the source clip and prompt.

The node's `placement` string is a JSON array of `{ "frame": 0, "x": 0.4, "y": 0.3, "width": 0.2, "height": 0.25 }` keyframes. Frames are relative to the selected video section at 24 fps; coordinates are fractions of the output canvas. Use the same JSON on the Object Swap Plan so moving the box invalidates resumed clips. Connect the placement mask to the masked-edit path and leave the reference video inputs disconnected: the original clip shows the spot empty, and H3 copies that back. Add mode does not need SAM tracking.

## Models

The Openjutsu default generation workflows reference these files:

| ComfyUI model folder | File |
| --- | --- |
| `models/diffusion_models` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| `models/loras` | `minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors` |
| `models/text_encoders` | `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` |
| `models/text_encoders` | `qwen3vl_8b_int8_convrot.safetensors` |
| `models/vae` | `minimax_h3_video_vae_int8_convrot.safetensors` |
| `models/vae` | `minimax_h3_audio_vae_fp32.safetensors` |
| `models/checkpoints` | `sam3.1_multiplex_fp16.safetensors` |

The workflow's model loaders select the actual files. Fix faces also requires the detector, SAM and identity-matching models documented by FaceRefine. No models are downloaded by this pack.

## Known limitations

- Mirrors and reflections can leave the original person's appearance visible in a motion reference, causing it to return in the result.
- Automatic role matching can make mistakes. Use descriptive roles and review Montage appearances with Find people when needed.
- Very brief or small Montage appearances may be left unchanged; the workflow reports them.

## License

MIT. See [LICENSE](LICENSE). Model weights and companion packs have their own licenses.

### Painted Change edits

`H3PaintedEditArea` takes a full tracked `MASK` timeline without temporal hold, a `paint` JSON string, its `start_frame`, and the output canvas `width` and `height`. It matches the source clip's centre crop at SAM's scale. Each keyframe is `{ "frame": 0, "width": 144, "height": 256, "runs": [start, length, ...] }`: ordered foreground runs in a row-major screen mask, at most 256 pixels per side. Frames are relative to the selected section at 24 fps.

The node anchors each painting to SAM's bounding box at its painted frame, interpolates the relative bounds and shape between keyframes, and unions the result with the tracked mask. An empty track has no paint; a painting with no track at its anchor is skipped. Motion and scale follow the combined tracked bbox, without rotation. Incorrect or oversized SAM detections also move or enlarge the painting.

In Openjutsu, Change with paint tracks the full section once, then `MiniMaxH3ContexLoopMaskSlice` selects each generation clip before Grow Mask. Pass the same paint string to `H3ObjectSwapPlan` to invalidate resumed clips when repainting. Empty paint preserves the existing Change prompt and run hash.
