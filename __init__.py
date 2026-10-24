import base64
import hashlib
import io
import json
import logging
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import comfy.model_management
import comfy.utils
import folder_paths
from comfy.ldm.sam3.tracker import unpack_masks

FPS = 24
CUT_SNAP = 4
RESOLUTIONS = {"480p": 480, "576p": 576, "736p": 736, "768p": 768, "1080p": 1080}
SHORT_EDGE = ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "Custom short edge in pixels, rounded to 32; replaces resolution when above 0."})


def grid_floor(frames):
    return (frames - 5) // 17 * 17 + 5


def clip_plan(total, clip_frames, overlap, cuts, fits=None):
    """(length, context) per clip. A clip that can end on a hard cut does, and the next one starts without
    overlap; ending up to CUT_SNAP frames before the cut, the next one continues from its last five frames.
    With fits(start, length) on a clip's new frames (after its context), each clip is the longest that fits, where one does."""
    longest = grid_floor(clip_frames)
    clips, end, context = [], 0, 0
    while True:
        start = end - context
        length = grid_floor(min(longest, total - start))
        if fits:
            length = next((n for n in range(length, context + 16, -17) if fits(end, n - context)), length)
        if clips and length - context < 17:
            return clips
        following = overlap
        if total - start > longest:
            near = [(n, c - start - n) for n in range(length, longest // 2, -17) for c in cuts
                    if 0 <= c - start - n <= CUT_SNAP and n - context >= 17 and total - start - n >= 22]
            if near:
                length, off = max(near, key=lambda x: (x[0], -x[1]))
                following = 5 if off else 0
        clips.append((length, context))
        # the next clip's context comes from this clip's new frames only; a clip cut short by fits may have fewer than the overlap
        end, context = start + length, min(following, grid_floor(length - context))


def numbered_lines(text):
    lines = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.):]|picture\s*\d+\s*[:=-])\s*", "", line, flags=re.I).strip().rstrip(".")
        if line:
            lines.append(line)
    return lines


def split_person(line):
    parts = re.split(r"\s+[-–—]\s+", line, maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""


SHOTS_PLACEHOLDER = "{SHOTS}"

SHOT_TABLE = "{SHOT_TABLE}"

SHOTS_PROMPT = """The images are from one video clip. The people to track:
%s
{SHOT_TABLE}
Describe each shot in two or three sentences. Start with the framing, then give every visible subject's place on screen as "on the left", "in the centre" or "on the right" (add foreground or background when it helps), and what they hold when that tells them apart, for example the drumsticks or the microphone. Then say what they do with their body and hands and where they look. Refer to the people only as <Subject 1>, <Subject 2> and so on, with they/them; never mention gender, age, hair, skin, clothing, glasses, hats, jewellery or any other accessory. Leave out anyone not visible in that shot. Output only the shot lines."""


def timestamp(seconds):
    return "%02d:%06.3f" % (seconds // 60, seconds % 60)


def ordinal_people(count):
    if count == 1:
        return ["the main person"]
    names = ["", "second ", "third ", "fourth "]
    return ["the person with the %smost screen time" % names[i] for i in range(count)]


SOUND = """overall_soundscape:
Any ambience and physical sounds already present in <Audio 1> remain part of its unchanged mix. Add no new room noise, footsteps, voices or impacts.

non_diegetic_music:
Reuse the original music and vocals in <Audio 1> at their original tempo, pitch, mix and synchronization. Do not generate a new performance or additional score."""


LIP_SYNC = "The visible people lip-sync any vocals in <Audio 1> where the source people sing or speak; the soundtrack itself is unchanged. "
NO_LIP_SYNC = ("Nobody on screen sings, speaks or lip-syncs; the music in <Audio 1> is a background track added later. "
               "Mouths stay closed or keep the source expressions such as smiles; the soundtrack itself is unchanged. ")


def build_prompt(descriptions, roles, length, overlap, only=None, lip_sync=True):
    """With only=k, the prompt for one pass that replaces Subject k from <Picture 1> and keeps the other subjects as they are."""
    setting = len(descriptions) + 1
    swap = [(i, role, desc, 1 if only else i) for i, (desc, role) in enumerate(zip(descriptions, roles), 1) if only in (None, i)]
    kept = [(i, role) for i, role in enumerate(roles, 1) if only not in (None, i)]
    lines = ["subject_definitions:"]
    for i, role, desc, pic in swap:
        lines.append("<Subject %d> is the person from <Picture %d>. %sThey replace %s in <Video 1>, in every shot where that person appears. "
                     "Their face, hair, body and clothing all come from <Picture %d>." % (i, pic, desc + " " if desc else "", role, pic))
    for i, role in kept:
        lines.append("<Subject %d> is %s in <Video 1>, who is not replaced here and keeps exactly their look in <Video 1>." % (i, role))
    lines += [
        "<Subject %d> is the setting, props and lighting in <Video 1>." % setting,
        "<Video 1> is the source footage being edited, a %.3f-second section at 24 fps, in which the people to replace are shown as smooth grey "
        "figures. It supplies the exact camera framing, camera movement, performance timing, body poses, interactions and hard cuts. "
        "The grey figures supply movement only; faces, hair, skin and clothing come from the pictures." % (length / FPS),
        "<Audio 1> is the synchronized original soundtrack supplied with <Video 1>, reused unchanged as the complete final soundtrack.",
        "",
        "summary:",
    ]
    mapping = "; ".join("<Subject %d> takes the place of %s" % (i, role) for i, role, _, _ in swap)
    summary = ("[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. %s. "
               "Identities and complete outfits come from the pictures. " % mapping)
    if overlap:
        summary += "Its first %.3f seconds are already-edited footage carried over from the preceding section. " % (overlap / FPS)
    lines += [summary + "The source performance, camera, cuts, setting, props and soundtrack retain their timing and layout.", "", "retention_analysis:"]
    for i, role, _, _ in swap:
        lines.append("<Subject %d> (all shots): fully_preserved - the referenced facial identity, hair, skin tone and complete outfit remain consistent "
                     "while poses and mouth movements follow %s." % (i, role))
    for i, role in kept:
        lines.append("<Subject %d> (all shots): fully_copy - unchanged from <Video 1>." % i)
    lines += [
        "<Subject %d> (setting and props): fully_preserved - the location, props and lighting retain their source appearance and position." % setting,
        "<Video 1> (edited source): partially_preserved - the replaced people's identities and outfits change; camera movement, shot composition, "
        "actions, interaction, cuts and timing are retained.",
        "<Audio 1>: fully_copy - the original synchronized soundtrack is the complete final audio, without additional generated vocals or sound effects.",
        "",
        "detailed_description:",
        "Live-action footage in the same setting as <Video 1>, with natural skin tones and the source lighting and lens perspective. "
        "The replacement people appear only where the people they replace are visible; the backgrounds of the pictures do not enter the scene.",
        SHOTS_PLACEHOLDER,
    ]
    for i, role, _, _ in swap:
        lines.append("<Subject %d> performs every action of %s: the same gestures, steps, gaze, expressions and mouth shapes at the same moments." % (i, role))
    lines += [
        "Throughout all shots, %s, regardless of screen position or size. Their outfits replace the original clothing completely; "
        "no face, hair, skin, costume, fabric pattern or accessory of %s remains. %sContinue the original camera movement and performance through the last frame "
        "without adding transitions or changing speed." % (", ".join("%s is always the person in <Picture %d>" % (role, pic) for _, role, _, pic in swap),
                                                           " or ".join(role for _, role, _, _ in swap), LIP_SYNC if lip_sync else NO_LIP_SYNC),
        "",
        SOUND,
    ]
    return "\n".join(lines)


SETTING_SHOTS = " Never describe the room, background or setting; the setting comes from a picture."


def build_picture_prompt(desc, role, length, overlap, lip_sync=True):
    """Whole frame from <Picture 1>: the person and the setting. No source video is shown; a depth control video holds the framing and poses."""
    summary = ("[reference generation + audio reuse] The target video shows <Subject 1> in the setting of <Picture 1>, performing every action of %s "
               "from the source clip with its framing, camera movement and timing. " % role)
    if overlap:
        summary += "Its first %.3f seconds are already-generated footage carried over from the preceding section. " % (overlap / FPS)
    return "\n".join([
        "subject_definitions:",
        "<Subject 1> is the person from <Picture 1>. %sTheir face, hair, body and clothing all come from <Picture 1>." % (desc + " " if desc else ""),
        "<Subject 2> is the setting, background people, props and lighting of <Picture 1>.",
        "<Audio 1> is the synchronized original soundtrack of this %.3f-second section at 24 fps, reused unchanged as the complete final soundtrack." % (length / FPS),
        "",
        "summary:",
        summary + "Identity, outfit and setting come from <Picture 1>.",
        "",
        "retention_analysis:",
        "<Subject 1> (all shots): fully_preserved - the referenced facial identity, hair, skin tone and complete outfit remain consistent while poses and "
        "mouth movements follow %s." % role,
        "<Subject 2> (setting): fully_preserved - the location, lighting and atmosphere of <Picture 1> surround <Subject 1> in every shot.",
        "<Audio 1>: fully_copy - the original synchronized soundtrack is the complete final audio, without additional generated vocals or sound effects.",
        "",
        "detailed_description:",
        "Live-action footage in the setting of <Picture 1>, with natural skin tones and the lighting of <Picture 1>.",
        SHOTS_PLACEHOLDER,
        "<Subject 1> performs every action of %s: the same gestures, steps, gaze, expressions and mouth shapes at the same moments. %sContinue the "
        "camera movement and performance through the last frame without adding transitions or changing speed." % (role, LIP_SYNC if lip_sync else NO_LIP_SYNC),
        "",
        SOUND,
    ])


class H3OneClickSwapPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frames": ("IMAGE", {"tooltip": "Source video frames at 24 fps."}),
                "picture_1": ("IMAGE",),
                "clip_seconds": ("FLOAT", {"default": 6.0, "min": 2.5, "max": 15.0, "step": 0.1, "tooltip": "Longest clip per H3 generation, rounded down to the 17k+5 grid."}),
                "overlap_frames": ([22, 39, 56], {"default": 39, "tooltip": "Frames each clip repeats from the previous clip."}),
                "resolution": (list(RESOLUTIONS), {"default": "736p", "tooltip": "Short edge of the generated video; the shape follows the source video. 480p is a quick preview; 1080p is about 2.5x slower than 736p and needs much more memory."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "roles": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional. One line per picture naming who it replaces, e.g. 'the drummer - man in a red cap'. Text after ' - ' only helps the shot describer find the person; it never reaches H3. Blank uses the auto-detected people."}),
            },
            "optional": {
                "picture_2": ("IMAGE",),
                "picture_3": ("IMAGE",),
                "picture_4": ("IMAGE",),
                "picture_1_text": ("STRING", {"forceInput": True}),
                "picture_2_text": ("STRING", {"forceInput": True}),
                "picture_3_text": ("STRING", {"forceInput": True}),
                "picture_4_text": ("STRING", {"forceInput": True}),
                "video_people": ("STRING", {"forceInput": True, "tooltip": "Numbered list of the people in the video, most screen time first."}),
                "no_lip_sync": ("BOOLEAN", {"default": False, "tooltip": "Turn on when the people in the video do not sing or talk along to the soundtrack, so H3 does not make them lip-sync it. Also turn on OPTIONAL - No lip-sync, which mutes the audio H3 hears and shows it the faces."}),
                "short_edge": SHORT_EDGE,
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("plan_json", "width", "height", "run_name", "shot_prompt")
    OUTPUT_TOOLTIPS = ("", "", "", "", "Instructions for a vision model that writes each clip's shot list; its answer replaces {SHOTS} in the clip prompt.")
    FUNCTION = "build"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = "Splits a source video into overlapping H3 clips and writes the character-swap prompt for each clip."

    def build(self, source_frames, picture_1, clip_seconds, overlap_frames, resolution, steps, seed, roles,
              picture_2=None, picture_3=None, picture_4=None, picture_1_text="", picture_2_text="",
              picture_3_text="", picture_4_text="", video_people="", no_lip_sync=False, short_edge=0):
        return self.plan(source_frames, [picture_1, picture_2, picture_3, picture_4], [picture_1_text, picture_2_text, picture_3_text, picture_4_text],
                         clip_seconds, overlap_frames, short_edge or RESOLUTIONS[resolution], steps, seed, roles, video_people, no_lip_sync)[:5]

    def plan(self, source_frames, pictures, texts, clip_seconds, overlap_frames, short_edge, steps, seed, roles, video_people, no_lip_sync,
             setting_from_picture=False):
        pictures = [p for p in pictures if p is not None]
        descriptions = []
        for i in range(len(pictures)):
            desc = " ".join((texts[i] or "").split())
            descriptions.append(desc if desc.endswith(".") or not desc else desc + ".")

        count = len(pictures)
        people = numbered_lines(roles) or numbered_lines(video_people)
        people += ordinal_people(count)[len(people):]
        people = [split_person(p) for p in people[:count]]
        people_lines = "\n".join("<Subject %d>: %s" % (i, role + (" - " + looks if looks else "")) for i, (role, looks) in enumerate(people, 1))
        shot_prompt = SHOTS_PROMPT % people_lines + SETTING_SHOTS * setting_from_picture
        # Gendered roles ("the woman") pull H3 toward the source person's gender when the grey figure is all it sees.
        people = [neutral_words(role) for role, _ in people]

        clips, out_w, out_h = plan_clips(source_frames, clip_seconds, overlap_frames, short_edge)
        if setting_from_picture:
            prompts = [build_picture_prompt(descriptions[0], people[0], length, overlap, not no_lip_sync) for length, overlap in clips]
        else:
            prompts = [build_prompt(descriptions, people, length, overlap, lip_sync=not no_lip_sync) for length, overlap in clips]
        subject_prompts = [[build_prompt(descriptions, people, length, overlap, k, not no_lip_sync) for k in range(1, count + 1)] for length, overlap in clips]
        # No lip-sync only joins the hash when on, so existing runs keep their names and resume.
        run_name = ("picture_" if setting_from_picture else "swap_") + run_hash(
            source_frames, pictures, [people, descriptions, clips, out_w, out_h, steps, seed] + ["no lip-sync"] * no_lip_sync)
        return (plan_json("swap", prompts, clips, overlap_frames, steps, seed), out_w, out_h, run_name, shot_prompt, people_lines,
                json.dumps(subject_prompts))


def plan_clips(source_frames, clip_seconds, overlap_frames, short_edge, fits=None):
    """(length, context) per clip, split at the source's hard cuts, and the canvas size."""
    total, height, width = source_frames.shape[:3]
    scale = short_edge / min(width, height)
    out_w = max(32, round(width * scale / 32) * 32)
    out_h = max(32, round(height * scale / 32) * 32)
    cuts = [a for a, _ in find_shots(source_frames[:, ::4, ::4], 40.0)[1:]]
    return clip_plan(total, int(clip_seconds * FPS), overlap_frames, cuts, fits), out_w, out_h


def plan_json(kind, prompts, clips, overlap_frames, steps, seed):
    shots = []
    for i, (prompt, (length, overlap)) in enumerate(zip(prompts, clips)):
        shot = {"id": "%02d_%s" % (i + 1, kind), "prompt": prompt, "length": length, "steps": steps, "seed": str(seed + i)}
        if overlap:
            # generated-audio carry needs the full overlap; the final video uses the source audio anyway
            shot.update({"context_length": overlap, "continuation_mode": "masked_av", "audio_context_length": 22 if overlap == overlap_frames else 0})
        elif i:
            shot.update({"context_length": 0, "audio_context_length": 0})
        shots.append(shot)
    return json.dumps({"defaults": {"steps": steps}, "shots": shots}, indent=2)


def run_hash(source_frames, pictures, settings):
    """Same inputs, same hash, so a re-run resumes from the saved clips."""
    h = hashlib.sha1()
    for t in [source_frames[::FPS, ::32, ::32]] + [p[:, ::32, ::32] for p in pictures]:
        h.update(t.float().mean(dim=-1).cpu().numpy().tobytes())
    h.update(json.dumps(settings).encode())
    return h.hexdigest()[:10]


class H3AdvancedSwapPlan(H3OneClickSwapPlan):
    RETURN_TYPES = H3OneClickSwapPlan.RETURN_TYPES + ("STRING", "STRING")
    RETURN_NAMES = H3OneClickSwapPlan.RETURN_NAMES + ("people", "subject_prompts")
    OUTPUT_TOOLTIPS = H3OneClickSwapPlan.OUTPUT_TOOLTIPS + ("One '<Subject N>: role' line per picture, for H3 Track Crops.",
                                                            "Per clip, one prompt per subject for one-pass-per-person runs; see H3 Subject Prompt.")
    DESCRIPTION = "Like H3 One-Click Swap Plan, for role-labelled tracks: also lists the subjects for the track labeller and one prompt per subject."

    def build(self, source_frames, picture_1, clip_seconds, overlap_frames, resolution, steps, seed, roles,
              picture_2=None, picture_3=None, picture_4=None, picture_1_text="", picture_2_text="",
              picture_3_text="", picture_4_text="", video_people="", no_lip_sync=False, short_edge=0):
        out = self.plan(source_frames, [picture_1, picture_2, picture_3, picture_4], [picture_1_text, picture_2_text, picture_3_text, picture_4_text],
                        clip_seconds, overlap_frames, short_edge or RESOLUTIONS[resolution], steps, seed, roles, video_people, no_lip_sync)
        people = out[5]
        if numbered_lines(roles) and video_people.strip():
            # Bare roles ("the drummer") cannot be told apart in close-ups; the auto-detected looks can.
            people += ("\nThe people seen in the video, by look (their names may differ from the roles above):\n"
                       + "\n".join("- " + p for p in numbered_lines(video_people)))
        return out[:5] + (people,) + out[6:]


class H3PictureSwapPlan(H3OneClickSwapPlan):
    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        for name in ["picture_2", "picture_3", "picture_4", "picture_2_text", "picture_3_text", "picture_4_text"]:
            del types["optional"][name]
        return types

    DESCRIPTION = "Like H3 One-Click Swap Plan, but the whole frame comes from the picture: its person and its setting. For a depth ControlNet run without a source video reference."

    def build(self, source_frames, picture_1, clip_seconds, overlap_frames, resolution, steps, seed, roles,
              picture_1_text="", video_people="", no_lip_sync=False, short_edge=0):
        return self.plan(source_frames, [picture_1], [picture_1_text], clip_seconds, overlap_frames, short_edge or RESOLUTIONS[resolution],
                         steps, seed, roles, video_people, no_lip_sync, setting_from_picture=True)[:5]


MONTAGE_PASSES = 2


class H3MontagePlan:
    @classmethod
    def INPUT_TYPES(cls):
        types = H3OneClickSwapPlan.INPUT_TYPES()
        types["required"]["track_data"] = ("SAM3_TRACK_DATA", {"tooltip": "The tracks Find people labelled, from H3 Montage Tracks."})
        types["required"]["labels"] = ("STRING", {"multiline": True, "default": "[]", "tooltip": "JSON list of the appearances to replace: "
                                                  "[{\"track\": SAM track id, \"start\": frame, \"end\": frame, \"role\": picture number}]."})
        types["required"]["roles"] = ("STRING", {"multiline": True, "default": "", "tooltip": "One 'who - looks' line per picture; only the part before ' - ' reaches H3."})
        del types["optional"]["video_people"]
        return types

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("plan_json", "width", "height", "run_name", "montage")
    OUTPUT_TOOLTIPS = ("", "", "", "", "Per clip, each pass's photo, labelled tracks and prompt; for H3 Montage Pass.")
    FUNCTION = "build"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = ("Splits the video into overlapping H3 clips like the swap plans. Each clip gets one pass per picked photo on screen in it "
                   "(the two with the most screen time), masking only that photo's labelled tracks, with a prompt for that person only.")

    def build(self, source_frames, picture_1, clip_seconds, overlap_frames, resolution, steps, seed, roles, track_data, labels,
              picture_2=None, picture_3=None, picture_4=None, picture_1_text="", picture_2_text="",
              picture_3_text="", picture_4_text="", no_lip_sync=False, short_edge=0):
        if track_data["n_frames"] != source_frames.shape[0]:
            raise ValueError("Find people tracked %d frames, this video has %d. Find people again with the same video, part and size."
                             % (track_data["n_frames"], source_frames.shape[0]))
        pictures = [p for p in [picture_1, picture_2, picture_3, picture_4] if p is not None]
        descriptions = [" ".join((t or "").split()) for t in [picture_1_text, picture_2_text, picture_3_text, picture_4_text][:len(pictures)]]
        descriptions = [d if d.endswith(".") or not d else d + "." for d in descriptions]
        people = [neutral_words(split_person(line)[0]) for line in numbered_lines(roles)]
        people += ordinal_people(len(pictures))[len(people):]
        labels = [l for l in json.loads(labels) if l["role"]]

        # the frames each appearance is really on screen; a label also spans the gaps around it
        packed = track_data["packed_masks"]
        shown = []
        who = [set() for _ in range(source_frames.shape[0])]
        for l in labels:
            m = unpack_masks(packed[l["start"]:l["end"], l["track"]])
            seen = (m.sum((1, 2)) > MIN_AREA * m.shape[1] * m.shape[2]).tolist()
            shown.append({f for f, on in enumerate(seen, l["start"]) if on})
            for f in shown[-1]:
                who[f].add(l["role"])

        # A clip carries its last frames into the next one as finished footage, so a clip has to replace everyone in its new frames
        # (its context frames are already finished): clips are cut shorter where more photos' people than passes would be on screen.
        fits = lambda start, length: len(set().union(*who[start:start + length])) <= MONTAGE_PASSES
        clips, out_w, out_h = plan_clips(source_frames, clip_seconds, overlap_frames, short_edge or RESOLUTIONS[resolution], fits)
        plan, end = [], 0
        for i, (length, context) in enumerate(clips, 1):
            start = end - context
            end = start + length
            frames = {}
            for l, on in zip(labels, shown):
                frames.setdefault(l["role"], set()).update(f for f in on if start <= f < end)
            fresh = {k: sum(f >= start + context for f in on) for k, on in frames.items()}
            photos = sorted((k for k in fresh if fresh[k]), key=lambda k: -fresh[k])
            if len(photos) > MONTAGE_PASSES:
                logging.warning("H3 Montage Plan: clip %d shows %d of the photos' people within a few frames; only the %d with the most screen time are replaced.",
                                i, len(photos), MONTAGE_PASSES)
            passes = []
            for k in photos[:MONTAGE_PASSES] or [1]:
                prompt = build_prompt(descriptions[k - 1:k], people[k - 1:k], length, context, lip_sync=not no_lip_sync)
                on = sorted(frames.get(k, ()))
                when = ""
                if on and (on[0] > start or on[-1] < end - 1):
                    when = ("<Subject 1> is on screen from %s to %s of <Video 1>; before and after that, <Video 1> stays as it is.\n"
                            % (timestamp((on[0] - start) / FPS), timestamp((on[-1] + 1 - start) / FPS)))
                passes.append({"photo": k, "prompt": prompt.replace(SHOTS_PLACEHOLDER + "\n", when),
                               "labels": [{"track": l["track"], "start": max(l["start"], start) - start, "end": min(l["end"], end) - start, "role": 1}
                                          for l in labels if l["role"] == k and l["start"] < end and start < l["end"]]})
            plan.append({"start": start, "length": length, "passes": passes})

        run_name = "montage_" + run_hash(source_frames, pictures, [people, descriptions, labels, clips, out_w, out_h, steps, seed]
                                         + ["no lip-sync"] * no_lip_sync)
        return (plan_json("montage", [c["passes"][0]["prompt"] for c in plan], clips, overlap_frames, steps, seed), out_w, out_h, run_name,
                json.dumps({"clips": plan}))


class H3MontagePass:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "montage": ("STRING", {"forceInput": True, "tooltip": "From H3 Montage Plan."}),
                "track_data": ("SAM3_TRACK_DATA", {"tooltip": "The tracks Find people labelled, from H3 Montage Tracks."}),
                "clip_index": ("INT", {"forceInput": True, "tooltip": "From Current Clip."}),
                "pass_number": ("INT", {"default": 1, "min": 1, "max": 4}),
                "picture_1": ("IMAGE",),
            },
            "optional": {
                "picture_2": ("IMAGE",),
                "picture_3": ("IMAGE",),
                "picture_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("SAM3_TRACK_DATA", "STRING", "STRING", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("track_data", "labels", "prompt", "picture", "next_pass")
    OUTPUT_TOOLTIPS = ("This clip's frames of the tracks.", "This pass's person as Subject 1, for H3 Role Mask.", "", "This pass's photo, as <Picture 1>.",
                       "Whether this clip has another pass.")
    FUNCTION = "pick"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = "This clip's pass of a montage: the person it replaces, their photo and the prompt that describes only them."

    def pick(self, montage, track_data, clip_index, pass_number, picture_1, picture_2=None, picture_3=None, picture_4=None):
        clip = json.loads(montage)["clips"][clip_index - 1]
        p = clip["passes"][pass_number - 1]
        a, length = clip["start"], clip["length"]
        packed = track_data["packed_masks"]
        clip_tracks = {**track_data, "packed_masks": None if packed is None else packed[a:a + length], "n_frames": length}
        picture = [picture_1, picture_2, picture_3, picture_4][p["photo"] - 1]
        return (clip_tracks, json.dumps(p["labels"]), p["prompt"], picture, len(clip["passes"]) > pass_number)


EDIT_SHOTS_PROMPT = """The images are from one video clip.
{SHOT_TABLE}
Describe each shot in two or three sentences. Start with the framing and any camera movement, then say where the visible people are on screen ("on the left", "in the centre" or "on the right"), what they do with their body and hands and where they look. Refer to them only by their place on screen, with they/them; never mention gender, age, hair, skin, clothing or accessories. %s Output only the shot lines."""


def build_edit_prompt(mode, thing, plural, change_to, picture, desc, length, overlap, outline):
    """H3 video-editing prompt. With a reference video, thing becomes <Subject 1>: change shows thing in <Video 1> as a grey
    shape (outline), background shows everything around it as a grey blur. H3 copies a grey shape's outline, so change without
    outline and remove have none: thing becomes change_to, or for remove is never named."""
    is_, its, s = ("are", "their", "") if plural else ("is", "its", "s")
    seconds = length / FPS
    ref = mode == "background" or mode == "change" and outline
    if ref:
        new = "<Subject 1>" + (", " + change_to if change_to else "")
        what = "new look of " + thing if mode == "change" else "new setting"
        source = ("in which %s %s shown as a smooth grey shape that supplies only %s position, size and movement" % (thing, is_, its) if mode == "change" else
                  "in which the setting around %s is shown as a smooth grey blur that supplies only the camera movement and rough layout" % thing)
        lines = ["subject_definitions:"]
        lines.append(("<Subject 1> is the %s from <Picture 1>. %s" % (what, desc)).strip() if picture else
                     "<Subject 1> is the %s: %s." % (what, change_to or "a new look"))
        lines += [
            "<Video 1> is the source footage being edited, a %.3f-second section at 24 fps, %s. It supplies the exact camera framing, "
            "camera movement, people, performance timing and hard cuts." % (seconds, source),
            "<Audio 1> is the synchronized original soundtrack supplied with <Video 1>, reused unchanged as the complete final soundtrack.",
            "",
            "summary:",
            "[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. "
            + ("%s become%s %s. " % (thing[0].upper() + thing[1:], s, new) if mode == "change" else
               "%s keeps their exact look and performance, while the setting around them becomes %s. " % (thing[0].upper() + thing[1:], new)),
        ]
    else:
        new = ("<Subject 1>" + (", " + change_to if change_to else "") if picture else
               change_to or "a new look, unlike the original")
        lines = ["subject_definitions:"]
        if picture:
            lines.append(("<Subject 1> is the new look of %s from <Picture 1>. %s" % (thing, desc)).strip())
        lines.append("The source footage is a %.3f-second section at 24 fps. It supplies the exact camera framing, camera movement, people, "
                     "performance timing, setting and hard cuts, and its original soundtrack is reused unchanged." % seconds)
        lines += ["", "summary:"]
        if mode == "remove":
            lines.append("[video editing + audio reuse] The target video is an edited version of the source footage in which the regenerated "
                         "area shows only %s, as if nothing had ever been there. " % (change_to or "the background and surfaces that continue naturally around it"))
        else:
            lines.append("[video editing + %saudio reuse] The target video is an edited version of the source footage in which %s %s "
                         "replaced by %s, with %s own shape, cut, colours and materials. " %
                         ("reference generation + " if picture else "", thing, is_, new, its))
    if overlap:
        lines[-1] += "Its first %.3f seconds are already-edited footage carried over from the preceding section. " % (overlap / FPS)
    lines[-1] += "Everything else keeps its source look, timing and layout."
    lines += ["", "retention_analysis:"]
    if mode == "remove":
        lines.append("The source footage: partially_preserved - only the regenerated area changes; the people, actions, camera movement, "
                     "shot composition, cuts and timing are retained.")
        lines.append("The soundtrack: fully_copy - the original soundtrack is the complete final audio.")
    elif ref and mode == "change":
        lines += [
            "<Subject 1> (all shots): fully_preserved - its shape, colours, materials and details stay consistent while it follows the "
            "movement of %s in <Video 1>." % thing,
            "<Video 1> (edited source): partially_preserved - only %s change%s; the people, actions, camera movement, shot composition, "
            "setting, cuts and timing are retained." % (thing, s),
            "<Audio 1>: fully_copy - the original synchronized soundtrack is the complete final audio, without additional "
            "generated vocals or sound effects.",
        ]
    elif mode == "change":
        if picture:
            lines.append("<Subject 1> (all shots): fully_preserved - its shape, colours, materials and details stay consistent while it "
                         "follows the movement of the people.")
        lines.append("The source footage: partially_preserved - only %s change%s; the people, actions, camera movement, shot composition, "
                     "setting, cuts and timing are retained." % (thing, s))
        lines.append("The soundtrack: fully_copy - the original soundtrack is the complete final audio.")
    else:
        lines += [
            "<Subject 1> (setting): partially_preserved - the look, materials and lighting of the new setting come from it; the camera "
            "framing and movement are those of <Video 1>.",
            "<Video 1> (edited source): partially_preserved - only the setting changes; %s, the actions, camera movement, cuts and "
            "timing are retained." % thing,
            "<Audio 1>: fully_copy - the original synchronized soundtrack is the complete final audio, without additional "
            "generated vocals or sound effects.",
        ]
    lines += ["", "detailed_description:", "Live-action footage with the source lighting and lens perspective.", SHOTS_PLACEHOLDER]
    if mode == "change":
        lines.append("Throughout all shots, %s %s always %s, regardless of screen position or size; no colour, pattern or detail of the "
                     "original remains." % (thing, is_, new))
    elif mode == "background":
        lines.append("Throughout all shots, the surroundings are %s, lit to match %s; no part of the original location remains." % (new, thing))
    else:
        lines.append("Throughout all shots, the regenerated area shows %s, matching the surrounding light, texture and perspective." %
                     (change_to or "the background and surfaces that continue naturally around it"))
    lines += ["Continue the original camera movement and performance through the last frame without adding transitions or changing speed.",
              "", SOUND if ref else SOUND.replace("<Audio 1>", "the source soundtrack")]
    return "\n".join(lines)


ADD_VERB = re.compile(r"^(?:add|insert|put)\s+", re.I)


def placement_words(placement):
    """Where the first placement box sits, in words: H3's text encoder never sees the mask."""
    p = sorted(json.loads(placement), key=lambda k: k["frame"])[0]
    cx, cy = p["x"] + p["width"] / 2, p["y"] + p["height"] / 2
    col = "left" if cx < 0.36 else "right" if cx > 0.64 else "centre"
    row = "upper" if cy < 0.36 else "lower" if cy > 0.64 else "middle"
    if row == "middle":
        return "in the centre of the frame" if col == "centre" else "on the %s of the frame" % col
    return "in the %s %s of the frame" % (row, col)


def build_add_prompt(target, change_to, picture, desc, length, overlap, where):
    """Describes the finished clip with the object in it, like the change prompt. A shot list of the bare surface and
    "keep / do not replace" lines made H3 return the clip unchanged."""
    new = "<Subject 1>" + (" (%s)" % change_to if change_to else "") if picture else change_to
    lines = ["subject_definitions:"]
    if picture:
        lines.append(("<Subject 1> is the new object from <Picture 1>. %s" % desc).strip())
    lines += [
        "<Video 1> is the source video for the target video edit, a %.3f-second section at 24 fps. It supplies the exact camera "
        "framing, camera movement, people, performance timing and hard cuts." % (length / FPS),
        "<Audio 1> is the synchronized original soundtrack supplied with <Video 1>, reused unchanged as the complete final soundtrack.",
        "", "summary:",
        "[video editing + %saudio reuse] The target video is an edited version of <Video 1> in which %s is %s, %s, a real "
        "object with contact shadows that match the source light and perspective. " % ("reference generation + " if picture else "", new, target, where),
    ]
    if overlap:
        lines[-1] += "Its first %.3f seconds are already-edited footage carried over from the preceding section, with it already in place. " % (overlap / FPS)
    lines[-1] += "Everything else keeps its source look, timing and layout."
    lines += ["", "retention_analysis:"]
    if picture:
        lines.append("<Subject 1> (all shots): fully_preserved - its shape, colours, materials and details stay consistent in every frame.")
    lines += [
        "<Video 1> (edited source): partially_preserved - the only change is %s %s; the people, actions, camera movement, shot "
        "composition, cuts and timing are retained." % (new, target),
        "<Audio 1>: fully_copy - the original synchronized soundtrack is the complete final audio, without additional "
        "generated vocals or sound effects.",
        "", "detailed_description:", "Live-action footage with the source lighting and lens perspective.", SHOTS_PLACEHOLDER,
        "Throughout all shots, %s is clearly visible %s, %s, with the same size, identity and materials in every frame; people "
        "who move in front of it pass in front of it." % (new, target, where),
        "Continue the original camera movement and performance through the last frame without adding transitions or changing speed.",
        "", SOUND,
    ]
    return "\n".join(lines)

class H3ObjectSwapPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frames": ("IMAGE", {"tooltip": "Source video frames at 24 fps."}),
                "mode": (["change", "remove", "background", "add"], {"default": "change", "tooltip": "change: replace the tracked thing. remove: erase it. background: change its surroundings. add: insert a new object inside a placement mask."}),
                "target": ("STRING", {"default": "hair", "tooltip": "What SAM tracks, as the prompt names it: hair, jacket, car, or several separated by commas (hair, dress). For background, what stays (person)."}),
                "change_to": ("STRING", {"multiline": True, "default": "", "tooltip": "What it becomes (short bright pink hair), or for remove what shows instead (optional)."}),
                "clip_seconds": ("FLOAT", {"default": 6.0, "min": 2.5, "max": 15.0, "step": 0.1, "tooltip": "Longest clip per H3 generation, rounded down to the 17k+5 grid."}),
                "overlap_frames": ([22, 39, 56], {"default": 39, "tooltip": "Frames each clip repeats from the previous clip."}),
                "resolution": (list(RESOLUTIONS), {"default": "736p"}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "keep_outline": ("BOOLEAN", {"default": True, "tooltip": "change only. On: the grey shape goes into Reference Conditioning, so the real face, hands and arm motion around it are kept, but the new look keeps the old outline. Off: delete the two ref video inputs; the outline can change (long hair to a bob, pants to a skirt), but edges near the face and hands may be lost."}),
            },
            "optional": {
                "picture_1": ("IMAGE", {"tooltip": "What it should look like: an outfit, a hairstyle, a product or a place. Not used by remove."}),
                "picture_1_text": ("STRING", {"forceInput": True}),
                "short_edge": SHORT_EDGE,
                "placement": ("STRING", {"default": "", "tooltip": "Add mode: the placement keyframes used by H3 Object Placement Mask. Included in the run name so moving the object starts a new run."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("plan_json", "width", "height", "run_name", "shot_prompt")
    OUTPUT_TOOLTIPS = H3OneClickSwapPlan.OUTPUT_TOOLTIPS
    FUNCTION = "build"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = "Splits a source video into overlapping H3 clips and writes the prompt for changing, removing or re-setting the tracked things."

    def build(self, source_frames, mode, target, change_to, clip_seconds, overlap_frames, resolution, steps, seed, keep_outline=True, picture_1=None, picture_1_text="", short_edge=0, placement=""):
        parts = [re.sub(r"^(?:the|a|an)\s+", "", " ".join(p.split()), flags=re.I) for p in target.split(",")]
        parts = [p for p in parts if p] or ["person"]
        thing = "the " + (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1])
        change_to = " ".join(change_to.split()).rstrip(".")
        if mode == "add":
            change_to = ADD_VERB.sub("", change_to)
        picture = picture_1 is not None and mode != "remove"
        desc = " ".join(picture_1_text.split()) if picture else ""
        if desc and not desc.endswith("."):
            desc += "."
        never = {"change": "Never mention %s." % thing, "remove": "Never mention %s." % thing,
                 "background": "Say nothing about the location, background or setting.",
                 "add": "Say nothing about what is or is not %s." % target}[mode]
        clips, out_w, out_h = plan_clips(source_frames, clip_seconds, overlap_frames, short_edge or RESOLUTIONS[resolution])
        prompts = [build_add_prompt(target, change_to, picture, desc, length, overlap, placement_words(placement)) if mode == "add" else
                   build_edit_prompt(mode, thing, len(parts) > 1, change_to, picture, desc, length, overlap, keep_outline) for length, overlap in clips]
        pictures = [picture_1] if picture else []
        run_name = "edit_" + run_hash(source_frames, pictures, [mode, thing, change_to, desc, clips, out_w, out_h, steps, seed, keep_outline] + ([placement] if mode == "add" else []))
        return (plan_json("edit", prompts, clips, overlap_frames, steps, seed), out_w, out_h, run_name, EDIT_SHOTS_PROMPT % never)


class H3ObjectPlacementMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "placement": ("STRING", {"multiline": True}),
                             "start_frame": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff})}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "build"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = "An editable box per video frame, interpolated between normalized placement keyframes. One keyframe keeps the box fixed on screen."

    def build(self, images, placement, start_frame=0):
        points = sorted(json.loads(placement), key=lambda p: p["frame"])
        if not points:
            raise ValueError("Draw an object placement area first.")
        for p in points:
            if not isinstance(p["frame"], int) or any(not isinstance(p[k], (int, float)) or not math.isfinite(p[k]) for k in ("x", "y", "width", "height")):
                raise ValueError("Placement keyframes need a frame number and finite normalized box coordinates.")
            if not (0 <= p["x"] < 1 and 0 <= p["y"] < 1 and 0 < p["width"] <= 1 - p["x"] + 1e-6 and 0 < p["height"] <= 1 - p["y"] + 1e-6):
                raise ValueError("The placement area must be inside the video frame.")
        n, h, w = images.shape[:3]
        scale = min(1.0, 256 / max(h, w))
        mh, mw = max(1, round(h * scale)), max(1, round(w * scale))
        mask = torch.zeros(n, mh, mw, device=comfy.model_management.intermediate_device())
        j = 0
        for i in range(n):
            frame = start_frame + i
            while j + 1 < len(points) and points[j + 1]["frame"] <= frame:
                j += 1
            a, b = points[j], points[min(j + 1, len(points) - 1)]
            t = max(0.0, min(1.0, (frame - a["frame"]) / (b["frame"] - a["frame"]))) if b["frame"] != a["frame"] else 0.0
            x, y, width, height = [a[k] + (b[k] - a[k]) * t for k in ("x", "y", "width", "height")]
            left, top = int(x * mw), int(y * mh)
            right, bottom = max(left + 1, math.ceil((x + width) * mw)), max(top + 1, math.ceil((y + height) * mh))
            mask[i, top:bottom, left:right] = 1.0
        return (mask,)


FACE_OVERLAP = 24
FACE_RAMP = 0.04


def build_face_prompt(desc, follow_mouth, length):
    lines = ["subject_definitions:", ("<Subject 1> is the person from <Picture 1>. %s Their face, skin tone, eyebrows and facial hair come from <Picture 1>." % desc).replace("  ", " ")]
    if follow_mouth:
        lines.append("<Video 1> is a soft, blurred copy of the same close-up footage. It supplies only the exact head movement, gaze, expressions, "
                     "mouth shapes and framing, not the facial details.")
    lines += [
        "<Audio 1> is the synchronized original soundtrack, reused unchanged as the complete final audio.",
        "",
        "summary:",
        "[video editing + reference generation + audio reuse] The target video is an edited %.3f-second close-up of <Subject 1>'s head and "
        "shoulders in which only the face is regenerated with the facial identity from <Picture 1>. The hair, ears, collar, background, "
        "lighting, camera framing, head movement and timing keep their source look and layout." % (length / FPS),
        "",
        "retention_analysis:",
        "<Subject 1> (face): fully_preserved - the referenced facial identity, skin tone and facial hair"
        + (", with head movement, gaze, expressions and mouth shapes following <Video 1>." if follow_mouth else "."),
        "<Audio 1>: fully_copy - the original soundtrack is the complete final audio.",
        "",
        "detailed_description:",
        "Realistic live-action close-up with sharp natural skin texture, clearly formed eyes, lips and teeth, and the source lighting. "
        "The mouth moves naturally with the soundtrack.",
    ]
    return "\n".join(lines)


class H3FaceChunks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crops": ("IMAGE", {"tooltip": "Face crops of the whole video from H3 Face Track + Crop, 24 fps."}),
                "masks": ("MASK", {"tooltip": "Edit masks on the crops (white = regenerate)."}),
                "audio": ("AUDIO", {"tooltip": "The video's soundtrack; each chunk gets its part."}),
                "transform": ("H3FACEXFORM",),
                "chunk_seconds": ("FLOAT", {"default": 6.5, "min": 2.5, "max": 15.0, "step": 0.1, "tooltip": "Longest chunk per H3 generation, rounded down to the 17k+5 grid. Chunks overlap by at least 1 s and are crossfaded."}),
                "skip_faces_above": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 1.0, "step": 0.01, "tooltip": "Faces taller than this share of the frame height are left as they are; chunks with only such faces are not generated."}),
            },
            "optional": {
                "follow_mouth": ("IMAGE", {"tooltip": "Blurred crops. When connected, H3 follows their head movement, expressions and mouth shapes; otherwise it draws them anew from the picture."}),
                "picture_text": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "AUDIO", "STRING", "H3FACECHUNKS")
    RETURN_NAMES = ("crops", "masks", "follow_mouth", "audio", "prompt", "plan")
    OUTPUT_IS_LIST = (True, True, True, True, True, False)
    FUNCTION = "split"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = "Splits face crops into independent overlapping chunks, one H3 generation each; H3 Face Join puts them back together."

    def split(self, crops, masks, audio, transform, chunk_seconds, skip_faces_above, follow_mouth=None, picture_text=""):
        total = crops.shape[0]
        height = transform["src_size"][1]
        face = [box[3] / transform["crop_factor"] / height for box in transform["boxes"]]
        weights = [min(max((skip_faces_above + FACE_RAMP / 2 - f) / FACE_RAMP, 0.0), 1.0) for f in face]

        length = grid_floor(int(chunk_seconds * FPS))
        if total <= length:
            length = grid_floor(total + 16)
            starts = [0]
        else:
            n = math.ceil((total - FACE_OVERLAP) / (length - FACE_OVERLAP))
            starts = [round(i * (total - length) / (n - 1)) for i in range(n)]
        kept = [s for s in starts if any(weights[s:s + length])] or starts[:1]

        def chunk(x, s):
            part = x[s:s + length]
            if part.shape[0] < length:
                part = torch.cat([part, part[-1:].expand(length - part.shape[0], *part.shape[1:])])
            return part

        rate = audio["sample_rate"]
        wave = audio["waveform"]
        samples = round(length * rate / FPS)
        audios = []
        for s in kept:
            part = wave[..., round(s * rate / FPS):][..., :samples]
            audios.append({"waveform": F.pad(part, (0, samples - part.shape[-1])), "sample_rate": rate})

        desc = " ".join((picture_text or "").split())
        if desc and not desc.endswith("."):
            desc += "."
        prompt = build_face_prompt(desc, follow_mouth is not None, length)
        plan = {"starts": kept, "length": length, "total": total, "weights": weights}
        return ([chunk(crops, s) for s in kept], [chunk(masks, s) for s in kept],
                [chunk(follow_mouth, s) for s in kept] if follow_mouth is not None else [], audios, [prompt] * len(kept), plan)


class H3FaceJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The regenerated chunks, in order."}),
                "plan": ("H3FACECHUNKS",),
                "crops": ("IMAGE", {"tooltip": "The original face crops, kept where no chunk was generated."}),
                "transform": ("H3FACEXFORM",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "H3FACEXFORM")
    RETURN_NAMES = ("crops", "transform")
    FUNCTION = "join"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = "Crossfades the H3 Face Chunks back into one crop video; the transform fades the paste out where faces are left alone."

    def join(self, images, plan, crops, transform):
        plan, crops, transform = plan[0], crops[0], transform[0]
        starts, length, total = plan["starts"], plan["length"], plan["total"]
        acc = torch.zeros_like(crops)
        wsum = torch.zeros(total, 1, 1, 1)
        for k, (s, image) in enumerate(zip(starts, images)):
            n = min(length, total - s)
            # linear ramps across each overlap, falling to 0 at a chunk's end where its neighbour takes over
            w = torch.ones(n)
            if k and starts[k - 1] + length > s:
                o = starts[k - 1] + length - s
                w[:o] = torch.arange(1, o + 1) / (o + 1)
            if k + 1 < len(starts) and s + length > starts[k + 1]:
                o = s + length - starts[k + 1]
                w[n - o:] = torch.minimum(w[n - o:], torch.arange(o, 0, -1) / (o + 1))
            acc[s:s + n] += image[:n].to(acc) * w[:, None, None, None]
            wsum[s:s + n, 0, 0, 0] += w
        covered = wsum[:, 0, 0, 0] > 0
        out = torch.where(covered[:, None, None, None], acc / wsum.clamp(min=1e-6), crops)
        transform = dict(transform)
        transform["weights"] = [tw * pw * float(c) for tw, pw, c in zip(transform["weights"], plan["weights"], covered.tolist())]
        return (out, transform)


SHARP_BLUR = 1.0
SOFT_BLUR = 8.0


def gaussian_blur(x, sigma):
    r = max(1, math.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def graded_blur(x, sigma):
    """Blur with a per-pixel sigma from SHARP_BLUR to SOFT_BLUR, mixed from blurs at doubling sigmas."""
    t = sigma.clamp(SHARP_BLUR, SOFT_BLUR).log2()
    out = gaussian_blur(x, SHARP_BLUR)
    for i, s in enumerate((2.0, 4.0, SOFT_BLUR)):
        out = torch.lerp(out, gaussian_blur(x, s), (t - i).clamp(0, 1))
    return out


class H3GreyMannequins:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK", {"tooltip": "White over the people to hide, one mask per frame."}),
                "size": ("MASK", {"tooltip": "Per-pixel person height from H3 People Mask."}),
                "blur": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005, "tooltip": "Blur as a share of each person's height (capped at 8 px), so close-ups lose their face and clothing detail while small, distant people keep readable arms and hands."}),
            },
            "optional": {
                "faces": ("MASK", {"tooltip": "Faces from H3 People Mask on a SAM 'face' track. They get the lighter face_blur and keep their lips darker than the skin, so H3 sees when mouths open and close."}),
                "face_size": ("MASK", {"tooltip": "Per-pixel face height from the same H3 People Mask."}),
                "face_blur": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.005, "tooltip": "Blur inside faces as a share of the face's height (1 to 8 px)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "hide"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = ("Turns the masked people into smooth light-grey figures that keep their pose and shading but not their face, "
                   "hair colour or clothes, so a motion reference cannot pull the original people back into the edit.")

    def hide(self, images, mask, size, blur, faces=None, face_size=None, face_blur=0.05):
        n, h, w = images.shape[:3]
        dev = comfy.model_management.get_torch_device()
        out = torch.empty_like(images)
        for i in range(0, n, 16):
            f = images[i:i + 16].to(dev, torch.float32).movedim(-1, 1)
            m = F.interpolate(mask[i:i + 16, None].to(dev, torch.float32), size=(h, w), mode="bilinear")
            height = F.interpolate(size[i:i + 16, None].to(dev, torch.float32), size=(h, w), mode="nearest") * h
            soft = ((height * blur - SHARP_BLUR) / (SOFT_BLUR - SHARP_BLUR)).clamp(0, 1)
            sharp = (f * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
            luma = torch.lerp(gaussian_blur(sharp, SHARP_BLUR), gaussian_blur(sharp, SOFT_BLUR), soft)
            if faces is not None:
                face = F.interpolate(faces[i:i + 16, None].to(dev, torch.float32), size=(h, w), mode="bilinear")
                face_height = F.interpolate(face_size[i:i + 16, None].to(dev, torch.float32), size=(h, w), mode="nearest") * h
                # Faces use the green channel, where lips are darker than skin, so a closed or open mouth still reads in grey.
                luma = torch.lerp(luma, graded_blur(f[:, 1:2], face_height * face_blur), face)
            grey = (0.45 + 0.45 * luma).expand(-1, 3, -1, -1)
            out[i:i + 16] = torch.lerp(f, grey, m).movedim(1, -1).to(out.device, out.dtype)
        return (out,)


def person_mask(m, headroom, hold_frames):
    """One person's [frames, h, w] mask with hold and headroom, and per pixel their height as a share of the frame."""
    n, mh = m.shape[:2]
    if hold_frames:
        m = F.max_pool1d(m.flatten(1).T[None], 2 * hold_frames + 1, stride=1, padding=hold_frames)[0].T.reshape(m.shape)
    inside = m > 0.5
    rows = torch.arange(mh, device=m.device)
    occupied = inside.any(2)
    height = (torch.where(occupied, rows, -1).amax(1) - torch.where(occupied, rows, mh).amin(1) + 1).clamp(min=0)
    top = torch.where(inside, rows.view(1, mh, 1), mh).amin(1, keepdim=True)
    room = (height * headroom).round().view(n, 1, 1)
    # Headroom only above the columns at the top of the person (head, hair, raised hands),
    # not above outstretched arms or along the bottom of columns the person is not in.
    near_top = (top < mh) & (top <= top.amin(2, keepdim=True) + room)
    person = inside | (near_top & (rows.view(1, mh, 1) >= top - room) & (rows.view(1, mh, 1) < top))
    return person, person * (height / mh).view(n, 1, 1)


class H3PeopleMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA",),
                "headroom": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Extra mask above each person, as a share of that person's height."}),
                "hold_frames": ("INT", {"default": 4, "min": 0, "max": 24, "tooltip": "Each person's mask also covers their mask this many frames before and after, so a quick dip or duck does not shrink the area."}),
            },
            "optional": {
                "max_people": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Keep only this many people, the largest over the clip first. 0 keeps everyone."}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask", "size")
    OUTPUT_TOOLTIPS = ("All tracked people, each with headroom above their head.",
                       "Per pixel, the height of the person covering it as a share of the frame height; 0 elsewhere.")
    FUNCTION = "build"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = ("Joins the people SAM tracked into one mask, giving each person their own headroom so a taller replacement, "
                   "bigger hair or a hat is not cut off, and reports each person's size for the grey mannequins.")

    def build(self, track_data, headroom, hold_frames, max_people=0):
        packed = track_data["packed_masks"]
        h, w = track_data["orig_size"]
        if packed is None:
            empty = torch.zeros(track_data["n_frames"], h, w, device=comfy.model_management.intermediate_device())
            return (empty, empty)
        areas = [int(unpack_masks(packed[:, i]).sum()) for i in range(packed.shape[1])]
        order = [i for i in sorted(range(len(areas)), key=lambda i: -areas[i]) if areas[i]]
        union = size = None
        for i in order[:max_people or None]:
            person, person_size = person_mask(unpack_masks(packed[:, i]).float(), headroom, hold_frames)
            union = person if union is None else union | person
            size = person_size if size is None else torch.maximum(size, person_size)
        if union is None:
            empty = torch.zeros(packed.shape[0], h, w, device=comfy.model_management.intermediate_device())
            return (empty, empty)
        mask = F.interpolate(union.float()[:, None], size=(h, w), mode="bilinear")[:, 0]
        size = F.interpolate(size[:, None], size=(h, w), mode="nearest")[:, 0]
        return (mask, size)


def find_shots(images, cut_threshold):
    """(start, end) frame ranges of the shots between hard cuts."""
    grey = F.interpolate(images[..., :3].mean(-1, keepdim=True).movedim(-1, 1).float(), size=(90, 160), mode="area")[:, 0] * 255
    diffs = [0.0] + (grey[1:] - grey[:-1]).abs().mean(dim=(1, 2)).tolist()
    starts = [0]
    for i in range(1, len(diffs)):
        around = sorted(diffs[max(1, i - 6):i] + diffs[i + 1:i + 7])
        median = around[len(around) // 2] if around else 0.0
        if diffs[i] > cut_threshold and diffs[i] > 2.5 * median and i - starts[-1] >= 6:
            starts.append(i)
    return list(zip(starts, starts[1:] + [images.shape[0]]))


# Outline colour per subject on the shot-list frames, the same in every shot and clip.
ROLE_COLOURS = [("blue", (0.25, 0.5, 1.0)), ("orange", (1.0, 0.55, 0.15)), ("green", (0.3, 0.85, 0.35)), ("purple", (0.7, 0.4, 1.0))]


class H3ClipShots:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "This clip's frames at 24 fps."}),
                "shot_prompt": ("STRING", {"forceInput": True, "tooltip": "From H3 One-Click Swap Plan."}),
                "cut_threshold": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 128.0, "step": 0.5, "tooltip": "Mean grey-level change (0-255) between frames that counts as a hard cut. It must also be 2.5x the change around it, so fast motion is not a cut."}),
            },
            "optional": {
                "role_map": ("H3_ROLE_MAP", {"tooltip": "From H3 Role Mask. Outlines each labelled person in their subject's colour, so the shot list names the same people the masks do."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("shot_frames", "prompt")
    FUNCTION = "split"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = ("Finds the hard cuts in a clip and returns the first and last frame of every shot, with a prompt telling a vision "
                   "model how many shots there are and when each one starts.")

    def split(self, images, shot_prompt, cut_threshold, role_map=None):
        bounds = find_shots(images, cut_threshold)
        picks = []
        lines = ["The clip has %d shot%s. The images come in pairs, the first and last frame of each shot:" % (len(bounds), "" if len(bounds) == 1 else "s")]
        for k, (a, b) in enumerate(bounds, 1):
            picks += [a + (b - a) // 6, b - 1 - (b - a) // 6]
            lines.append("Images %d and %d: shot %d, from %s to %s." % (2 * k - 1, 2 * k, k, timestamp(a / FPS), timestamp(b / FPS)))
        frames = images[picks]
        if role_map is not None:
            frames = frames.clone()
            h, w = frames.shape[1:3]
            roles = F.interpolate(role_map[picks, None].float(), size=(h, w), mode="nearest").to(frames.device)
            t = max(3, h // 100)
            shown = []
            for k, (name, colour) in enumerate(ROLE_COLOURS, 1):
                m = (roles == k).float()
                if m.any():
                    edge = (F.max_pool2d(m, 2 * t + 1, stride=1, padding=t) - m)[:, 0] > 0
                    frames[edge] = torch.tensor(colour, dtype=frames.dtype, device=frames.device)
                    shown.append("<Subject %d> in %s" % (k, name))
            if shown:
                lines.append("Each subject is outlined in their colour: %s. Use the outlines only to tell who is who; never mention them." % ", ".join(shown))
        lines.append("Write exactly %d line%s, in this form:" % (len(bounds), "" if len(bounds) == 1 else "s"))
        lines.append("[Shot 1] <the shot>")
        lines += ["[Shot %d] At %s, the camera cuts to <the shot>" % (k, timestamp(a / FPS)) for k, (a, _) in enumerate(bounds[1:], 2)]
        return (frames, shot_prompt.replace(SHOT_TABLE, "\n".join(lines)))


CROP_SIZE = 336
MAX_CROPS = 12
MIN_AREA = 0.002
TRACK_GAP = 12
# close-ups per appearance: the biggest frame of each third of it, so a turn, a step closer or a lost track shows
VIEWS = 3
# images per question and answer tokens per image; a long answer cut off by max_length loses its last labels
BATCH = 8
ANSWER_TOKENS = 60
RETRIES = 2

TRACK_PROMPT = """Each image shows one person SAM tracked in a video: on the left one whole frame, darkened, with that person in colour inside a yellow box and other tracked people blacked out; on the right the same person alone on plain grey at three moments of that appearance, in time order. Describe the person on grey, not anyone else in the frame. The people to find:
%s
%s
For each image, first say in a few words who the person is (hair, skin, clothes, what they hold), then which subject that is (%s), or "other" if it is none of them (an extra, a passer-by, a reflection, a poster, or only an object). Match people by their look: the same person keeps their hair, skin and clothes across shots, even when their role is not visible in a shot. A subject can be in several images of one shot when they leave the frame and come back, but people on screen at the same time are never the same subject. Small people far off in the background are usually extras: name a subject only when their clothes clearly match. Answer with exactly %d line%s, for %s, each with that image's own number and description, in this form:
Image <number>: <who the person is> -> Subject <number>
Image <number>: <what it shows> -> other"""


def fit(image, height, width):
    """[h, w, 3] image scaled to fit a height x width box, centred on dark grey."""
    h, w = image.shape[:2]
    scale = min(height / h, width / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    out = torch.full((height, width, 3), 0.1, device=image.device)
    y, x = (height - nh) // 2, (width - nw) // 2
    out[y:y + nh, x:x + nw] = F.interpolate(image.movedim(-1, 0)[None], size=(nh, nw), mode="area")[0].movedim(0, -1)
    return out


def mask_box(m):
    """y0, y1, x0, x1 of a [h, w] mask."""
    ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
    return ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1


def together(segs, i, j):
    return j in segs[i]["with"]


def track_prompt(data, batch, known):
    """The question for the appearances in batch (indices into data["segments"]), after the (index, subject) pairs in known
    shown first for comparison; the images are numbered from 1 in that order."""
    segs = data["segments"]
    images = [j for j, _ in known] + batch
    lines = []
    for k in sorted({segs[j]["shot"] for j in images}):
        idx = [n for n, j in enumerate(images, 1) if segs[j]["shot"] == k]
        a, b = data["shots"][k]
        which = "Image %d" % idx[0] if len(idx) == 1 else "Images %s" % ", ".join(map(str, idx))
        lines.append("%s: shot %d, from %s to %s." % (which, k + 1, timestamp(a / FPS), timestamp(b / FPS)))
    pairs = ["%d and %d" % (x, y) for x, i in enumerate(images, 1) for y, j in enumerate(images, 1) if x < y and together(segs, i, j)]
    if pairs:
        lines.append("On screen at the same time, so different people: Images %s." % ", ".join(pairs))
    if known:
        lines.append("Already matched, to compare with: %s. Do not answer for them." % ", ".join("Image %d is Subject %d" % (n, r) for n, (_, r) in enumerate(known, 1)))
    subjects = data["subjects"]
    names = "Subject 1" if subjects == 1 else "Subject 1 to Subject %d; there is no other subject number" % subjects
    first, last = len(known) + 1, len(images)
    which = "Image %d" % first if first == last else "Images %d to %d" % (first, last)
    return TRACK_PROMPT % (data["people"], "\n".join(lines), names, len(batch), "" if len(batch) == 1 else "s", which)


class H3SubjectPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject_prompts": ("STRING", {"forceInput": True, "tooltip": "From H3 Advanced Swap Plan."}),
                "clip_index": ("INT", {"forceInput": True, "tooltip": "From Current Clip."}),
                "subject": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "pick"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = "The clip prompt for a pass that replaces only this subject, with their picture as <Picture 1>; the other subjects stay as they are."

    def pick(self, subject_prompts, clip_index, subject):
        return (json.loads(subject_prompts)[clip_index - 1][subject - 1],)


class H3TrackCrops:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA",),
                "images": ("IMAGE", {"tooltip": "The frames SAM tracked."}),
                "people": ("STRING", {"forceInput": True, "tooltip": "From H3 Advanced Swap Plan."}),
                "cut_threshold": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 128.0, "step": 0.5, "tooltip": "Same as H3 Clip Shots."}),
            },
            "optional": {
                "max_crops": ("INT", {"default": MAX_CROPS, "min": 1, "max": 64, "tooltip": "Most pictures for the vision model, the biggest tracks first; smaller ones stay unlabelled."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("crops", "segments")
    OUTPUT_TOOLTIPS = ("One picture per track appearance, for the vision model.", "For H3 Track Roles.")
    FUNCTION = "crop"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = ("Shows each SAM track appearance lit in its frame and close up at three moments, so a vision model can say which subject it is. "
                   "Labelling per shot keeps identities right after hard cuts, where SAM starts new tracks.")

    def crop(self, track_data, images, people, cut_threshold, max_crops=MAX_CROPS):
        packed = track_data["packed_masks"]
        n, h, w = images.shape[:3]
        bounds = find_shots(images, cut_threshold)
        segments = []
        for i in range(0 if packed is None else packed.shape[1]):
            m = unpack_masks(packed[:, i])
            areas = m.sum((1, 2)).tolist()
            least = MIN_AREA * m.shape[1] * m.shape[2]
            for k, (a, b) in enumerate(bounds):
                on = [j for j in range(a, b) if areas[j] > least]
                if not on:
                    continue
                # SAM can give a lost track to someone else when it finds a person again, so each appearance is labelled
                # on its own; the appearances share the shot, split in the middle of the gaps between them
                runs = [[on[0], on[0] + 1]]
                for j in on[1:]:
                    if j - runs[-1][1] >= TRACK_GAP:
                        runs.append([j, j + 1])
                    else:
                        runs[-1][1] = j + 1
                edges = [a] + [(x[1] + y[0]) // 2 for x, y in zip(runs, runs[1:])] + [b]
                for (first, last), s, e in zip(runs, edges, edges[1:]):
                    f = max(range(first, last), key=lambda j: areas[j])
                    step = (last - first) / VIEWS
                    views = [max(range(first + round(step * v), first + round(step * (v + 1))) or [f], key=lambda j: areas[j]) for v in range(VIEWS)]
                    # a third can fall in a short gap where the track is off screen
                    views = [j if areas[j] > least else f for j in views]
                    segments.append({"track": i, "shot": k, "start": s, "end": e, "first": first, "last": last, "frame": f, "views": views, "area": areas[f]})
        segments = sorted(sorted(segments, key=lambda s: -s["area"])[:max_crops], key=lambda s: (s["shot"], s["track"], s["start"]))
        subjects = len(re.findall(r"^<Subject \d+>", people, flags=re.M))
        if not segments:
            return (torch.zeros(1, 64, 64, 3), json.dumps({"subjects": subjects, "people": people, "shots": bounds, "segments": []}))

        dev = comfy.model_management.get_torch_device()
        mask_at = lambda j, track: F.interpolate(unpack_masks(packed[j, track])[None, None].to(dev, torch.float32), size=(h, w), mode="bilinear")[0, 0]
        panel_w = round(CROP_SIZE * w / h)
        sep = torch.ones(CROP_SIZE, 4, 3, device=dev)
        crops = []
        for s in segments:
            frame = images[s["frame"], ..., :3].to(dev, torch.float32)
            m = mask_at(s["frame"], s["track"])
            grey = (frame * torch.tensor([0.299, 0.587, 0.114], device=dev)).sum(-1, keepdim=True)
            others = unpack_masks(packed[s["frame"]])
            others[s["track"]] = False
            others = F.interpolate(others.any(0)[None, None].to(dev, torch.float32), size=(h, w), mode="bilinear")[0, 0]
            # other people go black, so the model cannot describe a bigger person standing in front
            lit = torch.lerp(0.3 * grey * (1 - others[..., None]), frame, m[..., None])
            y0, y1, x0, x1 = mask_box(m)
            # yellow box around the person in the whole-frame view, so a small person behind a big one is still the one meant
            t = max(2, h // 120)
            yellow = torch.tensor([1.0, 0.9, 0.0], device=dev)
            lit[max(0, y0 - t):y0, x0:x1] = lit[y1:y1 + t, x0:x1] = yellow
            lit[y0:y1, max(0, x0 - t):x0] = lit[y0:y1, x1:x1 + t] = yellow
            panels = [fit(lit, CROP_SIZE, panel_w)]
            pixels = []
            for j in s["views"]:
                view, vm = (frame, m) if j == s["frame"] else (images[j, ..., :3].to(dev, torch.float32), mask_at(j, s["track"]))
                alone = torch.lerp(torch.full_like(view, 0.5), view, vm[..., None])
                y0, y1, x0, x1 = mask_box(vm)
                side = max(y1 - y0, x1 - x0) * 1.1
                cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
                ya, yb = max(0, round(cy - side / 2)), min(h, round(cy + side / 2))
                xa, xb = max(0, round(cx - side / 2)), min(w, round(cx + side / 2))
                panels += [sep, fit(alone[ya:yb, xa:xb], CROP_SIZE, CROP_SIZE)]
                pixels.append(view[vm > 0.5])
            crops.append(torch.cat(panels, 1))
            q = (torch.cat(pixels) * 3.999).long()
            hist = torch.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64).float()
            s["colour"] = [round(v, 4) for v in (hist / hist.sum()).tolist()]

        # appearances on screen at the same time are different people, unless SAM tracked one person twice and their masks cover each other
        for s in segments:
            s["with"] = []
        for x, s in enumerate(segments):
            for y, o in enumerate(segments[:x]):
                a, b = max(s["first"], o["first"]), min(s["last"], o["last"])
                if s["shot"] == o["shot"] and a < b:
                    ms, mo = unpack_masks(packed[a:b, s["track"]]), unpack_masks(packed[a:b, o["track"]])
                    if (ms & mo).sum().item() < 0.5 * min(ms.sum().item(), mo.sum().item()):
                        s["with"].append(y)
                        o["with"].append(x)
        segments = [{k: s[k] for k in ("track", "shot", "start", "end", "first", "last", "area", "with", "colour")} for s in segments]
        return (torch.stack(crops).to(comfy.model_management.intermediate_device()), json.dumps({"subjects": subjects, "people": people, "shots": bounds, "segments": segments}))


class H3TrackRoles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "Qwen3-VL."}),
                "crops": ("IMAGE", {"tooltip": "From H3 Track Crops."}),
                "segments": ("STRING", {"forceInput": True, "tooltip": "From H3 Track Crops."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("labels",)
    FUNCTION = "parse"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = ("Asks the vision model which subject each track appearance is, a few pictures at a time, and asks again about pictures it skipped "
                   "or gave one subject while on screen together; anything still unclear counts as other and is left untouched.")

    def parse(self, clip, crops, segments):
        data = json.loads(segments)
        segs = data["segments"]
        roles = {}
        todo = list(range(len(segs)))
        for attempt in range(RETRIES + 1):
            # asking again, each subject's biggest appearance labelled so far is shown first, to compare by look
            known = {}
            for j in sorted(roles, key=lambda j: -segs[j]["area"]):
                if attempt and roles[j] and j not in todo:
                    known.setdefault(roles[j], j)
            known = [(j, r) for r, j in sorted(known.items())]
            batches = []
            for k in sorted({segs[j]["shot"] for j in todo}):
                shot = [j for j in todo if segs[j]["shot"] == k]
                if batches and len(batches[-1]) + len(shot) <= BATCH:
                    batches[-1] += shot
                else:
                    batches += [shot[x:x + BATCH] for x in range(0, len(shot), BATCH)]
            for batch in batches:
                images = [j for j, _ in known] + batch
                tokens = clip.tokenize(track_prompt(data, batch, known), image=crops[images], min_length=1)
                answer = clip.decode(clip.generate(tokens, do_sample=False, max_length=ANSWER_TOKENS * len(batch)))
                for line in answer.splitlines():
                    m = re.match(r"\W*(?:image\s*)?(\d+)\s*[:=.)-]", line, flags=re.I)
                    answers = re.findall(r"subject\s*(\d+)|\bother\b", line.split("->")[-1], flags=re.I)
                    if m and len(known) < int(m.group(1)) <= len(images) and answers and int(answers[-1] or 0) <= data["subjects"]:
                        roles[images[int(m.group(1)) - 1]] = int(answers[-1] or 0)
            clashes = {j for i in roles for j in roles if i != j and roles[j] and roles[i] == roles[j] and together(segs, i, j)}
            todo = sorted(clashes | {j for j in range(len(segs)) if j not in roles})
            if not todo:
                break
        if len(roles) < len(segs):
            logging.warning("H3 Track Roles: no label for %d of %d track appearances; they stay as they are.", len(segs) - len(roles), len(segs))
        labels = [{"track": s["track"], "start": s["start"], "end": s["end"], "role": roles.get(j, 0)} for j, s in enumerate(segs)]
        # Asked again, the vision model can still name one subject twice in a shot. Only people on screen at the same time can be
        # that (one person whose track broke is the same subject twice). Give the duplicate whose colours better match another
        # subject (as seen elsewhere in the clip, unduplicated) to that subject.
        dupes = {j for j, l in enumerate(labels) for i, o in enumerate(labels) if i != j and l["role"] and o["role"] == l["role"] and together(segs, i, j)}
        claims = {}
        for j in sorted(dupes):
            claims.setdefault((segs[j]["shot"], labels[j]["role"]), []).append(j)
        protos = {}
        for j, l in enumerate(labels):
            if l["role"] and j not in dupes:
                protos.setdefault(l["role"], []).append(segs[j]["colour"])
        protos = {k: [sum(c) / len(cs) for c in zip(*cs)] for k, cs in protos.items()}
        sim = lambda a, b: sum(map(min, a, b))
        for (shot, k), js in claims.items():
            if k not in protos:
                continue
            js = sorted(js, key=lambda j: -sim(segs[j]["colour"], protos[k]))
            for j in js[1:]:
                taken = {l["role"] for i, l in enumerate(labels) if together(segs, i, j)}
                free = [r for r in protos if r not in taken]
                if free:
                    best = max(free, key=lambda r: sim(segs[j]["colour"], protos[r]))
                    if sim(segs[j]["colour"], protos[best]) > sim(segs[j]["colour"], protos[k]):
                        labels[j]["role"] = best
        return (json.dumps(labels),)


class H3RoleMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA",),
                "labels": ("STRING", {"forceInput": True, "tooltip": "From H3 Track Roles."}),
                "headroom": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Extra mask above each person, as a share of that person's height."}),
                "hold_frames": ("INT", {"default": 2, "min": 0, "max": 24, "tooltip": "Each person's mask also covers their mask this many frames before and after, within the shot."}),
            },
            "optional": {
                "subject": ("INT", {"default": 0, "min": 0, "max": 4, "tooltip": "Only this subject, for one pass per person. 0 = every subject with a picture."}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "H3_ROLE_MAP")
    RETURN_NAMES = ("mask", "size", "role_map")
    OUTPUT_TOOLTIPS = ("The people labelled with a subject, each with headroom; everyone else stays untouched.",
                       "Per pixel, the height of the person covering it as a share of the frame height; 0 elsewhere.",
                       "Per pixel, the subject number (0 = nobody), for the shot-list outlines in H3 Clip Shots.")
    FUNCTION = "build"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = "Like H3 People Mask, but only for the tracks labelled with a subject that has a picture, shot by shot."

    def build(self, track_data, labels, headroom, hold_frames, subject=0):
        packed = track_data["packed_masks"]
        h, w = track_data["orig_size"]
        labels = [s for s in json.loads(labels) if s["role"] and subject in (0, s["role"])]
        if packed is None or not labels:
            n = track_data["n_frames"]
            empty = torch.zeros(n, h, w, device=comfy.model_management.intermediate_device())
            return (empty, empty, torch.zeros(n, h, w, dtype=torch.uint8))
        union = size = role_map = None
        for s in labels:
            a, b = s["start"], s["end"]
            m = unpack_masks(packed[:, s["track"]]).float()
            m[:a] = 0
            m[b:] = 0
            person, person_size = person_mask(m, headroom, hold_frames)
            person[:a] = False
            person[b:] = False
            person_size = person_size * person
            if union is None:
                union, size, role_map = person, person_size, person.to(torch.uint8) * s["role"]
            else:
                union = union | person
                size = torch.maximum(size, person_size)
                role_map = torch.where(person, s["role"], role_map).to(torch.uint8)
        mask = F.interpolate(union.float()[:, None], size=(h, w), mode="bilinear")[:, 0]
        size = F.interpolate(size[:, None], size=(h, w), mode="nearest")[:, 0]
        return (mask, size, role_map)


# A labelled appearance starts as its photo only when it is on screen this long and this big: H3 draws a brief or small,
# far-off person as the grey figure of the motion reference instead of the photo.
BRIEF = 12
MIN_SHARE = 0.02


def appearances(track_data, labels):
    """Per label: the runs of frames it is on screen, its largest share of the frame and its default pick (its photo, or 0 to keep it)."""
    packed = track_data["packed_masks"]
    out = []
    for s in labels:
        m = unpack_masks(packed[s["start"]:s["end"], s["track"]])
        share = m.sum((1, 2)) / (m.shape[1] * m.shape[2])
        runs = []
        for i, on in enumerate((share > MIN_AREA).tolist(), s["start"]):
            if on and runs and runs[-1][1] == i:
                runs[-1][1] = i + 1
            elif on:
                runs.append([i, i + 1])
        area = share.max().item()
        big = area >= MIN_SHARE and sum(b - a for a, b in runs) >= BRIEF
        out.append({**s, "on": runs, "area": round(area, 4), "pick": s["role"] if big else 0})
    return out


def save_temp_json(prefix, data):
    folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, folder_paths.get_temp_directory())
    filename = "%s_%05d_.json" % (name, counter)
    with open(os.path.join(folder, filename), "w") as f:
        json.dump(data, f)
    return {"filename": filename, "subfolder": subfolder, "type": "temp"}


class H3MontagePeople:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA",),
                "crops": ("IMAGE", {"tooltip": "From H3 Track Crops."}),
                "labels": ("STRING", {"forceInput": True, "tooltip": "From H3 Track Roles."}),
                "source_frames": ("IMAGE", {"tooltip": "The video frames before SAM's resize, for H3 Montage Tracks to check it gets the same video."}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = ("Saves each labelled track appearance as JSON: its subject, the frames it is on screen, its largest share of the frame, its default pick and "
                   "the picture the vision model saw, so an app can pick a photo per person; and the tracks themselves, which H3 Montage Tracks loads for the render.")

    def save(self, track_data, crops, labels, source_frames):
        people = [{**s, "image": jpeg_url(crop)} for s, crop in zip(appearances(track_data, json.loads(labels)), crops)]
        folder, name, counter, _, _ = folder_paths.get_save_image_path("h3_montage_people", folder_paths.get_temp_directory())
        tracks = None
        if track_data["packed_masks"] is not None:
            # the bit-packed masks are mostly empty: compressed they are a few MB instead of hundreds
            tracks = "%s_%05d_.npz" % (name, counter)
            np.savez_compressed(os.path.join(folder, tracks), packed_masks=track_data["packed_masks"].cpu().numpy(),
                                orig_size=np.array(track_data["orig_size"]), source=np.array(run_hash(source_frames, [], [])))
        return {"ui": {"files": [save_temp_json("h3_montage_people", {"frames": track_data["n_frames"], "tracks": tracks, "people": people})]}}


def jpeg_url(image):
    buf = io.BytesIO()
    Image.fromarray((image[..., :3] * 255).round().byte().cpu().numpy()).save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class H3MontagePicks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA",),
                "labels": ("STRING", {"forceInput": True, "tooltip": "From H3 Track Roles."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("labels",)
    OUTPUT_TOOLTIPS = ("The appearances that become their photo, for H3 Montage Plan.",)
    OUTPUT_NODE = True
    FUNCTION = "pick"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = ("Makes the picks the person list would start with, for a montage without review, and saves which labelled people "
                   "were kept as they are (brief or small in the frame), so the app can say the swap is partial.")

    def pick(self, track_data, labels):
        found = appearances(track_data, json.loads(labels)) if track_data["packed_masks"] is not None else []
        picked = [{"track": a["track"], "start": a["start"], "end": a["end"], "role": a["pick"]} for a in found if a["pick"]]
        covered = {}
        for a in found:
            if a["pick"]:
                covered.setdefault(a["pick"], set()).update(f for x, y in a["on"] for f in range(x, y))
        # a second, brief track on someone who is replaced anyway (SAM sometimes tracks a face or a hand on its own) is not a kept person
        kept = [{"photo": a["role"], "start": a["on"][0][0], "end": a["on"][-1][1], "small": a["area"] < MIN_SHARE} for a in found
                if a["role"] and not a["pick"] and not all(f in covered.get(a["role"], ()) for x, y in a["on"] for f in range(x, y))]
        return {"ui": {"files": [save_temp_json("h3_montage_kept", {"kept": kept})]}, "result": (json.dumps(picked),)}


class H3MontageTracks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"default": "", "tooltip": "The tracks file H3 Montage People saved in ComfyUI's temp folder."}),
                "source_frames": ("IMAGE", {"tooltip": "The video frames before SAM's resize; they must be the ones Find people tracked."}),
            },
        }

    RETURN_TYPES = ("SAM3_TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = "Loads the tracks Find people labelled, so the render masks exactly the people that were checked."

    def load(self, file, source_frames):
        folder = folder_paths.get_temp_directory()
        path = os.path.join(folder, file)
        if not re.fullmatch(r"h3_montage_people_\d+_\.npz", file) or not os.path.isfile(path):
            raise ValueError("The tracks from Find people are gone (ComfyUI clears its temp folder on restart). Find people again.")
        with np.load(path) as data:
            if str(data["source"]) != run_hash(source_frames, [], []):
                raise ValueError("Find people tracked a different video, part or size. Find people again.")
            packed = torch.from_numpy(data["packed_masks"])
            h, w = data["orig_size"].tolist()
        return ({"packed_masks": packed, "n_frames": packed.shape[0], "orig_size": (h, w)},)


PRONOUNS = {"he": "they", "she": "they", "him": "them", "his": "their", "hers": "theirs", "himself": "themself", "herself": "themself",
            "man": "person", "woman": "person", "men": "people", "women": "people", "guy": "person", "girl": "person", "boy": "person"}


def neutral_words(text):
    text = re.sub(r"\bher\b", "their", text, flags=re.I)
    return re.sub(r"\b(%s)\b" % "|".join(PRONOUNS), lambda m: PRONOUNS[m.group(1).lower()], text, flags=re.I)

# Anything the shot list names here, H3 draws on the replacement person.
ACCESSORIES = (r"(?:(?:sun)?glasses|shades|hats?|caps?|beanies?|helmets?|hoods?|headbands?|bandanas?|scarf|scarves|masks?|necklaces?|chains?|earrings?|"
               r"bracelets?|watch(?:es)?|gloves?|outlines?|shirts?|t-shirts?|jackets?|coats?|hoodies?|dress(?:es)?|costumes?|uniforms?|"
               r"outfits?|clothes|clothing|suits?|robes?|bathrobes?|sweaters?|vests?|blouses?|skirts?|pants|trousers|jeans|shorts|boots|shoes|sneakers)")


def neutral_shots(text):
    """The vision model's shot list with gender, clothing and stray formatting removed, so only
    positions and actions reach H3."""
    item = r"(?:(?:a|an|the|her|his|their)\s+)?(?:(?!with\b|in\b)[\w-]+\s+){0,3}?%s\b" % ACCESSORIES
    text = re.sub(r",?\s*\b(?:with|in)\s+%s(?:\s*(?:,|and)\s+%s)*" % (item, item), "", text, flags=re.I)
    text = re.sub(r",?\s*(?:who is |while )?(?:wearing|dressed in|in (?:a|an|her|his|their) (?:black|white|red|blue|leather|denim)\b|(?:identifiable|recogni[sz]able) by)[^,.;]*", "", text, flags=re.I)
    text = re.sub(r"\bher\b(?=\s*(?:[.,;!?]|$)|\s+(?:and|to|at|with|from|in|on|as|while)\b)", "them", text, flags=re.I)
    text = neutral_words(text)
    text = re.sub(r"(^|[.!?]\s+|\]\s+)(they|their|them)\b", lambda m: m.group(1) + m.group(2).capitalize(), text, flags=re.M)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line.startswith("[Shot"))


class H3FillShots:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "tooltip": "The clip prompt from Current Shot, with a {SHOTS} slot."}),
                "shots": ("STRING", {"forceInput": True, "tooltip": "The vision model's shot list."}),
            },
            "optional": {
                "role_map": ("H3_ROLE_MAP", {"tooltip": "From H3 Role Mask. Subjects nobody is labelled as in this clip are named as absent, so H3 does not draw their picture on the people who are there."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "fill"
    CATEGORY = "conditioning/minimax/context_loop"
    DESCRIPTION = "Puts the clip's shot list into its prompt, after stripping gendered words and clothing from it."

    def fill(self, prompt, shots, role_map=None):
        shots = neutral_shots(shots)
        if role_map is not None:
            replaced = [int(k) for k in re.findall(r"^<Subject (\d+)> is the person from", prompt, flags=re.M)]
            present = set(torch.unique(role_map).tolist())
            here = [k for k in replaced if k in present]
            absent = [k for k in replaced if k not in present]
            if here and absent:
                shots += "\nOnly %s %s replaced in this clip; %s %s not appear." % (
                    " and ".join("<Subject %d>" % k for k in here), "is" if len(here) == 1 else "are",
                    " and ".join("<Subject %d>" % k for k in absent), "does" if len(absent) == 1 else "do")
        return (prompt.replace(SHOTS_PLACEHOLDER, shots),)


NODE_CLASS_MAPPINGS = {"H3OneClickSwapPlan": H3OneClickSwapPlan, "H3AdvancedSwapPlan": H3AdvancedSwapPlan, "H3PictureSwapPlan": H3PictureSwapPlan, "H3GreyMannequins": H3GreyMannequins,
                       "H3PeopleMask": H3PeopleMask, "H3ClipShots": H3ClipShots, "H3FillShots": H3FillShots,
                       "H3TrackCrops": H3TrackCrops, "H3TrackRoles": H3TrackRoles, "H3RoleMask": H3RoleMask, "H3SubjectPrompt": H3SubjectPrompt,
                       "H3MontagePeople": H3MontagePeople, "H3MontagePlan": H3MontagePlan, "H3MontagePass": H3MontagePass,
                       "H3MontageTracks": H3MontageTracks, "H3MontagePicks": H3MontagePicks, "H3ObjectSwapPlan": H3ObjectSwapPlan,
                       "H3ObjectPlacementMask": H3ObjectPlacementMask, "H3FaceChunks": H3FaceChunks, "H3FaceJoin": H3FaceJoin}
NODE_DISPLAY_NAME_MAPPINGS = {"H3OneClickSwapPlan": "H3 One-Click Swap Plan", "H3AdvancedSwapPlan": "H3 Advanced Swap Plan (labelled roles)",
                              "H3PictureSwapPlan": "H3 Picture Swap Plan (person + setting from the picture)",
                              "H3GreyMannequins": "H3 Grey Mannequins (hide people)", "H3PeopleMask": "H3 People Mask (headroom per person)",
                              "H3ClipShots": "H3 Clip Shots (find cuts)", "H3FillShots": "H3 Fill Shot List",
                              "H3TrackCrops": "H3 Track Crops (who is each track)", "H3TrackRoles": "H3 Track Roles (read labels)",
                              "H3RoleMask": "H3 Role Mask (labelled people only)", "H3SubjectPrompt": "H3 Subject Prompt (one pass per person)",
                              "H3MontagePeople": "H3 Montage People (save for the person list)", "H3MontagePlan": "H3 Montage Plan (labels, seed, size)",
                              "H3MontagePass": "H3 Montage Pass (this clip's person)", "H3MontageTracks": "H3 Montage Tracks (from Find people)", "H3MontagePicks": "H3 Montage Picks (no review)",
                              "H3ObjectSwapPlan": "H3 Object Swap Plan (change, remove, background, add)",
                              "H3ObjectPlacementMask": "H3 Object Placement Mask (keyframed box)",
                              "H3FaceChunks": "H3 Face Fix Chunks", "H3FaceJoin": "H3 Face Fix Join"}
