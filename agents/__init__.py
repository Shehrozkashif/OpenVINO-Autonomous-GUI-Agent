"""Reasoning layer — one model-facing job per file.

Each agent turns something the machine knows into something a model says, and
parses the answer back into a typed object. None of them decide what happens
next; they hand their answer to core/ and stop.

    router.py      instruction -> SubTask list (and replans, and questions)
    planning.py    subtask + screen -> the next ActionStep(s)
    grounding.py   "the Save button" -> (x, y), via UIA -> OCR -> VLM
    coords.py      a VLM's raw answer -> a screen pixel
    action.py      one ActionStep -> a real click, keypress or UIA call
    reflection.py  did that step work? (only when no ground truth exists)
    prompts.py     every prompt sent to a model, in one place

Rule: agents talk to models through the InferenceClient protocol
(core/inference.py), never to a concrete client class. Swapping OVMS for
another backend must not touch this package.
"""
