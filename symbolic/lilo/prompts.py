'''
prompts.py

LILO's prompt, reproduced rather than designed.

This file is a direct port of two upstream functions and nothing else:

  * `BasePrompt._get_dsl_description`  (third_party/lilo/src/models/gpt_base.py:263)
  * `Prompt.to_message_list`           (third_party/lilo/src/models/gpt_base.py:158)

An earlier version of this file was my own writing: a hand-authored preamble, a prose blurb
per primitive, a `CRITICAL_RULES` block spelling out de Bruijn index shifts, and two worked
examples on invented concepts. **None of that is in LILO**, so all of it is gone. The measured
justification for removing it rather than merely trimming it: the one rule I added (the
`saved` index shift) made results worse, not better -- typecheck rejections went 13 -> 22.

Two upstream facts shape what is here, both verified in the released code:

  * The DSL description is *generated from the grammar*, not written by hand. Each production
    contributes `name :: type`, with a body and a description only for invented abstractions.
    Base primitives get no description: `function_descriptions` starts empty
    (`laps_grammar.py:104`) and only the library namer ever writes to it
    (`library_namer.py:214`, `gpt_abstraction.py:268`).
  * `dsl_description_prefix` is a *single sentence* per domain (`config_builder.py:100-145`),
    e.g. "This is a domain-specific language for regular expressions that specify string
    transformations." Ours matches that form.

The one adaptation, agreed with the user and disclosed: LILO's target task is given by its
natural language (`final_task_types: ["language"]`), because in re2/CLEVR/LOGO the instruction
fully specifies the task. In SPL it does not -- the geometry lives in the demonstration -- so
the domain supplies the demonstration *as* the task's language. The prompt structure below is
unchanged; only what the domain returns for "language" is richer. That keeps B3-b level with
CaP / Demo2Code / SayCan, which all receive the serialized demonstration.
'''

from __future__ import annotations

from typing import List, Optional, Sequence

# gpt_base.py:39 -- Haskell-style comment prefix for the language line.
PREFIX_LANGUAGE = "-- "
PREFIX_PROGRAM = ""

# gpt_base.py:290-292, verbatim.
DSL_DESCRIPTION_HEADER = (
    "You are an expert programmer working in a language based on lambda calculus.\n"
    "Your goal is to write programs that accomplish the tasks specified by the user.\n"
)

# The single domain sentence, in the form config_builder.py:100-145 uses.
DSL_DESCRIPTION_PREFIX = (
    "This is a domain-specific language for building block structures on a 3-D lattice."
)


# Appended only when `allow_named_variables` is on. Kept apart from the upstream text above
# so the strict-LILO description is byte-identical to `_get_dsl_description`'s output, and
# the deviation is one clearly-marked block rather than a rewrite.
NAMED_VARIABLES_NOTE = '''
You may name the arguments of a lambda instead of using de Bruijn indices:

    (lambda (n) (lambda (s) (loop n (lambda (i) (lambda (s2) (shift RIGHT (place s2)))) s)))

Both forms are accepted, and named arguments are recommended: they are converted for you,
and they avoid the index arithmetic that nested `loop` and `saved` otherwise require.
'''


def dsl_description(grammar, documentation: Optional[dict] = None,
                    names: Optional[dict] = None,
                    named_variables: bool = False,
                    stats_block: str = "") -> str:
    '''Port of `BasePrompt._get_dsl_description` (gpt_base.py:263).

    One entry per grammar production: `name :: type`, plus the body and description for
    invented abstractions only. `documentation` and `names` supply what upstream reads from
    `grammar.function_names` / `grammar.function_descriptions`, both of which the library
    namer populates.
    '''
    documentation = documentation or {}
    names = names or {}

    text = DSL_DESCRIPTION_HEADER
    text += DSL_DESCRIPTION_PREFIX + "\n"
    text += "\nWrite programs using the available functions:\n\n"

    for primitive in grammar.primitives:
        invented = getattr(primitive, "isInvented", False)
        body = str(primitive)
        if invented:
            doc = documentation.get(body) or {}
            fn_name = doc.get("readable_name") or names.get(body) or body
            description = doc.get("description")
        else:
            fn_name = primitive.name
            description = None          # base primitives carry no description upstream
        try:
            fn_type = primitive.infer()
        except Exception:  # noqa: BLE001 - a description must never break a run
            fn_type = "?"

        docstring = f"{fn_name} :: {fn_type}"
        if invented:
            docstring += f"\n{body}"
        if description:
            docstring += f"\ndescription: {description}"
        text += docstring + "\n\n"

    if named_variables:
        text += NAMED_VARIABLES_NOTE
    # The shift_focus delta table, when the demonstration is given as raw coordinates. It
    # belongs here rather than in the per-task language because it describes the primitives,
    # not the task -- and because upstream's per-task slot holds only language and program.
    #
    # This is the same block CaP and Demo2Code get at include_primitive_stats=True, and it
    # is the same mu/sigma the Gaussian evaluator scores with. Telling the model what the
    # direction tokens mean, when the evaluator assumes it knows, is what keeps the two
    # halves of the comparison consistent: `RIGHT :: tdir` alone never says which axis it
    # moves along, which is what defeated the previous run's otherwise-correct staircase.
    if stats_block:
        text += "\n" + stats_block
    return text


def message_list(body_tasks: Sequence, target_language: str) -> List[dict]:
    '''Port of `Prompt.to_message_list` (gpt_base.py:158).

    `body_tasks` is (language, program source) for solved tasks, already in the order they
    should appear -- upstream randomises that ordering per query
    (`sample_generator.py:397`), which the proposer does.

    The `dsl_description` system message is prepended by the caller, matching upstream's
    `prepend_dsl_description` branch.
    '''
    messages: List[dict] = [{"role": "user", "content": "Here are some example programs:"}]
    for language, program in body_tasks:
        messages.append({"role": "user", "content": PREFIX_LANGUAGE + _one_line(language)})
        messages.append({"role": "assistant", "content": PREFIX_PROGRAM + str(program)})
    messages.append({"role": "user",
                     "content": PREFIX_LANGUAGE + _one_line(target_language)})
    return messages


def _one_line(text: str) -> str:
    '''Upstream strips line separators out of the language (gpt_base.py:252-253).'''
    return " ".join(str(text).split())


def _render(position, observation_mode: str) -> str:
    '''One placement, in the run's observation units.

    'lattice'    -> (0,-1,0)         integer cells
    'continuous' -> (0.00,-0.11,0.00) metres relative to the first block, 2 decimals,
                    exactly what `serialize_text._raw` gives CaP and Demo2Code at
                    coordinate_mode='raw'. Matching that rendering is the point: B3-b must
                    read the same demonstration the LLM baselines read.
    '''
    if observation_mode == "continuous":
        return "(" + ",".join(f"{float(v):.2f}" for v in position) + ")"
    return "(" + ",".join(str(int(v)) for v in position) + ")"


def task_language(task, *, include_demonstration: bool = True) -> str:
    '''The task's "language": its instruction, and usually its demonstration.

    In every LILO domain this is whatever `get_language_for_ids` returns. Ours returns the
    instruction together with the demonstrated placements, because the instruction alone does
    not determine the geometry -- see the module docstring.

    `include_demonstration=False` is the images-only modality: the geometry arrives as
    keyframes on the message instead, so repeating it as text would hand the model strictly
    more than either CaP-images or Demo2Code-vlm receives.
    '''
    parts = [task.instruction or task.name]
    if not include_demonstration:
        return parts[0]
    mode = getattr(task, "observation_mode", "lattice")
    for param, positions in task.examples:
        rendered = " ".join(_render(p, mode) for p in positions)
        parts.append(f"{task.param_name}={param} -> {rendered}")
    return " | ".join(parts)
