"""Generate the Phase 1 decision-and-challenge report as a PDF.

Deliberately not the same document as writeups/01-training.md. That one records
what exists; this one records why it exists, what was chosen over what, and what
went wrong on the way -- which is the material that is actually useful in an
interview, because it is the part nobody can rehearse from a README.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "writeups" / "Nanoserve-Phase1-Report.pdf"

INK = colors.HexColor("#14140F")
MUTED = colors.HexColor("#52514E")
ACCENT = colors.HexColor("#2A78D6")
WARN = colors.HexColor("#EB6834")
RULE = colors.HexColor("#DBDAD5")
BAND = colors.HexColor("#F4F3F0")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

ss = getSampleStyleSheet()


def _s(name, parent="BodyText", **kw):
    return ParagraphStyle(name, parent=ss[parent], **kw)


S = {
    "title": _s("t", "Title", fontName="Helvetica-Bold", fontSize=26, leading=30,
                textColor=INK, alignment=TA_LEFT, spaceAfter=4),
    "subtitle": _s("st", fontSize=12.5, leading=17, textColor=MUTED, spaceAfter=18),
    "h1": _s("h1", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=INK,
             spaceBefore=20, spaceAfter=8),
    "h2": _s("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=INK,
             spaceBefore=14, spaceAfter=5),
    "body": _s("b", fontSize=9.7, leading=14.2, textColor=INK, spaceAfter=8),
    "small": _s("sm", fontSize=8.6, leading=12.4, textColor=MUTED, spaceAfter=6),
    "quote": _s("q", fontSize=9.4, leading=14, textColor=MUTED, leftIndent=12,
                borderPadding=0, spaceBefore=4, spaceAfter=10),
    "code": _s("c", fontName="Courier", fontSize=8.3, leading=11.6, textColor=INK,
               backColor=BAND, borderPadding=7, leftIndent=2, spaceBefore=4, spaceAfter=10),
    "cell": _s("ce", fontSize=8.7, leading=12, textColor=INK, spaceAfter=0),
    "cellhead": _s("ch", fontName="Helvetica-Bold", fontSize=8.7, leading=12,
                   textColor=INK, spaceAfter=0),
    "cellmuted": _s("cm", fontSize=8.7, leading=12, textColor=MUTED, spaceAfter=0),
    "lead": _s("l", fontSize=10.6, leading=15.5, textColor=INK, spaceAfter=10),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def rule(space_before=2, space_after=10):
    return HRFlowable(width="100%", thickness=0.6, color=RULE,
                      spaceBefore=space_before, spaceAfter=space_after)


def table(rows, widths, header=True, zebra=True):
    data = []
    for r_i, row in enumerate(rows):
        style = "cellhead" if (header and r_i == 0) else "cell"
        data.append([c if hasattr(c, "wrap") else Paragraph(str(c), S[style]) for c in row])

    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if header:
        cmds += [("LINEBELOW", (0, 0), (-1, 0), 0.9, INK),
                 ("BACKGROUND", (0, 0), (-1, 0), colors.white)]
    if zebra:
        for i in range(1 if header else 0, len(data)):
            if (i % 2) == (0 if header else 1):
                cmds.append(("BACKGROUND", (0, i), (-1, i), BAND))
    return Table(data, colWidths=widths, style=TableStyle(cmds), hAlign="LEFT")


def decision(n, title, chose, over, why, line):
    """One entry in the decision log, kept on a single page."""
    body = [
        P(f'<font color="#2A78D6"><b>{n}.</b></font>  <b>{title}</b>', "h2"),
        table(
            [["Chose", Paragraph(chose, S["cell"])],
             ["Over", Paragraph(over, S["cellmuted"])]],
            [22 * mm, 143 * mm], header=False, zebra=False,
        ),
        Spacer(1, 6),
        P(why),
        P(f'<b>Say it like this:</b>  <i>"{line}"</i>', "quote"),
    ]
    return KeepTogether(body)


def challenge(n, title, symptom, cause, fix, lesson):
    body = [
        P(f'<font color="#EB6834"><b>{n}.</b></font>  <b>{title}</b>', "h2"),
        table(
            [["Symptom", Paragraph(symptom, S["cell"])],
             ["Cause", Paragraph(cause, S["cell"])],
             ["Fix", Paragraph(fix, S["cell"])]],
            [22 * mm, 143 * mm], header=False, zebra=False,
        ),
        Spacer(1, 6),
        P(f'<b>Why it is worth telling:</b>  {lesson}'),
    ]
    return KeepTogether(body)


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------


def decorate(canvas, doc):
    canvas.saveState()
    w, h = A4
    if doc.page > 1:
        canvas.setFont("Helvetica", 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, h - 12 * mm, "Nanoserve — Phase 1 decisions and challenges")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, h - 14 * mm, w - 20 * mm, h - 14 * mm)
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(w - 20 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def build(story):
    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title="Nanoserve - Phase 1 decisions and challenges",
        author="Ayush Kumar",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
    doc.build(story)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


def content():
    st: list = []

    # ---- cover ----
    st += [
        Spacer(1, 6 * mm),
        P("Nanoserve", "title"),
        P("Phase 1 — decisions, trade-offs, and what went wrong<br/>"
          "Training a 27M-parameter language model from scratch", "subtitle"),
        rule(0, 12),
        P("This document is the companion to <font face='Courier'>writeups/01-training.md</font>. "
          "That file records <i>what</i> was built. This one records <b>why</b> — every place a "
          "choice was made between two reasonable options, and every place something broke and "
          "had to be diagnosed.", "lead"),
        P("The second half is the more valuable half. Anyone can describe a transformer. "
          "Far fewer people can say <i>“I measured 33 tokens/sec, expected 3,500, and here is "
          "the four-step diagnosis that got me to 47,000”</i> — because that only comes from "
          "having actually hit the wall.", "lead"),
        Spacer(1, 6),
        table(
            [["Result", "Value", "Notes"],
             ["Parameters", "26,747,392", "tied embeddings; untied would be 30.9M"],
             ["Training tokens", "536,174,127", "Chinchilla-optimal is 535M — a 0.2% miss"],
             ["Validation loss", "<b>1.1485 nats/token</b>", "acceptance bar was &lt; 1.8"],
             ["Perplexity", "<b>3.153</b>", "on held-out TinyStories V2"],
             ["Wall clock", "3.27 hours", "single RTX 3050 Laptop, 4 GB"],
             ["Throughput", "47,000 tokens/s", "2.5× the first working version"],
             ["Tests", "114 passing", "including HF-parity and bit-exact resume"]],
            [34 * mm, 38 * mm, 93 * mm],
        ),
        Spacer(1, 8),
        P("Everything below was measured on the machine, not quoted from a paper. Where a number "
          "turned out to be wrong, the correction is included rather than the number quietly "
          "replaced.", "small"),
        PageBreak(),
    ]

    # ---- section 1 ----
    st += [
        P("1 · What was built", "h1"),
        P("Three components, each written by hand with no library doing the interesting part."),
        Spacer(1, 4),
        table(
            [["Component", "What it is", "Deliberately not used"],
             ["<font face='Courier'>model/transformer.py</font>",
              "Decoder-only transformer: RMSNorm, RoPE, grouped-query attention, SwiGLU, "
              "pre-norm residual blocks",
              "<font face='Courier'>AutoModel</font>, <font face='Courier'>nn.MultiheadAttention</font>"],
             ["<font face='Courier'>model/tokenizer.py</font>",
              "Byte-level BPE: training loop, merge application, encoder, decoder",
              "<font face='Courier'>tokenizers</font>, <font face='Courier'>sentencepiece</font>"],
             ["<font face='Courier'>model/train/</font>",
              "Data pipeline and a resumable training loop with gradient accumulation",
              "<font face='Courier'>Trainer</font>, <font face='Courier'>accelerate</font>"]],
            [42 * mm, 73 * mm, 50 * mm],
        ),
        Spacer(1, 10),
        P("The one thing borrowed from a library is "
          "<font face='Courier'>scaled_dot_product_attention</font>, which is a fused CUDA kernel "
          "rather than a model. Writing attention as three matmuls and a softmax would have been "
          "a correctness exercise, not a systems one, and it would have thrown away the "
          "memory-efficient kernel that Phase 2 needs.", "body"),

        P("The architecture", "h2"),
        table(
            [["", "", "", ""],
             ["<b>d_model</b>", "512", "<b>vocab</b>", "8,192"],
             ["<b>layers</b>", "8", "<b>context</b>", "1,024"],
             ["<b>query heads</b>", "8", "<b>d_ff</b>", "1,408 (SwiGLU)"],
             ["<b>KV heads</b>", "2 (GQA)", "<b>embeddings</b>", "tied"]],
            [26 * mm, 32 * mm, 30 * mm, 77 * mm], header=False,
        ),
        Spacer(1, 8),
        P("The single most consequential number there is <b>2 KV heads</b>. Four query heads share "
          "each KV head, so the cache stores 2 heads instead of 8, which puts the KV cache at "
          "exactly <b>4 KB per token</b>. A 16-token block is 64 KB; a full 1,024-token sequence "
          "is 4 MB. Every memory figure in Phase 2's paged allocator is four times better because "
          "of a decision made in week one."),
        PageBreak(),
    ]

    # ---- section 2 : decisions ----
    st += [
        P("2 · Decisions — what was chosen over what", "h1"),
        P("Sixteen places where two options were both defensible. In an interview these are the "
          "questions that separate “I followed a tutorial” from “I understood the trade-off”.",
          "lead"),
        rule(),
    ]

    st += [
        decision(
            1, "Tied embeddings",
            "Sharing one weight matrix between the input embedding and the output head.",
            "Untied — giving the output projection its own 4.2M parameters.",
            "With an 8,192-token vocabulary and d_model 512, an untied head costs 4,194,304 "
            "parameters — <b>15% of the entire model</b> — to learn a mapping the embedding table "
            "already encodes. Untied the model is 30.9M; tied it is 26.7M. At this scale that "
            "capacity is far better spent in the layers. Tying is also standard for small models "
            "for exactly this reason.",
            "At 27M parameters an untied head is 15% of your budget spent re-learning the "
            "vocabulary. Tying it buys eight layers' worth of capacity back.",
        ),
        decision(
            2, "HuggingFace-compatible parameter names",
            "Naming weights <font face='Courier'>model.layers.N.self_attn.q_proj.weight</font>, "
            "matching HF's Llama exactly.",
            "A cleaner naming scheme of my own.",
            "This turns the correctness test from a conversion script into a plain "
            "<font face='Courier'>load_state_dict</font>. I can instantiate HF's "
            "<font face='Courier'>LlamaForCausalLM</font> with the same shape, copy my weights "
            "straight in, and compare logits. It also means publishing the weights later needs no "
            "remapping. The cost is aesthetic; the benefit is a test I would otherwise have "
            "skipped because it was fiddly.",
            "I named the parameters after HF's Llama so my parity test is a plain state-dict load. "
            "Making the test cheap is how you guarantee it actually gets written.",
        ),
        decision(
            3, "A KV-cache protocol, in week one",
            "Attention calls exactly one method: "
            "<font face='Courier'>cache.update(layer_idx, k, v)</font>.",
            "Writing attention against a concrete cache and refactoring in Phase 2.",
            "Phase 2 replaces a contiguous cache with a paged block allocator. If attention knows "
            "anything about the cache's internals, that swap becomes a rewrite of the model. "
            "Behind a single-method interface it is a new class. This cost about fifteen lines in "
            "week one and removes the largest structural risk from week five.",
            "The paged allocator implements one method. Nothing above it changes — I designed "
            "that seam three weeks before I needed it.",
        ),
        decision(
            4, "Byte-level BPE",
            "The 256 byte values as the base alphabet.",
            "Character-level or word-level vocabulary with an UNK token.",
            "Byte-level means <b>every possible input is representable</b> — there is no UNK "
            "token and no out-of-vocabulary code path that can be wrong, because there is no such "
            "path at all. An emoji or a Cyrillic character the model has never seen decomposes "
            "into bytes it has. This is what GPT-2, GPT-4 and Llama 3 all do.",
            "Byte-level means UNK doesn't exist. There's no out-of-vocabulary branch to get "
            "wrong because there's no branch.",
        ),
        decision(
            5, "Pre-tokenization before BPE",
            "Splitting text on a regex first; merges never cross those boundaries.",
            "Running BPE directly on the raw byte stream.",
            "Without it, BPE learns <font face='Courier'>\". The\"</font> as a single token — the "
            "period and the following word glued together — and the vocabulary fills with "
            "fragments that only match in one context. A test asserts that every learned token "
            "decodes to something the splitter would have produced as a single piece.",
            "Unconstrained BPE learns punctuation glued to the next word. Pre-tokenization is what "
            "stops your vocabulary filling with garbage that only matches in one place.",
        ),
        decision(
            6, "Training BPE on a frequency table",
            "Collapsing the corpus to unique pre-token → count before merging.",
            "Running the merge loop over the 530M-token stream.",
            "This is the difference between hand-written BPE being practical and being an "
            "overnight job. 530,381,032 pre-tokens collapse to <b>59,888 distinct rows</b>. Each "
            "merge then costs work proportional to the number of <i>distinct words</i> containing "
            "the pair, not to corpus size. The full 2.07 GB corpus trains in 16 minutes of pure "
            "Python.",
            "You never run BPE over the token stream. Collapse to a frequency table first — 530 "
            "million pre-tokens become 60,000 rows and the whole thing runs in Python in minutes.",
        ),
        decision(
            7, "Vocabulary size 8,192",
            "8,192 tokens.",
            "The 32k–50k that general-purpose models use.",
            "TinyStories deliberately restricts itself to roughly the vocabulary a child would "
            "use. Published work on this corpus finds the top 8k tokens cover 475.5M of 476.6M "
            "tokens — <b>99.8%</b>. A 50k vocabulary would have spent 4.2M extra embedding "
            "parameters on tokens that essentially never appear. Measured result: 3.965 "
            "bytes/token and <b>98.1% of words encoding to a single token</b>.",
            "8k covers 99.8% of this corpus. A 50k vocabulary would be embedding parameters spent "
            "on tokens that never appear.",
        ),
        decision(
            8, "TinyStories V2 over V1",
            "The GPT-4-only regeneration, 2.08 GB.",
            "V1, available as 945 MB of parquet — less than half the download.",
            "The dataset card states plainly that V1 contains GPT-3.5 generations “of lesser "
            "quality” and that V2 is GPT-4 only and larger. Phase 1's acceptance criterion is "
            "“grammatical English stories”, which makes corpus quality the thing most directly "
            "under test. Paying 1.1 GB of disk for cleaner supervision was an easy trade.",
            "V1 mixes in GPT-3.5 output. When your acceptance test is fluency, the data quality "
            "is the thing being tested — so I took the bigger, cleaner one.",
        ),
        decision(
            9, "uint16 token storage",
            "Storing token ids as unsigned 16-bit.",
            "int32, the default almost everywhere.",
            "8,192 ids fit in 16 bits with room to spare, so the token file is <b>1.0 GB instead "
            "of 2.1 GB</b>. That halves disk, halves page-cache pressure, and halves the bytes "
            "read per training step. The risk is silent wraparound if the vocabulary ever exceeds "
            "65,536, so <font face='Courier'>prepare()</font> raises rather than truncate.",
            "Vocab is 8k, so ids fit in uint16 — half the disk and half the read bandwidth per "
            "step, with an explicit guard against wraparound.",
        ),
        decision(
            10, "Packed flat token stream",
            "Concatenating all documents with EOS separators and sampling random windows.",
            "Padding each story to a fixed length.",
            "TinyStories averages ~198 tokens per story against a 1,024-token context. Padding "
            "each story would leave roughly <b>80% of every batch as padding</b> — 80% of the "
            "compute producing no gradient. Packing means every position in every batch carries "
            "signal. Windows do sometimes straddle a document boundary, and the EOS token is "
            "precisely what teaches the model that the text before it does not predict the text "
            "after.",
            "Stories average 198 tokens against a 1024 context. Padding would have wasted 80% of "
            "every batch — so I packed the stream and let EOS mark the boundaries.",
        ),
        decision(
            11, "bf16 with no GradScaler",
            "bfloat16 mixed precision, no loss scaling.",
            "fp16 plus <font face='Courier'>GradScaler</font>, which the project plan assumed.",
            "The plan was written for a Kaggle T4, which has no bf16 and therefore needs fp16 with "
            "dynamic loss scaling — and carries the loss-scale-collapse failure mode that comes "
            "with it. The RTX 3050 is Ampere, so bf16 is native, and bf16 carries fp32's exponent "
            "range. <b>An entire class of failure disappears</b> along with the code that guards "
            "against it.",
            "The plan assumed a T4, which needs fp16 and a GradScaler. My card is Ampere, so bf16 "
            "is native — and the loss-scale-collapse failure mode simply doesn't exist.",
        ),
        decision(
            12, "Weight decay on matrices only",
            "Decaying parameters with 2+ dimensions; norms and biases excluded.",
            "Applying weight decay uniformly to everything.",
            "Decaying a LayerNorm gain pulls it toward zero, which does not regularise the layer — "
            "it changes the function the layer computes. The same argument applies to biases. "
            "Only the projections and the embedding table get decay.",
            "Decaying a norm gain isn't regularisation, it's changing what the layer computes. "
            "Two parameter groups, decay on matrices only.",
        ),
        decision(
            13, "Resumability built first, not later",
            "Full checkpoint state from the first commit: weights, optimizer moments, step "
            "counter, RNG states, and the data sampler's generator.",
            "Saving weights only, and adding the rest when something crashes.",
            "The failure this guards against is not hypothetical: a 3-hour run on a laptop meets "
            "thermal events and Windows updates. But the real argument is subtler — a run that "
            "reloads weights but not optimizer moments <i>still trains</i> and <i>still shows a "
            "falling loss</i>. It is simply worse than it should have been, and nothing surfaces "
            "that except an explicit test.",
            "A resume that drops AdamW's moments still trains and the loss still falls. It's just "
            "quietly worse. That's why the test asserts bit-identical weights.",
        ),
        decision(
            14, "Training locally, not on Kaggle",
            "The 4 GB laptop GPU.",
            "Kaggle's free T4s, as the project plan specified.",
            "Measured rather than assumed: the plan budgeted ~10 GPU-hours on a T4. The 3050 "
            "finished in 3.27 hours. Local also means native bf16, no 12-hour session cap, no "
            "1 GB dataset upload, and a live loss curve. The decision flipped because a "
            "measurement contradicted the plan.",
            "The plan budgeted ten GPU-hours on a T4. I measured my own card first and it came in "
            "at three — so the whole Kaggle pipeline became unnecessary.",
        ),
        decision(
            15, "micro-batch 8",
            "8 sequences per forward pass, with gradient accumulation ×4.",
            "The largest batch that does not crash — which is 24.",
            "See challenge 4. On Windows, exceeding VRAM does not raise an error; the driver backs "
            "the allocation with system RAM over PCIe. Micro-batch 24 “fits” and runs <b>12× "
            "slower</b>. Picking the largest working size would have selected the single worst "
            "option available.",
            "The largest batch that runs isn't the fastest. Past VRAM, Windows pages to system RAM "
            "instead of failing — 24 fits and runs twelve times slower than 8.",
        ),
        decision(
            16, "torch.compile",
            "Fusing the graph with inductor, via triton-windows.",
            "Staying in eager mode.",
            "An eager training step launches roughly 700 tiny kernels. On Windows' WDDM driver "
            "each launch costs ~12.9 µs against 3–5 µs on bare Linux, so the GPU spends most of "
            "its time idle between kernels. Compiling fused them for <b>1.61×</b>. Verified "
            "numerically equivalent: from the same seed, step 60 gave 5.1541 eager and 5.1538 "
            "compiled — fused reduction order, not a behaviour change.",
            "Eager launches ~700 kernels a step and Windows charges 13 microseconds each. "
            "Compiling fused them for 1.6× — and I checked the loss matched to four decimals "
            "before trusting it.",
        ),
    ]

    # No page break here: the section intro plus the first challenge block do
    # not fit together on a fresh page, which left a near-empty page. Each
    # challenge is still KeepTogether, so none of them straddle a break.
    # ---- section 3 : challenges ----
    st += [
        P("3 · Challenges — what broke, and how it was found", "h1"),
        P("Nine things went wrong. Three of them I caused myself, and one of those I caused twice. "
          "The diagnoses are the point: each one is a story with a measurement in it.", "lead"),
        rule(),
    ]

    st += [
        challenge(
            1, "Decode ran at 1% of the hardware roofline",
            "Batch-1 decode measured <b>33 tokens/sec</b>. The card has ~166 GB/s of bandwidth and "
            "the weights are 51 MB, so the bandwidth roofline is ~3,100 tokens/sec.",
            "Four separate things, found in sequence. (i) The benchmark used 3 warm-up iterations "
            "on a GPU sitting at 210 MHz — it never clocked up. (ii) <font face='Courier'>no_grad</font> "
            "instead of <font face='Courier'>inference_mode</font>. (iii) A genuine host sync I had "
            "written: <font face='Courier'>int(position_ids.max())</font> inside the RoPE table's "
            "auto-grow, reading a GPU value on the host once per token. (iv) The real bottleneck — "
            "~700 kernel launches per token at 12.9 µs each under Windows' WDDM driver.",
            "Proper warm-up, <font face='Courier'>inference_mode</font>, and moving the capacity "
            "check to a host-side call. Then <font face='Courier'>torch.compile</font> for the "
            "launch overhead itself.",
            "The arithmetic closes exactly: 701 kernels × 12.9 µs = 9.04 ms against 9.4 ms "
            "measured — 96% explained. It also contains an honest wrong turn. I first claimed "
            "“<font face='Courier'>host-launch == wall</font> proves a sync”. It does not — it is "
            "equally consistent with the host being the bottleneck, which is what it actually was. "
            "<font face='Courier'>torch.cuda.set_sync_debug_mode(\"error\")</font> settled it by "
            "raising on any sync, and the decode step passed clean.",
        ),
        challenge(
            2, "My optimisation made it 20% slower",
            "Replacing a manual <font face='Courier'>repeat_kv</font> with SDPA's native "
            "<font face='Courier'>enable_gqa=True</font> — which avoids materialising an 8 MB "
            "tensor per token — made the step <b>slower</b>: 9.4 → 11.3 ms, and kernel count rose "
            "from 701 to 909.",
            "The fused attention kernels require query and key to have the <i>same</i> head count. "
            "Passing mismatched heads with <font face='Courier'>enable_gqa=True</font> silently "
            "drops SDPA to its unfused MATH backend. Materialising the heads is what <i>buys</i> "
            "the fused kernel.",
            "Reverted, with the measurement recorded in a comment so I do not retry it in three "
            "weeks.",
            "This is the cleanest possible argument for measuring instead of reasoning. The "
            "optimisation was correct on paper — it removed a real copy and 32 real kernel "
            "launches — and it was a pessimisation in practice. A backend probe showed why: "
            "<font face='Courier'>['EFFICIENT_ATTENTION', 'MATH']</font> became "
            "<font face='Courier'>['MATH']</font>.",
        ),
        challenge(
            3, "A benchmark that contradicted itself",
            "A benchmark measured 28,715 tokens/sec. The real training loop measured 18,800. "
            "Re-running the <i>benchmark's own code</i> then gave 435 ms where it had given 285 ms.",
            "Identical code, different machine state. Thermal sampling during a sustained run "
            "showed the GPU was not thermally limited (62 °C) but pinned at exactly <b>30.0 W</b> "
            "with the clock at ~1,000 MHz of a 2,100 MHz maximum. The laptop was on battery, where "
            "the GPU enforces 30 W of a 60 W default. On AC it enforces 75 W and clocks at "
            "~1,950 MHz.",
            "Plug in the charger. Nothing in the code changed.",
            "Both measurements were correct — they measured two different machines. The lesson is "
            "that a benchmark result is only meaningful alongside the machine state that produced "
            "it, which is exactly why serious benchmark harnesses record hardware and driver "
            "versions in their output. The largest single speedup in this project was not a code "
            "change.",
        ),
        challenge(
            4, "Exceeding VRAM did not fail — it got 12× slower",
            "A micro-batch sweep showed sizes 1 through 24 all running, with throughput peaking at "
            "8 and then <b>collapsing</b>: 28,715 → 13,350 → 6,938 → 2,485 tokens/sec.",
            "On Windows the WDDM driver does not refuse an allocation that exceeds VRAM. It backs "
            "it with system RAM over PCIe. Micro-batch 24 reserved 8,586 MB on a 4,096 MB card and "
            "ran anyway — 12× slower than micro-batch 8.",
            "The benchmark now reports the <i>fastest</i> size and flags over-commitment "
            "explicitly, rather than reporting the largest that does not raise.",
            "My first version of that script printed “largest fitting micro-batch: 24”, which "
            "would have recommended <b>the single worst option in the table</b>. A benchmark that "
            "measures the wrong quantity is more dangerous than no benchmark, because it carries "
            "authority.",
        ),
        challenge(
            5, "An RMSNorm bug that returns zeros, not NaN",
            "Writing a test to prove the fp32 reduction in RMSNorm was necessary, I asserted the "
            "naive fp16 version would produce NaN. The test failed — the output was finite.",
            "In fp16 an activation of 1e3 squares to 1e6, which overflows to "
            "<font face='Courier'>inf</font>. The mean is then "
            "<font face='Courier'>inf</font>, <font face='Courier'>rsqrt(inf)</font> is <b>0</b>, "
            "and the layer returns <b>all zeros</b>.",
            "The test now asserts the value, not just finiteness — because finiteness passes.",
            "This is worse than a NaN in every way that matters. A NaN propagates loudly and "
            "stops the run. Silent zeros annihilate the activation, nothing raises, no NaN "
            "appears, and the only symptom is a model that does not learn as well as it should. "
            "I found it by writing a test to prove my own code was necessary.",
        ),
        challenge(
            6, "The dataset's own train/valid split leaks",
            "Checking the downloaded files, the train file ends "
            "<font face='Courier'>...She said, \"Yo</font> and the validation file begins "
            "<font face='Courier'>u don't have to be scared of the loud dog...</font>",
            "The official split is a <b>byte offset, not a document boundary</b>. It cuts a story "
            "in half — the same story appears on both sides.",
            "<font face='Courier'>drop_last_document</font> on train and "
            "<font face='Courier'>drop_first_document</font> on validation.",
            "One story in 2.7 million is a negligible leak, and the honest answer is that it would "
            "not have changed any number in this report. But it cost four lines to remove, and "
            "“I checked the boundaries of my train/test split” is a different sentence from "
            "“I assumed the published split was clean”.",
        ),
        challenge(
            7, "best.pt pointed at a worse model",
            "After training, loading "
            "<font face='Courier'>best.pt</font> reported step 16,000 — not the final step 16,364.",
            "The final evaluation in the training loop ran <i>after</i> the last checkpoint save "
            "and was never compared against <font face='Courier'>best_val</font>. A run that "
            "improves on its closing steps therefore leaves "
            "<font face='Courier'>best.pt</font> pointing at a strictly worse model.",
            "Compare the final eval too. Measured over 200 validation batches: step 16,000 gave "
            "1.1496, step 16,364 gave 1.1485.",
            "The gap is small — 0.001 nats. The failure mode is not: this is precisely the bug "
            "that publishes the wrong weights to a public model hub, and nobody would ever notice.",
        ),
        challenge(
            8, "Task Manager reported 0% GPU during training",
            "With training running and <font face='Courier'>nvidia-smi</font> reporting 100% "
            "utilisation, Windows Task Manager showed the GPU at 0%.",
            "Task Manager's default graphs read the <b>3D engine</b> and exclude CUDA. Confirmed "
            "on the machine rather than taken on faith: Windows' own "
            "<font face='Courier'>GPU Engine</font> performance counter read "
            "<font face='Courier'>3d = 19.3%</font> at the same instant "
            "<font face='Courier'>nvidia-smi</font> read <font face='Courier'>util = 100%</font>.",
            "Nothing to fix in the code. In Task Manager, right-click the GPU graph and switch a "
            "pane to <b>Cuda</b>.",
            "Worth knowing because it is a false alarm that looks exactly like a real one, and "
            "acting on it — “the GPU is idle, something is broken” — would send you debugging a "
            "problem that does not exist.",
        ),
        challenge(
            9, "torch.compile poisons your checkpoint keys",
            "<font face='Courier'>torch.compile</font> returns a wrapper module. Its "
            "<font face='Courier'>state_dict</font> keys are prefixed with "
            "<font face='Courier'>_orig_mod.</font>",
            "Saving the compiled wrapper would make every checkpoint incompatible with an "
            "uncompiled run — and with the export used to publish the weights.",
            "Keep the uncompiled module as the thing that is saved and loaded; compile a separate "
            "reference and use that only for the forward pass. Asserted in a test: 75 weight keys, "
            "0 polluted, and the checkpoint loads into a plain model.",
            "A one-line trap with a delayed blast radius. You would not discover it until weeks "
            "later, when the weights refuse to load somewhere else.",
        ),
    ]

    st += [PageBreak()]

    # ---- section 4 ----
    st += [
        P("4 · How to talk about this", "h1"),
        P("Four questions an interviewer is likely to ask, and the honest answer to each.", "lead"),

        P("“Walk me through the project.”", "h2"),
        P("I trained a 27-million-parameter language model from scratch — the transformer, the "
          "tokenizer and the training loop are all hand-written — and I am now building the "
          "inference engine and distributed serving layer around it. Phase 1 finished at 1.15 "
          "nats/token on held-out data in 3.3 hours on a 4 GB laptop GPU. The model being small "
          "is deliberate: it means the distributed system in Phase 3 is actually testable on "
          "hardware I own."),

        P("“How did you know your implementation was correct?”", "h2"),
        P("Loss going down is far too weak a signal — a model with a wrong RoPE or an off-by-one "
          "causal mask still trains, just worse, and you find out three days into a run. So I "
          "built the same architecture in HuggingFace's Llama, copied one set of weights into "
          "both, and required the logits to agree to 2e-5 and greedy generation to match token "
          "for token. Then I tested the failure that still produces fluent text: decoding position "
          "412 with 411 cached, and 256 sequential decode steps checked at every position rather "
          "than only the last."),

        P("“Tell me about something that went wrong.”", "h2"),
        P("The best one is the 33 tokens/sec story — challenge 1 — because it has four layers and "
          "I got the first diagnosis wrong. The short version: I claimed a symptom proved there "
          "was a host sync; it did not, it was equally consistent with the host simply being the "
          "bottleneck. I stopped inferring and used "
          "<font face='Courier'>set_sync_debug_mode(\"error\")</font>, which raises on any sync. "
          "It passed clean. The real answer was 700 kernel launches per token at 13 microseconds "
          "each — and the arithmetic closed to 96%."),
        P("The second-best is challenge 3, because the largest speedup in the project was "
          "plugging in a charger. A benchmark and the real loop disagreed by 50%; the code was "
          "identical; the GPU was power-capped at 30 W on battery versus 75 W on AC. It is a good "
          "story precisely because it is not a clever one — it is a reminder that a benchmark "
          "number without its machine state is not a number."),

        P("“What would you do differently?”", "h2"),
        P("Two things. I wrote the same class of bug twice — a host synchronisation in a hot loop "
          "— and the second time was <i>after</i> I had already found and fixed the first. That "
          "should have become a checklist item the moment I found it, not a lesson I had to learn "
          "again."),
        P("And I would benchmark on Linux from the start. Batch-1 decode on this machine is "
          "launch-bound, and Windows' WDDM driver charges 3–4× what bare Linux does per kernel "
          "launch. That inflates the apparent benefit of batching by roughly 2–3×. It does not "
          "affect any Phase 1 number, but it would have quietly corrupted every Phase 2 "
          "comparison, so the serving benchmarks are going on Linux."),

        P("Numbers worth having memorised", "h2"),
        table(
            [["Number", "What it is"],
             ["26,747,392", "parameters — tied embeddings; 30.9M untied"],
             ["4 KB / token", "KV cache, thanks to 2 KV heads. 16-token block = 64 KB"],
             ["536M / 535M", "training tokens vs Chinchilla-optimal — a 0.2% miss"],
             ["1.1485", "final validation loss, against an acceptance bar of 1.8"],
             ["3.965 bytes/token", "tokenizer compression; 98.1% of words are one token"],
             ["701 × 12.9 µs", "kernels per step × WDDM launch cost = the eager bottleneck"],
             ["30 W → 75 W", "battery vs AC power cap — the largest single speedup"],
             ["2.5×", "total throughput gain: 18,800 → 47,000 tokens/sec"]],
            [40 * mm, 125 * mm],
        ),

        P("5 · What Phase 2 inherits", "h1"),
        P("Four things were built in Phase 1 specifically to make Phase 2 tractable:"),
        table(
            [["Asset", "Why it matters in Phase 2"],
             ["4 KB/token KV cache",
              "The number the paged block manager is sized against. GQA made it 4× smaller than "
              "it would otherwise be."],
             ["<font face='Courier'>KVCache</font> protocol",
              "The paged allocator implements one method. The model does not change."],
             ["<font face='Courier'>model/generate.py</font>",
              "Deliberately plain. The scheduler, paged cache and continuous batching must "
              "reproduce it token for token under greedy decoding."],
             ["Block-boundary tests",
              "Decode already asserted at positions 15, 16, 17, 63, 64, 65 — exactly the 16-token "
              "edges the allocator will cross."]],
            [46 * mm, 119 * mm],
        ),
        Spacer(1, 6),
        P("And one warning carries forward: the serving benchmarks belong on Linux, because "
          "batch-1 decode on this machine is launch-bound and would overstate the benefit of "
          "continuous batching by 2–3×. The whole point of those numbers is that they survive "
          "scrutiny.", "body"),
        Spacer(1, 10),
        rule(),
        P("Generated from <font face='Courier'>scripts/make_report.py</font>. All figures measured "
          "on an RTX 3050 Laptop (4 GB, Ampere), Windows 11, PyTorch 2.11.0+cu128.", "small"),
    ]
    return st


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build(content())
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
