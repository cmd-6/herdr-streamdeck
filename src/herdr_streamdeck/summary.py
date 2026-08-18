"""Short pane summaries from a fast hosted model.

A key can say *that* an agent is blocked. It cannot say what it is blocked on,
and that is the thing worth knowing -- "blocked" sends you to the pane, which is
the trip the deck exists to save. So on a status transition the pane's recent
output is sent to a small model that returns the end state in a few words, plus
the replies worth having one tap away.

This fork uses GPT-5.6 Luna through OpenAI's Responses API. Low reasoning gives
the model enough room to distinguish several similar active tasks, while a
strict forced function keeps the tiny display contract deterministic. Chat
Completions cannot combine Luna function tools with reasoning, so the endpoint
is part of the configuration rather than an interchangeable implementation
detail.

The same summariser refreshes active progress at a bounded cadence; the
controller owns that scheduling and rejects a result if the pane changed state
while the request was in flight. The schema is also spelled out in the prompt
because the retry path needs to restate the contract after a malformed answer.

The whole prompt is ours, which is the point: 237 tokens, none of them spent on
somebody else's sandbox rules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

ENDPOINT = "https://api.openai.com/v1/responses"
MODEL = "gpt-5.6-luna"

TOOL_NAME = "label_pane"
"""The one tool the model is forced to call. See Summariser._body."""

REPLY_KINDS = ("affirmative", "negative", "proceed", "alternative")

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["waiting", "summary", "responses"],
    "properties": {
        "waiting": {
            "type": "boolean",
            "description": "True only if the agent is blocked awaiting user input.",
        },
        "summary": {
            "type": "string",
            "description": (
                "The thread's standing task, in 4-6 short words, at most 42 "
                "characters including spaces. What it is FOR, not where it "
                "has got to. "
                "When the agent offers alternatives, name them: "
                "'remove or deprecate?', not 'endpoint deprecation'."
            ),
        },
        "responses": {
            "type": "array",
            "description": "One-tap shortcuts: answers if a question was asked, "
            "otherwise the obvious next instructions.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "label", "text"],
                "properties": {
                    "kind": {"type": "string", "enum": list(REPLY_KINDS)},
                    "label": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
        },
    },
}

SHAPE = """Return ONLY this JSON object, with every field present and no extra fields:
{"waiting": <true|false>, "summary": "<the thread's task, 4-6 short words>",
 "responses": [{"kind": "affirmative"|"negative"|"proceed"|"alternative",
                "label": "<1-3 words>", "text": "<full reply>"}]}
`waiting` is required and must always be present. Every response object must have
all three of kind, label and text."""
"""The output shape, kept separate so a retry can restate it verbatim."""


SYSTEM_PROMPT = (
    """You are naming what a coding agent's thread is working on, in 4-6 words,
and generating a few possible short replies. The name goes on a physical key
that the user glances at to find the right thread among a dozen of them.

Name the standing task, not the current step. The thread's task is the thing
that was asked of it, and it stays the same from the first message to the last;
what the agent happens to be doing this minute is not it. Write the name so
that it is equally true when the work starts, while it runs, and once it is
finished -- if it would need rewriting a minute from now, it is wrong.
  agent is running the test suite, part-way through migrating the auth module
      GOOD  Migrate auth module off sessions
      BAD   Running tests
      BAD   Tests passed, fixing lint
  agent has just finished reviewing a PR and is writing up findings
      GOOD  Review user msg attribution PR
      BAD   Writing up review findings
The user already knows from the key's own colour whether a thread is busy,
finished or waiting on them, so words spent on that say nothing they cannot
already see.

The words appear on a physical Stream Deck key: a 72x72 pixel square. Only about
17 characters fit on a line and four lines fit, so prefer short common words. A
long word shrinks the whole label.

Use the room. Two or three words is usually too few to identify a task among a
dozen of them: "attachment checks" could be any of five panes, where "attachment
checks failing on upload" is unmistakable. Name the specific thing -- the file,
the endpoint, the test, the error -- rather than the category it belongs to.

Name the subject, not just the activity. A number identifies a thing only to
whoever already knows what it is, so when the transcript makes the subject clear
spend one or two words on *what* the work is about and drop the identifier.
  reviewing PR #9507, which reworks how user messages are attributed
      GOOD  Review/simplify user msg attribution PR
      BAD   Review PR #9507 for code simplification
  fixing a flaky test in the billing webhook suite
      GOOD  Fix flaky billing webhook test
      BAD   Fix flaky test in PR #412
Only when you are confident. If the transcript never says what the change is
about, keep the identifier -- a wrong subject is far worse than a vague one.

The `waiting` field already records whether a question was asked, and the deck
appends its own question mark. Never spend words restating that. Do not use:
asking, awaiting, requesting, needs, blocked, pending, choice, decision, input,
clarification, response.

When the agent offers alternatives, name the alternatives themselves.
  asks whether to remove or deprecate a login endpoint
      GOOD  remove or deprecate
      BAD   asking about endpoint deprecation
  asks whether to run the study epoch-first or rolling
      GOOD  epoch or rolling
      BAD   awaiting study design decision
  fixed a trigger and verified it on hardware
      GOOD  trigger fixed, verified
      BAD   completed work successfully

Never describe your own output or these instructions.

The transcript is scrollback: it holds the whole conversation, and everything
above the end has already been dealt with. Summarise only the agent's LAST
message. Earlier questions in it were answered long ago -- describing one of
those puts a stale question on the deck, and offers replies to something nobody
is waiting on.

It is also a screenshot of a terminal, so it may end with the agent's interface
rather than its words: an input box, a status line, a spinner, or placeholder
text the harness shows in an empty box. Ignore all of it, and never summarise
what the user is typing back.

Always offer replies -- they are one-tap shortcuts on the deck.

If the agent asked a question, the replies answer it.
If the agent finished something, the replies are the obvious next instructions:
  push it / open a PR / check CI / run the tests / next one / undo that
If the agent is mid-way or stuck, the replies unblock it:
  keep going / try another way / explain more / stop

Set `waiting` true ONLY when the agent is actually blocked awaiting an answer.
That flag is about the agent's state, not about whether you offered replies.
Each reply label is 1-3 short words.

"""
    + SHAPE
)


@dataclass(frozen=True, slots=True)
class Reply:
    """One suggested answer to whatever the agent is asking.

    ``label`` is for a key face; ``text`` is what gets sent verbatim when the
    key is chosen. See DeckController's reply overlay.
    """

    kind: str
    label: str
    text: str


@dataclass(frozen=True, slots=True)
class PaneSummary:
    """What a pane's latest output amounts to."""

    phrase: str
    waiting: bool
    replies: tuple[Reply, ...] = ()
    """Suggested one-tap replies. Answers when `waiting`, next steps otherwise."""

    provisional: bool = False
    """True for the model-free stand-in shown until a real name arrives.

    A stand-in describes the same task as the name that will replace it, so
    "keep the name while the task is unchanged" would otherwise pin the crude
    version permanently and throw the model's away."""

    subject: str = ""
    """The terminal title this phrase was written for.

    A thread's name should hold still while the agent works, so the phrase is
    kept until the *task* changes rather than refreshed whenever the activity
    does. This records which task it describes, so a change can be told from a
    mere status flip. Replies carry no such thing: they answer the latest
    message and are meant to be replaced."""

    @property
    def display(self) -> str:
        """The phrase as it should appear on a key.

        The question mark is appended here rather than asked for, because
        ``waiting`` already carries that fact and a word spent restating it is
        a quarter of the key wasted.
        """
        if self.waiting and not self.phrase.endswith("?"):
            return self.phrase + "?"
        return self.phrase


Transport = Callable[[bytes, float], bytes]
"""Sends a request body, returns the response body. Injectable so the tests can
exercise parsing and failure handling without a network."""


def _urllib_transport(api_key: str) -> Transport:
    def send(body: bytes, timeout: float) -> bytes:
        request = urllib.request.Request(
            ENDPOINT,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            return bytes(data)

    return send


RULE_CHARACTERS = frozenset(
    "\u2500\u2501\u2504\u2505\u2508\u2509\u254c\u254d\u2550\u2015\u2014-=_"
)
"""Characters a terminal draws a horizontal rule out of."""

PROMPT_MARKERS = frozenset("\u276f\u203a\u276d\u00bb")
"""Glyphs a harness starts its input line with: Claude Code's heavy chevron,
Codex's single angle quote, and near relatives.

Only counted at column zero. A dialog listing "  \u276f 1. Resume from summary"
is indented, and it is content -- the question being asked -- not furniture."""

RULE_MINIMUM = 30
"""Shortest run that counts as a rule, in characters.

Long enough that a markdown `---` in the agent's own output is not mistaken for
the top of the input box."""

TRAILING_LINES = 5
"""How close to the very end the box's lower rule must be.

This is what separates the input box from anything else drawn with rules. The
box sits at the bottom of the screen with only a status line or two beneath it,
whereas a dialog the agent is *showing* you has content below it. Without this,
a pane offering "1. Resume from summary / 2. Resume as-is" had its question
stripped as though it were furniture.
"""

BOX_HEIGHT = 3
"""Greatest gap between two rules that still belong to the same box.

A prompt box is rule / input / rule, so its rules are two lines apart."""


def _is_prompt(line: str) -> bool:
    # A set, not a string: `line[:1] in "..."` is a substring test, so every
    # blank line matched and the walk-up chewed through the whole transcript.
    return bool(line) and line[0] in PROMPT_MARKERS


def _is_rule(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < RULE_MINIMUM:
        return False
    return sum(character in RULE_CHARACTERS for character in stripped) / len(stripped) >= 0.9


SPINNER = re.compile(r"^\W\s*\w+[\u2026.]*\s+(for\s+\d|\()")
"""A harness's own progress line: a glyph, a word, and how long it took.

"\u273b Brewed for 1m 7s" and "* Bootstrapping\u2026 (1m 1s \u00b7 \u2193 1.0k tokens)" are not
things the agent said. Left in place they become the last line of the
transcript, and the pointer at the newest message lands on one."""

RIGHT_ALIGNED = 40
"""Leading spaces past which a line is a status bar, not prose.

Measured: the token counters run to 125, 136 and 305 characters of padding
across the sampled panes, and no wrapped agent output comes close."""

LAST_MESSAGE_CHARS = 600
"""How much of the closing block to quote back. Enough to hold a message,
short enough that it cannot drown the transcript it is pointing into."""


def _is_status(line: str) -> bool:
    """A line the harness added under the agent's message."""
    if not line.strip():
        return False
    if len(line) - len(line.lstrip()) >= RIGHT_ALIGNED:
        return True
    return bool(SPINNER.match(line.strip()))


def strip_status_lines(text: str) -> str:
    """Drop the harness's own trailing lines: spinners and counters.

    Only from the end, and only while they keep matching. A progress line in
    the middle of the scrollback is a fair record of what happened; one at the
    end pretends to be the agent's latest word.
    """
    lines = text.rstrip().split("\n")
    while lines and (not lines[-1].strip() or _is_status(lines[-1])):
        lines.pop()
    return "\n".join(lines)


def last_message(text: str) -> str:
    """The final run of non-blank lines: where the agent's latest message ends.

    Quoted back to the model alongside the full scrollback, because on its own
    the scrollback does not say which part is current. A Codex pane holding an
    old "delete or deprecate?" above a newer "retry wrapper added" was labelled
    with the question every time -- and offered replies to it, which would have
    answered something nobody was waiting on.
    """
    block: list[str] = []
    for line in reversed(text.rstrip().split("\n")):
        if line.strip():
            block.append(line)
        elif block:
            break
    return "\n".join(reversed(block))[-LAST_MESSAGE_CHARS:]


def strip_input_box(text: str) -> str:
    """Drop the terminal's own furniture from the end of a transcript.

    A pane read is a screenshot, so it ends with the harness's input box and
    status bar rather than with the agent's words -- and the box contains
    whatever the user is part-way through typing, plus any autocompletion the
    harness has helpfully offered. Summarising that produced labels describing
    the user's half-finished message instead of the agent's answer.

    Anchored on the box's rules rather than on a line count. Claude Code draws
    rule / input / rule and then two status lines, so "drop the last three
    lines" nearly works -- but a pane showing a *dialog* has no input box, and
    dropping its last three lines removes the question being asked.

    Rules alone are not enough. A dialog is drawn with rules too, and Codex
    draws no rules at all -- its input line is a bare chevron with a status line
    under it, and when empty it shows a *placeholder* ("Explain this codebase")
    that reads exactly like something the agent said. So a bare prompt glyph at
    column zero counts as well.

    Two things keep that from eating real content. It must be within
    TRAILING_LINES of the end, which is what separates an input box from a
    dialog the agent is showing you. And the glyph must be the first character
    on the line: a dialog offering "  \u276f 1. Resume from summary" is indented,
    and it is the question, not the furniture.


    A pane with nothing matching is left entirely alone, which is the right
    answer for a harness whose furniture we have never seen -- it degrades to
    the old behaviour rather than guessing.
    """
    lines = text.rstrip().split("\n")
    start = max(0, len(lines) - TRAILING_LINES)
    found = [
        index
        for index in range(start, len(lines))
        if _is_rule(lines[index]) or _is_prompt(lines[index])
    ]
    if not found:
        return text.rstrip()

    # Walk up through the box: a bordered one has its rules BOX_HEIGHT apart,
    # and anything further back belongs to the agent's own output. The line
    # directly above the chevron is the top of the input, consistently, in both
    # harnesses -- a rule in one and a blank in the other.
    cut = found[-1]
    while True:
        above = [
            i
            for i in range(max(0, cut - BOX_HEIGHT), cut)
            if _is_rule(lines[i]) or _is_prompt(lines[i])
        ]
        if not above:
            break
        cut = above[0]
    return "\n".join(lines[:cut]).rstrip()


_NOT_IN_A_PHRASE = frozenset('{}[]"\\\n\r\t')
"""Characters that cannot occur in a label but do occur in leaked JSON.

Whitespace is not enough of a test. A quantized model emitted
``inverted]responses`` -- one "word" by any spacing rule, and unmistakably a
fragment of the serialiser bleeding into a string field.
"""

MAX_PHRASE_WORDS = 8
MAX_PHRASE_CHARS = 58
"""Generous ceilings, not the target.

The prompt asks for 4-6 words and 42 characters; these only catch a model that
has started writing prose. Rejecting at the target would throw away good labels
that ran one word long, which is a worse trade than rendering them slightly
smaller.
"""


def _phrase(value: object) -> str | None:
    """A short label, or None if the field is anything else.

    Rejecting rather than repairing is deliberate: a mangled label rendered on
    a key looks like a bug in the deck, and a missing summary looks like
    nothing at all. The second failure is much cheaper.
    """
    if not isinstance(value, str):
        return None
    phrase = " ".join(value.split()).strip(" ,;:-")
    if not phrase:
        return None
    if _NOT_IN_A_PHRASE & set(phrase):
        return None
    # A label with no letters in it is not a label. gpt-oss on the raw
    # completions endpoint answered "..." with replies of ["..."] on five of
    # six real panes -- structurally perfect, and it would have gone straight to
    # a key. Conformance was never the same thing as usefulness.
    if not any(character.isalpha() for character in phrase):
        return None
    if len(phrase.split()) > MAX_PHRASE_WORDS or len(phrase) > MAX_PHRASE_CHARS:
        return None
    return phrase


def parse(payload: object) -> PaneSummary | None:
    """Turn a model response into a summary, or None if it does not conform."""
    summary, _ = check(payload)
    return summary


def check(payload: object) -> tuple[PaneSummary | None, str]:
    """Turn a model response into a summary, or None if it does not conform.

    A response that omits ``waiting`` is discarded rather than defaulted.
    The flag decides whether the key shows a question mark, and guessing it
    either way puts a wrong claim on the deck.

    Note that ``waiting`` no longer gates the replies. It used to: replies were
    only meaningful as answers, so a pane that was not being asked anything had
    nothing to offer. Now replies double as next-step shortcuts -- `push it`,
    `check CI` -- which are most useful precisely when the agent has *finished*
    and is not waiting for anything.
    """
    if not isinstance(payload, dict):
        return None, "the response was not a JSON object"

    waiting = payload.get("waiting")
    if not isinstance(waiting, bool):
        return None, "`waiting` was missing or was not true/false"

    phrase = _phrase(payload.get("summary"))
    if phrase is None:
        return None, ("`summary` must be a few short words with no JSON punctuation in it")

    replies: list[Reply] = []
    raw_replies = payload.get("responses")
    if isinstance(raw_replies, list):
        for item in raw_replies:
            if not isinstance(item, dict):
                continue
            kind, label, text = item.get("kind"), item.get("label"), item.get("text")
            if kind not in REPLY_KINDS:
                continue
            if not isinstance(label, str) or not isinstance(text, str):
                continue
            if not label.strip() or not text.strip():
                continue
            replies.append(Reply(kind=kind, label=label.strip(), text=text.strip()))

    return PaneSummary(phrase=phrase, waiting=waiting, replies=tuple(replies)), ""


@dataclass
class Summariser:
    """Calls the model. Every failure returns None rather than raising.

    The deck must survive the summary service being slow, broken, rate-limited
    or unreachable, because it is an enhancement to a display that already works
    without it. Nothing here is allowed to take the deck down with it.
    """

    transport: Transport
    timeout: float = 6.0
    max_replies: int = 3
    """How many replies to ask for.

    One per row of the deck: the overlay puts them in a single column, so a
    fourth suggestion is unreachable however good it is. Asking for exactly what
    fits stops the model spending tokens on an option nobody can press -- it
    returned four unprompted in 13 of 24 trials.
    """

    attempts: int = 3
    """How many times to ask before giving up.

    A rejected response is usually one field away from usable, and the model is
    fast enough that a correction round trip still lands inside a second. Each
    retry shows the model its own output and names what was wrong with it.
    """

    max_chars: int = 3000
    """How much scrollback to send. Measured: 3000 characters of real herdr pane
    output -- box drawing, spinners, status lines and all -- is about 240 prompt
    tokens and summarises correctly. More context did not improve the answer."""

    def _question(self, transcript: str) -> str:
        """The user turn: the scrollback, and a pointer at the end of it."""
        body = strip_status_lines(strip_input_box(transcript))[-self.max_chars :]
        return (
            f"Agent transcript (scrollback):\n---\n{body}\n---\n\n"
            f"The agent's latest message ends here:\n---\n{last_message(body)}\n---\n"
            "Summarise the agent's latest message. Use the scrollback above only "
            "for context, and never describe a question from earlier in it -- "
            "those were answered already."
        )

    def _schema(self) -> dict[str, Any]:
        """The schema with the reply count bound to this deck's geometry."""
        responses = {**SCHEMA["properties"]["responses"], "maxItems": self.max_replies}
        return {
            **SCHEMA,
            "properties": {**SCHEMA["properties"], "responses": responses},
        }

    def _body(self, messages: list[dict[str, str]]) -> bytes:
        return json.dumps(
            {
                "model": MODEL,
                "input": messages,
                "max_output_tokens": 1024,
                "reasoning": {"effort": "low"},
                "store": False,
                "tools": [
                    {
                        "type": "function",
                        "name": TOOL_NAME,
                        "description": "Label a pane on the Stream Deck.",
                        "parameters": self._schema(),
                        "strict": True,
                    }
                ],
                "tool_choice": {"type": "function", "name": TOOL_NAME},
            }
        ).encode()

    async def _once(self, messages: list[dict[str, str]]) -> str | None:
        """One request. Returns the raw assistant content, or None if the call
        itself failed -- as opposed to succeeding with an unusable answer."""
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(self.transport, self._body(messages), self.timeout),
                timeout=self.timeout + 1.0,
            )
        except TimeoutError:
            logger.debug("summary timed out after %.1fs", self.timeout)
            return None
        except urllib.error.HTTPError as exc:
            logger.warning("summary rejected: HTTP %s", exc.code)
            return None
        except Exception:
            logger.warning("summary request failed", exc_info=True)
            return None

        try:
            output = json.loads(raw)["output"]
        except Exception:
            logger.warning("could not read summary response", exc_info=True)
            return None

        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call" and item.get("name") == TOOL_NAME:
                arguments = item.get("arguments")
                if isinstance(arguments, str):
                    return arguments
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict) or content.get("type") != "output_text":
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    return text
        return None

    async def summarise(self, transcript: str) -> PaneSummary | None:
        """Summarise a pane's recent output. None on any failure at all.

        A response that does not conform is retried, with the model shown its
        own output and told what was wrong with it. Transport failures are not
        retried: a timeout or a 429 will not be argued out of, and the deck is
        better off with no summary than with a stalled key.
        """
        if not transcript.strip():
            return None

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._question(transcript)},
        ]

        for attempt in range(1, max(1, self.attempts) + 1):
            content = await self._once(messages)
            if content is None:
                return None

            try:
                payload = json.loads(content)
            except Exception:
                payload = content

            summary, reason = check(payload)
            if summary is not None:
                if attempt > 1:
                    logger.info("summary conformed on attempt %d", attempt)
                return summary

            logger.debug("summary attempt %d rejected: %s", attempt, reason)
            if attempt == max(1, self.attempts):
                break
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"That did not match the schema: {reason}. "
                        "Do not apologise or explain. " + SHAPE
                    ),
                },
            ]

        logger.debug("summary did not conform in %d attempts; discarding", self.attempts)
        return None


def api_key(env: dict[str, str] | None = None) -> str | None:
    """The OpenAI key, from the environment or a configured env file.

    herdr starts plugins itself, so the daemon does not necessarily inherit a
    shell's environment -- reading the file is what makes it work when launched
    as a plugin rather than by hand.
    """
    source = os.environ if env is None else env
    key = source.get("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()

    configured = source.get("OPENAI_API_KEY_FILE")
    candidates = [configured] if configured else []
    candidates.extend(
        os.path.join(candidate, ".env")
        for candidate in (os.getcwd(), os.path.dirname(os.path.dirname(__file__)))
    )
    for path in candidates:
        if not path:
            continue
        try:
            with open(path) as handle:
                for line in handle:
                    name, _, value = line.strip().partition("=")
                    if name.strip() == "OPENAI_API_KEY":
                        cleaned = value.strip().strip('"').strip("'")
                        if cleaned:
                            return cleaned
        except OSError:
            continue
    return None


def build(key: str | None = None, max_replies: int = 3) -> Summariser | None:
    """A summariser, or None when no key is configured.

    None is a supported state, not an error: the deck runs exactly as it did
    before summaries existed.
    """
    resolved = key or api_key()
    if not resolved:
        return None
    return Summariser(transport=_urllib_transport(resolved), max_replies=max_replies)
