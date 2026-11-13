"""Speaker alias map: voice-cluster ids (speaker#N from asr.Speakers) -> names, learned from what people say.

Evidence (each a vote for speaker -> name):
  self    the speaker introduces themself: "I'm Leo", "it's Leo here", "my name is Leo", "Leo here"  (weight 3)
  reply   someone addresses a name ("Leo, ..." / "..., Leo?") and a DIFFERENT speaker talks next within
          REPLY_WINDOW seconds: that next speaker is probably the addressee                         (weight 1)
  not     a speaker who says a name vocatively is NOT that name                                        (veto)
A speaker gets a name when its best name has >= MIN_SCORE votes and leads the runner-up by >= 2.
rewrite("speaker#2", text) returns the line with the name substituted when known, so the gist sees "Leo: my keys..."
and the memory owner is right. The assistant's own name is never assigned to a person.
"""
import collections
import re

REPLY_WINDOW = 10.0
MIN_SCORE = 3
SELF = re.compile(r"\b(?:i'm|i am|it's|it is|this is|my name is|call me)\s+([A-Z][a-z]+)\b|^([A-Z][a-z]+) here\b")
VOC_HEAD = re.compile(r"^(?:hey |hi |yo |ok |okay )?([A-Z][a-z]+)\s*[,!?]")
VOC_TAIL = re.compile(r",\s*([A-Z][a-z]+)\s*[?.!]*$")
NOT_NAMES = {"Aria", "Hey", "Ok", "Okay", "Yes", "No", "Well", "So", "Oh", "Um", "Uh", "Wait", "Thanks", "Please",
             "Sorry", "Hi", "Yo", "Look", "Listen", "Dude", "Man", "Guys", "Babe", "Honey", "Mom", "Dad", "Ugh", "Lol"}


class AliasMap:
    def __init__(self, assistant="Aria"):
        self.votes = collections.defaultdict(collections.Counter)
        self.veto = collections.defaultdict(set)
        self.pending = None            # (time, speaker, addressed name)
        self.assistant = assistant

    def _names(self, text):
        out = []
        for rx in (VOC_HEAD, VOC_TAIL):
            m = rx.search(text.strip())
            if m and m.group(1) not in NOT_NAMES and m.group(1) != self.assistant:
                out.append(m.group(1))
        return out

    def observe(self, t, speaker, text):
        # reply evidence from the previous vocative
        if self.pending and speaker != self.pending[1] and t - self.pending[0] <= REPLY_WINDOW:
            self.votes[speaker][self.pending[2]] += 1
        self.pending = None
        m = SELF.search(text)
        if m:
            nm = m.group(1) or m.group(2)
            if nm and nm not in NOT_NAMES and nm != self.assistant:
                self.votes[speaker][nm] += 3
        for nm in self._names(text):
            self.veto[speaker].add(nm)
            self.pending = (t, speaker, nm)

    def name(self, speaker):
        c = [(n, v) for n, v in self.votes[speaker].most_common() if n not in self.veto[speaker]]
        if not c or c[0][1] < MIN_SCORE:
            return None
        if len(c) > 1 and c[0][1] - c[1][1] < 2:
            return None
        # a name already firmly owned by another speaker with a higher score is not reused
        for other, cnt in self.votes.items():
            if other != speaker and cnt.get(c[0][0], 0) > c[0][1] and c[0][0] not in self.veto[other]:
                return None
        return c[0][0]

    def hold_or_release(self, speaker, text):
        """facts from a speaker without a name are held (not written under 'speaker#N'); when the speaker gets a
        name, every held line comes back rewritten so it can be written to memory with the right owner"""
        self.held = getattr(self, "held", collections.defaultdict(list))
        out = []
        for s in list(self.held):
            nm = self.name(s)
            if nm:
                out += [f"{nm}: {x}" for x in self.held.pop(s)]
        if text is not None:
            if self.name(speaker):
                out.append(self.rewrite(speaker, text))
            else:
                self.held[speaker].append(text)
        return out

    def rewrite(self, speaker, text):
        nm = self.name(speaker)
        return f"{nm or speaker}: {text}"

    def table(self):
        return {s: self.name(s) for s in self.votes}
