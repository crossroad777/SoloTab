from music21 import stream, note

s = stream.Stream()
# The exact sequence of Romance:
# beat 1: E2 (dur=3.0) and B4 (dur=0.33)
# beat 1.33: G4 (dur=0.33)
# beat 1.66: E4 (dur=0.33)

n1 = note.Note("E2")
n1.duration.quarterLength = 3.0
s.insert(0.0, n1)

n2 = note.Note("B4")
n2.duration.quarterLength = 0.3333333333
s.insert(0.0, n2)

n3 = note.Note("G4")
n3.duration.quarterLength = 0.3333333333
s.insert(0.3333333333, n3)

n4 = note.Note("E4")
n4.duration.quarterLength = 0.3333333333
s.insert(0.6666666667, n4)

s.quantize(quarterLengthDivisors=(3,), processDurations=True, inPlace=True)
print(f"Bass dur: {n1.duration.quarterLength}")
