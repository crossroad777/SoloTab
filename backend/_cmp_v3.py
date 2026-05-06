import json,sys,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
sid = '20260429-114939'
with open(f'D:/Music/nextchord-solotab/uploads/{sid}/notes_assigned.json') as f:
    notes=json.load(f)
    notes=notes if isinstance(notes,list) else notes.get('notes',[])
with open(f'D:/Music/nextchord-solotab/uploads/{sid}/beats.json') as f:
    beats=json.load(f)['beats']
measures=[]
for i in range(0, len(beats)-2, 3):
    measures.append((beats[i], beats[i+3] if i+3 < len(beats) else beats[-1]+0.67))
ORIG={1:[7,7,7],2:[7,5,3],3:[3,2,0],4:[0,3,7],5:[12,12,12],6:[12,10,8],7:[8,7,5],8:[5,5,7],
      9:[7,7,7],10:[11,8,7],11:[7,5,3],12:[3,2,0],13:[2,2,2],14:[2,3,2],15:[0,0,0],16:[0,0,0],
      17:[9,9,9],18:[9,7,5],19:[5,4,2],20:[2,4,5],21:[9,9,9],22:[9,7,8],23:[7,8,7],24:[9,9,11],
      25:[9,9,9],26:[12,12,12],27:[12,11,10],28:[9,9,9],29:[9,7,5],30:[4,4,4],31:[4,5,2],32:[0,0,0]}
total_match=0;total_compare=0;em_match=0;em_total=0;major_match=0;major_total=0
for mi,(ms,me) in enumerate(measures[:34]):
    m_num=mi+1
    m_notes=sorted([n for n in notes if ms<=n['start']<me and n.get('string')==1],key=lambda n:n['start'])
    beat_frets=[]
    for b in range(3):
        bs=ms+(me-ms)*b/3;be=ms+(me-ms)*(b+1)/3
        bn=[n for n in m_notes if bs<=n['start']<be]
        beat_frets.append(bn[0]['fret'] if bn else '-')
    orig=ORIG.get(m_num)
    if orig:
        mc=sum(1 for a,o in zip(beat_frets,orig) if a==o)
        total_match+=mc;total_compare+=3
        if m_num<=16: em_match+=mc;em_total+=3
        else: major_match+=mc;major_total+=3
    else: mc=0
    orig_str=str(orig) if orig else "?"
    print("M%2d AI=%20s Orig=%20s %d/3" % (m_num,str(beat_frets),orig_str,mc))
print()
print("Total: %d/%d = %.1f%%" % (total_match,total_compare,total_match/total_compare*100))
print("Em section (M1-16): %d/%d = %.1f%%" % (em_match,em_total,em_match/em_total*100))
print("E major (M17-32): %d/%d = %.1f%%" % (major_match,major_total,major_match/major_total*100))
shifted=sum(1 for n in notes if n.get('_modulation_pitch_shift'))
print("Pitch-shifted: %d" % shifted)
