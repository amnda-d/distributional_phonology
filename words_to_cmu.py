from nltk.corpus import brown

cmu_to_ipa = {
    'AO': 'ɔ',
    'AA': 'ɑ',
    'IY': 'i',
    'UW': 'u',
    'EH': 'e',
    'IH': 'ɪ',
    'UH': 'ʊ',
    'AH': 'ʌ',
    'AE': 'æ',
    'AX': 'ə',
    'EY': 'eɪ',
    'AY': 'aɪ',
    'OW': 'oʊ',
    'AW': 'aʊ',
    'OY': 'ɔɪ',
    'P': 'p',
    'B': 'b',
    'T': 't',
    'D': 'd',
    'K': 'k',
    'G': 'g',
    'CH': 'tʃ',
    'JH': 'dʒ',
    'F': 'f',
    'V': 'v',
    'TH': 'θ',
    'DH': 'ð',
    'S': 's',
    'Z': 'z',
    'SH': 'ʃ',
    'ZH': 'ʒ',
    'HH': 'h',
    'M': 'm',
    'N': 'n',
    'NG': 'ŋ',
    'L': 'l',
    'R': 'r',
    'ER': 'ɜr',
    'AXR': 'ər',
    'W': 'w',
    'Y': 'j'
}

cmu = {}

with open('cmudict-0.7b.txt',  encoding='latin-1') as f:
    for l in f.readlines():
        if not l.startswith(';;;'):
            line = l.split()
            word = line[0]
            sound = line[1:]
            cmu[word] = [cmu_to_ipa[''.join(x for x in s if not x.isdigit())] for s in sound]


with open('corpora/brown.txt', 'w') as f:
    for w in brown.words(categories=brown.categories()):
        w = w.upper()
        if w in cmu:
            f.write(' '.join(cmu[w]) + '\n')
