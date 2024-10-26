from collections import Counter
words = [list("cam_")] * 5 + [list("nham_")] * 4 + [list("tam_")] * 3 + [list("can_")] * 6 + [list("ham_")] * 4
def most_common(words):
    counter = Counter()

    for word in words:
            for i in range(len(word)-1):
                token_pair = '{}{}'.format(word[i],word[i+1])
                counter[token_pair] += 1

    return counter.most_common(1)
def merge_token(words,token):
    token_to_merge = token
    for i in range(len(words)):
        word = words[i]
        for j in range(len(word)-1):
            check_token = '{}{}'.format(word[j],word[j+1])
            if check_token == token_to_merge:
                new_word = word[:j] + [token_to_merge] + word[j+2:]
                words[i] = new_word
def extract_bpe_vocab(words,num_iter=10):
    aSet = set()
    for w in words:
        for i in w:
            aSet.add(i)

    BPE_vocab = {char: 0 for char in aSet}
    for i in range(num_iter):
        token = most_common(words)

        if token:
            BPE_vocab[token[0][0]] = token[0][1]
            aSet.add(token[0][0])
            merge_token(words,token[0][0])
    BPE_vocab_sorted = dict(sorted(BPE_vocab.items(),key = lambda x : x[1],reverse=True))
    return BPE_vocab_sorted

def Encode(token_to_index,sent):
    sent_to_number =[]
    #cam_cam_nham_tam_can_ham
    words = list(sent)
    for token in token_to_index.keys():
        if len(token) == 1:
            continue
        for i in range(len(words)-len(token)+1):

            if "{}{}".format(words[i],words[i+1]) == token:
                words = words[:i]+[token]+words[i+2:]
            if i >= len(words)-len(token):
                break

    sent_to_number =[token_to_index.get(x) for x in words]

    return sent_to_number

def Decode(index_to_token,sen_to_number):
    number_to_sent = [index_to_token[id] for id in sen_to_number]
    number_to_sent = "".join(number_to_sent).replace("_"," ")
    return number_to_sent


BPE_vocab_sorted = extract_bpe_vocab(words,num_iter=10)
token_to_index = {key : i for i,key in enumerate(BPE_vocab_sorted.keys())}
print(token_to_index)
print(BPE_vocab_sorted)
sentence = "cam cam nham tam can ham"
sentence = sentence.replace(" ","_")
print(Encode(token_to_index,sentence))
index_to_token = {id :token for token,id in token_to_index.items()}
print(Decode(index_to_token,Encode(token_to_index,sentence)))


