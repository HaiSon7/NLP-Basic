from collections import Counter
from collections import defaultdict
words = ["ga"] * 5 + ["gấu"] * 6 + ["gan"] * 8 + ["gấm"] * 7 + ["ha"] * 3
def word_freq(words):

    counter =  Counter()

    for word in words:
        counter[word]+=1
    word_freq = {key:val for key,val in counter.items()}
    return word_freq
def generate_splits(word_freqs):
    splits = {keys :[] for keys in word_freqs.keys()}
    for key in splits.keys():
        for i in range(len(key)):
            if i != 0:
                s = "##{}".format(key[i])
                splits[key].append(s)
            else:
                splits[key].append(key[i])
    return splits
def test_generate_splits_output():
    data = defaultdict(int, {'ga': 5, 'gấu': 6, 'gan': 8, 'gấm': 7, 'ha': 3})
    expected_output = {
        'ga': ['g', '##a'],
        'gấu': ['g', '##ấ', '##u'],
        'gan': ['g', '##a', '##n'],
        'gấm': ['g', '##ấ', '##m'],
        'ha': ['h', '##a']
    }
    result = generate_splits(data)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_empty_input():
    empty_data = defaultdict(int)
    result = generate_splits(empty_data)
    assert result == {}, f"Expected an empty dictionary, but got {result}"
    print("Test passed!")
def test_single_word():
    single_word_data = defaultdict(int, {'gấu': 6})
    expected_output = {'gấu': ['g', '##ấ', '##u']}
    result = generate_splits(single_word_data)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_compute_letter_frequencies():
    # Dữ liệu đầu vào
    word_freqs = defaultdict(int, {'ga': 5, 'gấu': 6, 'gan': 8, 'gấm': 7, 'ha': 3})
    splits = {
        'ga': ['g', '##a'],
        'gấu': ['g', '##ấ', '##u'],
        'gan': ['g', '##a', '##n'],
        'gấm': ['g', '##ấ', '##m'],
        'ha': ['h', '##a']
    }

    # Kết quả mong đợi
    expected_output = defaultdict(int, {
        'g': 26,    # 5 (ga) + 6 (gấu) + 8 (gan) + 7 (gấm)
        '##a': 13,  # 5 (ga) + 8 (gan)
        '##ấ': 13,  # 6 (gấu) + 7 (gấm)
        '##u': 6,   # 6 (gấu)
        '##n': 8,   # 8 (gan)
        '##m': 7,   # 7 (gấm)
        'h': 3,     # 3 (ha)
        '##a': 16   # 5 (ga) + 8 (gan) + 3 (ha)
    })

    result = compute_letter_frequencies(splits, word_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_empty_inputs():
    # Test đầu vào rỗng
    empty_splits = {}
    empty_word_freqs = defaultdict(int)
    expected_output = defaultdict(int)

    result = compute_letter_frequencies(empty_splits, empty_word_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_single_entry():
    # Test trường hợp chỉ có một từ
    single_word_freqs = defaultdict(int, {'gấu': 6})
    single_splits = {'gấu': ['g', '##ấ', '##u']}
    expected_output = defaultdict(int, {'g': 6, '##ấ': 6, '##u': 6})

    result = compute_letter_frequencies(single_splits, single_word_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_compute_pair_frequencies():
    # Dữ liệu đầu vào
    word_freqs = defaultdict(int, {'ga': 5, 'gấu': 6, 'gan': 8, 'gấm': 7, 'ha': 3})
    splits = {
        'ga': ['g', '##a'],
        'gấu': ['g', '##ấ', '##u'],
        'gan': ['g', '##a', '##n'],
        'gấm': ['g', '##ấ', '##m'],
        'ha': ['h', '##a']
    }

    # Kết quả mong đợi
    expected_output = defaultdict(int, {
        ('g', '##a'): 5 + 8,      # (5 từ 'ga') + (8 từ 'gan')
        ('g', '##ấ'): 6 + 7,      # (6 từ 'gấu') + (7 từ 'gấm')
        ('##ấ', '##u'): 6,        # (6 từ 'gấu')
        ('##ấ', '##m'): 7,        # (7 từ 'gấm')
        ('##a', '##n'): 8,        # (8 từ 'gan')
        ('h', '##a'): 3           # (3 từ 'ha')
    })

    result = compute_pair_frequencies(splits, word_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_empty_inputs_pair_freq():
    # Test đầu vào rỗng
    empty_splits = {}
    empty_word_freqs = defaultdict(int)
    expected_output = defaultdict(int)

    result = compute_pair_frequencies(empty_splits, empty_word_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_single_entry_pair_freq():
    # Test trường hợp chỉ có một từ
    single_word_freqs = defaultdict(int, {'gấu': 6})
    single_splits = {'gấu': ['g', '##ấ', '##u']}
    expected_output = defaultdict(int, {('g', '##ấ'): 6, ('##ấ', '##u'): 6})

    result = compute_pair_frequencies(single_splits, single_word_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_calculate_pair_scores():
    # Dữ liệu đầu vào
    pair_freqs = defaultdict(int, {
        ('g', '##a'): 13,
        ('g', '##ấ'): 13,
        ('##ấ', '##u'): 6,
        ('##ấ', '##m'): 7,
        ('##a', '##n'): 8,
        ('h', '##a'): 3
    })
    letter_freqs = defaultdict(int, {
        'g': 26,
        '##a': 16,
        '##ấ': 13,
        '##u': 6,
        '##n': 8,
        '##m': 7,
        'h': 3
    })

    # Kết quả mong đợi
    expected_output = {
        ('g', '##a'): 13 / (26 * 16),
        ('g', '##ấ'): 13 / (26 * 13),
        ('##ấ', '##u'): 6 / (13 * 6),
        ('##ấ', '##m'): 7 / (13 * 7),
        ('##a', '##n'): 8 / (16 * 8),
        ('h', '##a'): 3 / (3 * 16)
    }

    result = calculate_pair_scores(letter_freqs,pair_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_empty_inputs_pair_scores():
    # Test đầu vào rỗng
    empty_pair_freqs = defaultdict(int)
    empty_letter_freqs = defaultdict(int)
    expected_output = {}

    result = calculate_pair_scores(empty_pair_freqs, empty_letter_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_single_pair_score():
    # Test trường hợp chỉ có một cặp ký tự
    single_pair_freqs = defaultdict(int, {('g', '##ấ'): 6})
    single_letter_freqs = defaultdict(int, {'g': 10, '##ấ': 5})
    expected_output = {('g', '##ấ'): 6 / (10 * 5)}

    result = calculate_pair_scores(single_letter_freqs,single_pair_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_merge_tokens_standard_case():
    # Test trường hợp b bắt đầu với "##"
    token_a = "g"
    token_b = "##ấu"
    expected_output = "gấu"

    result = merge_tokens(token_a, token_b)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_merge_tokens_no_prefix():
    # Test trường hợp b không bắt đầu với "##"
    token_a = "h"
    token_b = "a"
    expected_output = "ha"

    result = merge_tokens(token_a, token_b)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_merge_tokens_empty_b():
    # Test trường hợp b là một chuỗi rỗng
    token_a = "h"
    token_b = ""
    expected_output = "h"

    result = merge_tokens(token_a, token_b)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_process_split_standard_case():
    # Test trường hợp chuẩn với các token có thể được gộp
    split = ['g', '##ấ', '##u']
    a = 'g'
    b = '##ấ'
    expected_output = ['gấ', '##u']

    result = process_split(split, a, b)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_process_split_no_merge():
    # Test trường hợp không có cặp token nào để gộp
    split = ['g', '##a', '##u']
    a = 'h'
    b = '##a'
    expected_output = ['g', '##a', '##u']

    result = process_split(split, a, b)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_process_split_multiple_merges():
    # Test trường hợp có nhiều cặp token cần gộp
    split = ['g', '##a', 'n', '##g', '##a']
    a = '##g'
    b = '##a'
    expected_output = ['g', '##a', 'n', '##ga']

    result = process_split(split, a, b)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_merge_pair_standard_case():
    # Test trường hợp chuẩn khi cần gộp cặp token (a, b) trong splits
    word_freqs = defaultdict(int, {'gấu': 6, 'gan': 8})
    splits = {
        'gấu': ['g', '##ấ', '##u'],
        'gan': ['g', '##a', '##n']
    }
    a = 'g'
    b = '##ấ'

    expected_output = {
        'gấu': ['gấ', '##u'],
        'gan': ['g', '##a', '##n']
    }

    result = merge_pair(a, b, splits, word_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_merge_pair_no_merge():
    # Test trường hợp không có cặp token nào để gộp
    word_freqs = defaultdict(int, {'gấu': 6})
    splits = {'gấu': ['g', '##ấ', '##u']}
    a = 'h'
    b = '##a'

    expected_output = {'gấu': ['g', '##ấ', '##u']}

    result = merge_pair(a, b, splits, word_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def test_merge_pair_multiple_words():
    # Test trường hợp có nhiều từ cần gộp cặp token (a, b)
    word_freqs = defaultdict(int, {'gấu': 6, 'gan': 8, 'gấm': 7})
    splits = {
        'gấu': ['g', '##ấ', '##u'],
        'gan': ['g', '##a', '##n'],
        'gấm': ['g', '##ấ', '##m']
    }
    a = 'g'
    b = '##ấ'

    expected_output = {
        'gấu': ['gấ', '##u'],
        'gan': ['g', '##a', '##n'],
        'gấm': ['gấ', '##m']
    }

    result = merge_pair(a, b, splits, word_freqs)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")
def compute_letter_frequencies(splits,word_freq):
    letter_freq = defaultdict(int)
    for word,freq in word_freq.items():
        split = splits.get(word)
        for i in split:
            letter_freq[i] += freq


    return letter_freq
def compute_pair_frequencies(splits, word_freqs):
    pair_freq = defaultdict(int)

    for word,freq in word_freqs.items():
        split = splits.get(word)
        for i in range(len(split)-1):
            pair = tuple(split[i:i+2])
            pair_freq[pair] += freq

    return pair_freq
def calculate_pair_scores(letter_freq,pair_freq):
    scores = {}
    for pair,freq in pair_freq.items():

        freq1 = letter_freq.get(pair[0])
        freq2 = letter_freq.get(pair[1])
        if freq1 * freq2 != 0:
            f = freq/(freq1*freq2)
            scores[pair] = f

    return scores
def merge_tokens(a,b):
    token_to_merge = a+b.replace("##","")
    return token_to_merge
def process_split(split, a, b):
    for i in range(len(split)-1):
        if split[i] == a and split[i+1] == b:
            token_to_merge = merge_tokens(a,b)
            split = split[:i]+[token_to_merge]+split[i+2:]
    return split
def merge_pair(a, b, splits, word_freqs):

    for word,_ in word_freqs.items():
        split = splits[word]
        for i in range(len(split)-1):
            if split[i]== a and split[i+1]==b:
                splits[word] = process_split(split,a,b)

    return splits

# # Chạy các test case
# test_generate_splits_output()
# test_empty_input()
# test_single_word()
# # Chạy các unit test
# test_compute_letter_frequencies()
# test_empty_inputs()
# test_single_entry()
# Chạy các unit test
# test_compute_pair_frequencies()
# test_empty_inputs_pair_freq()
# test_single_entry_pair_freq()
# Chạy các unit test
# test_calculate_pair_scores()
# test_empty_inputs_pair_scores()
# test_single_pair_score()
# Chạy các unit test
# test_merge_tokens_standard_case()
# test_merge_tokens_no_prefix()
# test_merge_tokens_empty_b()
# Chạy các unit test
# test_process_split_standard_case()
# test_process_split_no_merge()
# test_process_split_multiple_merges()
# test_merge_pair_standard_case()
# test_merge_pair_no_merge()
# test_merge_pair_multiple_words()

def setUp():
    # Data setup for all test cases
    splits = {
        'ga': ['g', '##a'],
        'gấu': ['g', '##ấ', '##u'],
        'gan': ['g', '##a', '##n'],
        'gấm': ['g', '##ấ', '##m'],
        'ha': ['h', '##a']
    }

    word_freqs = defaultdict(int, {
        'ga': 5,
        'gấu': 6,
        'gan': 8,
        'gấm': 7,
        'ha': 3
    })

    return splits, word_freqs

def test_correct_vocab_items():
    splits, word_freqs = setUp()
    vocab = build_vocab(2, splits, word_freqs)
    print(vocab)
    assert '##a' in vocab

def test_vocab_ordering():
    splits, word_freqs = setUp()
    vocab = build_vocab(3, splits, word_freqs)
    assert '##a' in vocab


def test_empty_input():
    vocab = build_vocab(5, {}, defaultdict(int))
    assert vocab == []  # Should return an empty list

def test_vocab_size_exceeding():
    splits, word_freqs = setUp()
    vocab = build_vocab(10, splits, word_freqs)
    assert len(vocab) <= 10  # Vocabulary should not exceed available tokens

def build_vocab(vocab_size, splits, word_freqs):
    # Initialize vocabulary with all unique characters and subwords
    vocab = set()
    for word, split in splits.items():
        vocab.update(split)
    vocab = sorted(list(vocab))

    while len(vocab) < vocab_size:
        letter_freq = compute_letter_frequencies(splits, word_freqs)
        pair_freq = compute_pair_frequencies(splits, word_freqs)
        scores = calculate_pair_scores(letter_freq, pair_freq)

        if not scores:
            break

        best_pair, _ = max(scores.items(), key=lambda x: x[1])
        new_token = merge_tokens(best_pair[0], best_pair[1])

        if new_token not in vocab:
            vocab.append(new_token)

        splits = merge_pair(best_pair[0], best_pair[1], splits, word_freqs)

    return vocab[:vocab_size]  # Ensure we don't exceed the specified vocab_size

# Example usage
words = ["ga"] * 5 + ["gấu"] * 6 + ["gan"] * 8 + ["gấm"] * 7 + ["ha"] * 3
word_freqs = word_freq(words)
initial_splits = generate_splits(word_freqs)
vocab = build_vocab(15, initial_splits, word_freqs)
print("Final vocabulary:", vocab)
# Run all test functions
test_correct_vocab_items()
test_vocab_ordering()
test_empty_input()
test_vocab_size_exceeding()


print("All tests passed!")

word_freqs = word_freq(words)
print("word_freqs :",end = " ")
print(word_freqs)
splits = generate_splits(word_freqs)
print("Splits : ",end = " ")
print(splits)
letter_freq = compute_letter_frequencies(splits,word_freqs)
print("Letter_Frequencies :" ,end = " ")
print(letter_freq)

pair_freq = compute_pair_frequencies(splits,word_freqs)
print("Pair_Frequencies :" ,end = " ")
print(pair_freq)

scores = calculate_pair_scores(letter_freq,pair_freq)
print("Scores : ", end = " ")
print(scores)

vocab = build_vocab(100,splits,word_freqs)
print("Vocab : ", end = " ")
print(vocab)

def find_longest_word(word, vocab):
    """
    Find the longest subword in the vocabulary that matches the beginning of the word.
    """
    for i in range(len(word), 0, -1):
        subword = word[:i]
        if subword in vocab:
            return subword
        if f"##{subword}" in vocab:  # Kiểm tra cả dạng có tiền tố '##'
            return subword
    return None

def encode_subword(subword, is_first=True):
    """
    Encode a subword by adding '##' prefix if it's not the start of a word.
    """
    return subword if is_first else f"##{subword}"

def encode_word(word, vocab):
    """
    Encode a word using the WordPiece vocabulary.
    Return 'UNK' if the word cannot be fully encoded.
    """
    encoded = []
    is_first = True
    while word:
        subword = find_longest_word(word, vocab)
        if subword is None:
            return ['[UNK]']
        encoded.append(encode_subword(subword, is_first))
        word = word[len(subword):]
        is_first = False
    return encoded

# Example usage

words = ["gấu", "hello", "gan", "gấm"]
for word in words:
    encoded_word = encode_word(word, vocab)
    print(f"Encoded word '{word}': {encoded_word}")

def test_encode_word_full_match():
    # Test khi từ khớp hoàn toàn với một mục trong từ vựng
    global vocab
    vocab = ['gấu', 'gan', 'gấm']
    word = 'gấu'
    expected_output = ['gấu']
    result = encode_word(word, vocab)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")

def test_encode_word_partial_match():
    # Test khi từ cần được chia thành nhiều token
    global vocab
    vocab = ['gấu', 'g', '##ấm', '##an']
    word = 'gấm'
    expected_output = ['g', '##ấm']
    result = encode_word(word, vocab)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")


def test_encode_word_unknown():
    # Test khi từ không khớp với bất kỳ mục nào trong từ vựng
    global vocab
    vocab = ['gấu', 'gan', 'gấm']
    word = 'vịt'
    expected_output = ['[UNK]']
    result = encode_word(word, vocab)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test passed!")


# Chạy các unit test
test_encode_word_full_match()
test_encode_word_partial_match()
test_encode_word_unknown()