#!/usr/bin/env python3

from typing import List, Tuple, Dict, Union, Iterable, Iterator
from collections import defaultdict

import regex as re

import pickle

from tqdm import tqdm


# VARIABLES

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# CLASSES

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally)
        a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] | None = None):
        """
        Construct and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally)
        a list of special tokens.
        """
        # initialize vocabulary and merges
        vocab: Dict[int, bytes] = {}
        merges: List[Tuple[bytes, bytes]] = []

        # read vocabulary
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)

       # read merges
        with open(merges_filepath, 'rb') as file:
            merges = pickle.load(file)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        # step 0. format reversed vocab and merges to int
        # print("part 0")
        reversed_vocab = reverse_vocab_dict(self.vocab)
        merges_dict: Dict[Tuple[int, int], int] = {}
        for merge in self.merges:
            tuple1 = reversed_vocab[merge[0]]
            tuple2 = reversed_vocab[merge[1]]
            merges_dict[(tuple1, tuple2)] = reversed_vocab[merge[0] + merge[1]]

        # step 1. pre-tokenize
        # print("part 1")
        text = pretokenize_encode(text, self.special_tokens)

        # step 2. apply the merges
        # print("part 2")
        int_list = map_to_int(text, reversed_vocab, merges_dict)

        return int_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.
        """
        # set up buffer to accumulate partial tokens across chunks
        buffer = ""

        for chunk in iterable:
            # add remainder buffer from previous chunk
            chunk = buffer + chunk

            # find index of last whitespace character
            last_space_index = chunk.rfind(' ')

            # if no whitespace found, yield entire chunk
            if last_space_index == -1:
                token_ids = self.encode(chunk)
                yield from token_ids
                buffer = ""

            else:
                # split chunk at last whitespace chatacter
                tokens = chunk[:last_space_index]
                token_ids = self.encode(tokens)
                yield from token_ids
                buffer = chunk[last_space_index:]

        if buffer:
            # yield remaining partial token in the buffer
            token_ids = self.encode(buffer)
            yield from token_ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        byte_string = b''
        # iterate through ids and convert to byte string
        for id in ids:
            byte = self.vocab[id]
            byte_string += byte

        decoded_text = byte_string.decode("utf-8", errors="replace")

        return decoded_text


# FUNCTIONS

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Given path to an input text file, train a (byte-level) BPE tokenizer.
    """
    ## part 1. vocabulary initialization
    # initialize vocabulary with special tokens
    # print("part1")
    vocab = initialize_vocabulary(special_tokens)

    ## part 2. pre-tokenization
    # load text from file
    # print("part2.1")
    with open(input_path, "r") as file:
        text = file.read()

    # pretokenize text and count
    # print("part2.2")
    byte_counts = pretokenize_text(text)

    ## part 3. compute bpe merges
    # initialize merges dictionary
    # print("part3.1")
    merges: List[Tuple[bytes, bytes]] = []

    # find number of remaining merges
    num_merges = vocab_size - 256 - len(special_tokens)

    # initialize counts for byte pairs
    pair_counts = initialize_byte_pairs(byte_counts)

    # iterate over remaining merges
    # print("part3.2")
    for i in tqdm(range(num_merges)):
        # find highest frequency pair and merge
        max_pair = find_highest_frequency_pair(pair_counts)
        changes = find_byte_changes(byte_counts, max_pair)
        new_byte_counts = merge_bytes(byte_counts, changes)
        new_pair_counts = update_byte_pairs(pair_counts, byte_counts, max_pair, changes)

        # update vocab dictionary
        new_index = 256 + i + len(special_tokens)
        vocab[new_index] = max_pair[0] + max_pair[1]

        # update bpe merges list
        merges.append(max_pair)

        # update byte_counts and pair_counts
        byte_counts = new_byte_counts
        pair_counts = new_pair_counts

    return vocab, merges


def initialize_vocabulary(special_tokens: List[str]) -> Dict[int, bytes]:
    # initialize byte values from 0 to 255: index -> bytes
    vocab: Dict[int, bytes] = {
        x: bytes([x]) for x in range(256)
    }

    # add special tokens
    for i, token in enumerate(special_tokens):
        token_index = 256 + i
        vocab[token_index] = token.encode("utf-8")

    return vocab


def pretokenize_text(text: str) -> Dict[Tuple[bytes], int]:
    # split text into list of pretokens
    # pretokens = re.findall(PAT, text)  # naive implementation gives oom
    pretoken_iter = (match.group() for match in re.finditer(PAT, text))

    # count pretokens
    pretoken_counts = defaultdict(int)
    for pretoken in pretoken_iter:
        pretoken_counts[pretoken] += 1

    # encode unique pretokens into sequence of UTF-8 bytes (int)
    byte_counts = {
        tuple(bytes((i,)) for i in pretoken.encode("utf-8")):
        count for pretoken, count in pretoken_counts.items()
        }

    return byte_counts


def initialize_byte_pairs(byte_counts: Dict[Tuple[bytes], int]) -> Dict[Tuple[bytes], int]:
    pair_counts = defaultdict(int)

    # count initial occurrences for each pair of bytes within pretoken boundaries
    for byte_tuple, count in byte_counts.items():
        for pair in zip(byte_tuple, byte_tuple[1:]):  # for each adjacent pair
            pair_counts[pair] += count

    return dict(pair_counts)


def find_highest_frequency_pair(pair_counts: Dict[Tuple[bytes], int]) -> Tuple[bytes]:
    # find max frequency and associated max pairs
    max_frequency = max(pair_counts.values())
    max_pairs = [pair for pair, frequency in pair_counts.items() if frequency == max_frequency]

    # break ties by preferring lexicographically greater pair
    max_pair = max(max_pairs)

    return max_pair


def find_byte_changes(byte_counts: Dict[Tuple[bytes], int], max_pair: Tuple[bytes]) -> List[Tuple[Tuple[bytes]]]:
    changes = []

    # collect changes where merged pair replaces individual bytes
    for byte_tuple in byte_counts.keys():
        merged_byte_tuple = apply_merges(byte_tuple, max_pair)
        if merged_byte_tuple != byte_tuple:
            changes.append((byte_tuple, merged_byte_tuple))

    return changes


def merge_bytes(byte_counts: Dict[Tuple[bytes], int], changes: List[Tuple[Tuple[bytes]]]) -> Dict[Tuple[bytes], int]:
    # apply changes to byte dictionary
    for old_key, new_key in changes:
        value = byte_counts.pop(old_key)
        byte_counts[new_key] = value

    return byte_counts


def update_byte_pairs(pair_counts: Dict[Tuple[bytes], int], merged_byte_counts: Dict[Tuple[bytes], int], max_pair: Tuple[bytes], changes: List[Tuple[Tuple[bytes]]]) -> Dict[Tuple[bytes], int]:
    # adjust input parameters to correct format
    pair_counts = defaultdict(int, pair_counts)
    merged_pair = max_pair[0] + max_pair[1]

    # filter words that only contain merged pair
    bytes_that_change = [t[1] for t in changes]
    filtered_byte_counts = {byte_tuple: merged_byte_counts[byte_tuple] for byte_tuple in bytes_that_change if byte_tuple in merged_byte_counts}

    # increment byte pairs on either side of merged pair
    for byte_tuple, count in filtered_byte_counts.items():
        if len(byte_tuple) > 1:
            for i, byte in enumerate(byte_tuple):
                if byte == merged_pair:
                    if i == 0:  # start of byte_tuple so increment check right
                        pair_counts[(merged_pair, byte_tuple[i + 1])] += count
                        pair_counts[(max_pair[1], byte_tuple[i + 1])] -= count
                    elif i == len(byte_tuple) - 1:  # end of byte_tuple so only increment left
                        pair_counts[(byte_tuple[i - 1], merged_pair)] += count
                        pair_counts[(byte_tuple[i - 1], max_pair[0])] -= count
                    else:  # middle of sequence so increment both ways
                        pair_counts[(merged_pair, byte_tuple[i + 1])] += count
                        pair_counts[(max_pair[1], byte_tuple[i + 1])] -= count
                        pair_counts[(byte_tuple[i - 1], merged_pair)] += count
                        pair_counts[(byte_tuple[i - 1], max_pair[0])] -= count

    # delete item with merged pair
    del pair_counts[max_pair]

    # delete items with count 0
    cleaned_pair_counts = {pair: count for pair, count in pair_counts.items() if count != 0}

    return dict(cleaned_pair_counts)


def pretokenize_naive(text:str) -> List[Tuple[int]]:
    # pretokenize text and encode each pretoken into sequence of UTF-8 bytes
    # pretokens = re.findall(PAT, text)  # naive implementation gives oom
    pretokens = (match.group() for match in re.finditer(PAT, text))
    return [tuple(bytes((i,)) for i in pretoken.encode("utf-8")) for pretoken in pretokens]


def pretokenize_encode(text: str, special_tokens: List[str] | None = None) -> List[Union[Tuple[int], str]]:
    # case 1. no special tokens
    if special_tokens is None:
        return pretokenize_naive(text)

    # case 2. special tokens
    # define pattern to match special tokens
    special_tokens.sort(key=len, reverse=True)
    pattern = "|".join(re.escape(token) for token in special_tokens)

    # split text on special token regex pattern
    chunks = re.split(f"({pattern})", text)
    chunks = [chunk for chunk in chunks]

    # pretokenize non-special token chunks
    pretokenized_list = []
    for chunk in chunks:
        if chunk in special_tokens:
            pretokenized_list.append(chunk)
        else:
            pretokens_byte = pretokenize_naive(chunk)
            pretokenized_list.extend(pretokens_byte)

    return pretokenized_list


def reverse_vocab_dict(dictionary: Dict[int, bytes]) -> Dict[int, Tuple[int]]:
    # reverse keys and values in dictionary
    reversed_dict = {}
    for key, value in dictionary.items():
        reversed_dict[value] = key

    return reversed_dict


def map_to_int(tokenized_list: List[Union[Tuple[bytes], str]], reversed_vocab: Dict[Tuple[int], int], merges: List[Tuple[bytes, bytes]]) -> List[int]:
    overall_list = []
    # convert all tokens to corresponding integer
    for chunk in tqdm(tokenized_list):
        # case 1. special token
        if isinstance(chunk, str):
            overall_list.append(reversed_vocab[chunk.encode("utf-8")])

        # case 2. regular tokens
        else:
            int_list = [reversed_vocab[byte] for byte in chunk]
            if len(int_list) == 1:
                overall_list.append(int_list[0])
            
            else:
                paired_int_list = list(zip(int_list, int_list[1:]))

                while True:
                    paired_int_list = [tuple(x) for x in paired_int_list]
                    if all(x not in merges for x in paired_int_list):
                        break
                    min_value_pair = min(paired_int_list, key=lambda x: merges.get(x, 1e9))
                    min_value = merges[min_value_pair]
                    while min_value_pair in paired_int_list:
                        min_index = paired_int_list.index(min_value_pair)
                        paired_int_list[min_index] = min_value
                        if len(paired_int_list) == 1:
                            paired_int_list[0] = (min_value,)
                            break
                        else:
                            if min_index == len(paired_int_list) - 1:
                                paired_int_list[min_index - 1] = (paired_int_list[min_index - 1][0], min_value)
                            elif min_index == 0:
                                paired_int_list[min_index + 1] = (min_value, paired_int_list[min_index + 1][1])
                            else:
                                paired_int_list[min_index - 1] = (paired_int_list[min_index - 1][0], min_value)
                                paired_int_list[min_index + 1] = (min_value, paired_int_list[min_index + 1][1])
                        del paired_int_list[min_index]
   
                if len(paired_int_list) <= 1:
                    if paired_int_list == [] or isinstance(paired_int_list, int):
                        result_final = paired_int_list
                    else:
                        result_final = list(paired_int_list[0])
                else:
                    result_final = [tuple[0] for tuple in paired_int_list]
                    result_final.append(paired_int_list[-1][1])
                overall_list.extend(result_final)

    return overall_list

# def map_to_int(tokenized_list: List[Union[Tuple[bytes], str]], reversed_vocab: Dict[bytes, int], merges: Dict[Tuple[int, int], int]) -> List[int]:
#     overall_list = []
    
#     for chunk in tokenized_list:
#         if isinstance(chunk, str):
#             overall_list.append(reversed_vocab[chunk.encode("utf-8")])
#         else:
#             token_ids = [reversed_vocab[byte] for byte in chunk]
            
#             # Merge and replace tokens in order
#             merge_queue = [(merges.get(tuple(sorted(pair)), float('inf')), pair) for pair in zip(token_ids, token_ids[1:])]
#             merge_queue.sort(key=lambda x: x[0])
#             new_token_ids = []
            
#             while merge_queue:
#                 merge_cost, pair = merge_queue[0]
#                 if merge_cost == float('inf'):
#                     new_token_ids.append(pair[0])
#                     merge_queue.pop(0)
#                 else:
#                     new_token = merges[tuple(sorted(pair))]
#                     new_token_ids.append(new_token)
#                     merge_queue.pop(0)
                    
#                     # Check if the new pairs formed can be merged at a lower cost
#                     if len(new_token_ids) > 1:
#                         prev_pair = (new_token_ids[-2], new_token)
#                         prev_pair_cost = merges.get(tuple(sorted(prev_pair)), float('inf'))
#                         if prev_pair_cost < merge_cost:
#                             new_token_ids[-1] = prev_pair_cost
#                             continue
                    
#                     if len(token_ids) > len(new_token_ids) + 1:
#                         next_pair = (new_token, token_ids[len(new_token_ids)])
#                         merge_cost = merges.get(tuple(sorted(next_pair)), float('inf'))
#                         merge_queue.append((merge_cost, next_pair))
#                         merge_queue.sort(key=lambda x: x[0])
                    
#             overall_list.extend(new_token_ids)
    
#     return overall_list


def apply_merges(byte_tuple: Tuple[bytes], byte_pair: Tuple[bytes, bytes]) -> Tuple[bytes]:
     # collect changes where merged pair replaces individual bytes
    merged_pretoken_bytes = []
    skip_next_byte = False  # flag to skip if pair is merged

    for byte1, byte2 in zip(byte_tuple, byte_tuple[1:]):
        if skip_next_byte:
            skip_next_byte = False
            continue  # skip adding the next byte

        if (byte1, byte2) == byte_pair:
            merged_pretoken_bytes.append(byte1 + byte2)
            skip_next_byte = True  # set flag to skip
        else:
            merged_pretoken_bytes.append(byte1)

    # add last byte if it was not part of a merged pair
    if not skip_next_byte:
        merged_pretoken_bytes.append(byte_tuple[-1])

    merged_byte_tuple = tuple(merged_pretoken_bytes)

    return merged_byte_tuple


def merge_from_merges(byte_tuple: Tuple[bytes], merges: List[Tuple[bytes, bytes]]) -> Tuple[bytes]:
    # apply merges to pretokens in the same order of creation
    for merge in merges:
        merged_byte_tuple = apply_merges(byte_tuple, merge)
        byte_tuple = merged_byte_tuple

        if len(byte_tuple) == 1:
            break

    return byte_tuple