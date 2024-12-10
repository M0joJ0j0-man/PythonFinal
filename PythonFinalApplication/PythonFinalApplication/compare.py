#! /usr/bin/env python3
"""Give a string-oriented API to the generic "diff" module.

The "diff" module is very powerful but practically useless on its own.
The "search" and "empty_master" functions below resolve this problem."""

__author__ = 'Stephen Paul Chappell'
__date__ = '4 September 2019'
__version__ = '$Revision: 5 $'

import operator

import diff


def search(master, slave, *, case_and_punctuation=False):
    """Searches for differences in the master and slave strings.

    The strings are translated into key and data, and their difference
    is calculated. An answer is composed after further processing and
    returned with the number of right words and total number of words."""
    words = master.split()
    key = tuple(words) if case_and_punctuation else _simplify(words, True)
    data = (tuple if case_and_punctuation else _simplify)(slave.split())
    tree = diff.search(key, data)
    if tree.value:
        array = _flatten_tree(_connect_tree(tree))
        # preprocess the array for only the data of interest
        pairs = ((flag, load) for chunk, flag in (
            (chunk, isinstance(chunk[0], int)) for chunk in array if chunk
        ) for load in (chunk[1] if flag else chunk))
        # build an answer based on if it should be CAP sensitive
        answer = ' '.join((
            (load if flag else '_' * len(load))
            for flag, load in pairs
        ) if case_and_punctuation else (
            (word if flag else '_' * len(load))
            for word, (flag, load) in zip(words, pairs)
        ))
    else:
        answer = _default(key)
    return tree.value, len(key), answer


def empty_master(master, *, case_and_punctuation=False):
    """Computes the representation of a master without a slave."""
    words = master.split()
    if not case_and_punctuation:
        words = _simplify(words)
    return _default(words)


def _simplify(words, check_length=False):
    """Removes non-alphabetic characters from an array of words."""
    result = tuple(filter(None, map(lambda word: ''.join(
        filter(str.isalpha, word)), map(str.casefold, words))))
    if check_length and len(words) != len(result):
        raise ValueError('cannot simplify words')
    return result


def _connect_tree(tree, key=operator.attrgetter('value')):
    """Takes the master and finds out what part of the slave matches it.

    The tree from "diff.search" may contain several different routes for
    finding matches. This function takes the best one, gets the master
    match, and fills in the prefix and suffix with the best choices."""
    best_match = max(tree.nodes, key=key)
    node = best_match.a
    if best_match.prefix.value:
        node.prefix = _connect_tree(best_match.prefix)
    if best_match.suffix.value:
        node.suffix = _connect_tree(best_match.suffix)
    return node


def _flatten_tree(node):
    """Flattens a tree from "_connect_tree" for linear iteration.

    The root node created after connecting a tree must be traversed from
    beginning to end in a linear fashion. This function flattens the tree
    to make that possible. Further processing is done by other functions."""
    index, array = 0, []
    append = array.append

    def flatten(current_node):
        """Recursively traverses and flattens the given tree.

        This is a helper function that takes "node" and sequentially processes
        its prefix, root, and suffix. The results are appended to the array."""
        nonlocal index
        prefix = current_node.prefix
        (flatten if isinstance(prefix, diff.Slice) else append)(prefix)
        index += 1
        append((index, current_node.root))
        suffix = current_node.suffix
        (flatten if isinstance(suffix, diff.Slice) else append)(suffix)

    flatten(node)
    return array


def _default(words):
    """Creates a hint string indicating the length of each word."""
    return ' '.join('_' * len(word) for word in words)
