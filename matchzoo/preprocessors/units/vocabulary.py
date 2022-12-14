from .stateful_unit import StatefulUnit


class Vocabulary(StatefulUnit):
    """
    Vocabulary class.

    :param pad_value: The string value for the padding position.
    :param oov_value: The string value for the out-of-vocabulary terms.

    Examples:
        >>> vocab = Vocabulary(pad_value='[PAD]', oov_value='[OOV]')
        >>> vocab.fit(['A', 'B', 'C', 'D', 'E'])
        >>> term_index = vocab.state['term_index']
        >>> term_index  # doctest: +SKIP
        {'[PAD]': 0, '[OOV]': 1, 'D': 2, 'A': 3, 'B': 4, 'C': 5, 'E': 6}
        >>> index_term = vocab.state['index_term']
        >>> index_term  # doctest: +SKIP
        {0: '[PAD]', 1: '[OOV]', 2: 'D', 3: 'A', 4: 'B', 5: 'C', 6: 'E'}

        >>> term_index['out-of-vocabulary-term']
        1
        >>> index_term[0]
        '[PAD]'
        >>> index_term[42]
        Traceback (most recent call last):
            ...
        KeyError: 42
        >>> a_index = term_index['A']
        >>> c_index = term_index['C']
        >>> vocab.transform(['C', 'A', 'C']) == [c_index, a_index, c_index]
        True
        >>> vocab.transform(['C', 'A', '[OOV]']) == [c_index, a_index, 1]
        True
        >>> indices = vocab.transform(list('ABCDDZZZ'))
        >>> ' '.join(vocab.state['index_term'][i] for i in indices)
        'A B C D D [OOV] [OOV] [OOV]'

    """

    def __init__(self, pad_value: str = '<PAD>', oov_value: str = '<OOV>'):
        """Vocabulary unit initializer."""
        super().__init__()
        self._pad = pad_value
        self._oov = oov_value
        self._context['term_index'] = self.TermIndex()
        self._context['index_term'] = dict()
        self._added = False

    class TermIndex(dict):
        """Map term to index."""

        def __missing__(self, key):
            """Map out-of-vocabulary terms to index 1."""
            return 1

    def fit(self, tokens: list):
        """Build a :class:`TermIndex` and a :class:`IndexTerm`."""
        if not self._added:
            self._added = True
            self._context['term_index'][self._pad] = 0
            self._context['term_index'][self._oov] = 1
            self._context['index_term'][0] = self._pad
            self._context['index_term'][1] = self._oov

            terms = sorted(set(tokens))
            for index, term in enumerate(terms, 2):
                self._context['term_index'][term] = index
                self._context['index_term'][index] = term
        else:
            terms = sorted(set(tokens))
            offset = len(self.state['index_term'])

            for index, term in enumerate(terms, offset):
                self._context['term_index'][term] = index
                self._context['index_term'][index] = term

    def transform(self, input_: list) -> list:
        """Transform a list of tokens to corresponding indices."""
        return [self._context['term_index'][token] for token in input_]

    def detransform(self, input_: list) -> list:
        """DeTransform a list of indices to corresponding tokens."""
        return [self._context['index_term'][index] for index in input_]

    @property
    def v2i(self) -> TermIndex:
        return self._context['term_index']

    @property
    def i2v(self) -> TermIndex:
        return self._context['index_term']

    @property
    def pad_index(self):
        return self.v2i[self._pad]

    @property
    def oov_index(self):
        return self.v2i[self._oov]

    def __len__(self):
        return len(self.v2i)
