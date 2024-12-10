#! /usr/bin/env python3
"""Automate the indexing and processing of the verse library.

These three classes allow a library directory to automatically be
parsed and prepared for use in a categorized reference database."""

__author__ = 'Stephen Paul Chappell'
__date__ = '5 September 2019'
__version__ = '$Revision: 5 $'

import operator

import database


class VerseLibrary:
    """Generate a verse reference database from a directory path.

    Given a directory that has been specially formatted, this class will
    index its files along with one layer of subdirectories. An XHTML index
    suitable for a form is automatically generated from the collected data."""

    TITLE = operator.attrgetter('title')

    def __init__(self, path):
        """Initialize variables from data on path."""
        self.__option = _VerseGroup(path)
        self.__groups = [_VerseGroup(content, content.name)
                         for content in path.iterdir() if content.is_dir()]
        self.__groups.sort(key=self.TITLE)
        self.__cache = self.__cache()

    def __cache(self):
        """Generate an XHTML menu from collected information.

        This method is here to organize the construction process of a
        VerseLibrary instance. A menu is cached and bound to this name."""
        cache = ['<select id="{0}" name="{0}">']
        for gi, group in enumerate(self.__groups):
            prefix = '' if gi else '{1}'
            cache.append(f'{prefix}    <optgroup label="{group.title}">')
            cache.extend(
                f'        <option value="{gi}.{fi}">{file.title}</option>'
                for fi, file in enumerate(group)
            )
            cache.append('    </optgroup>')
        cache.extend(
            f'    <option value="{fi}">{file.title}</option>'
            for fi, file in enumerate(self.__option)
        )
        cache.append('</select>')
        return '\n'.join(cache)

    def html(self, name, default=None):
        """Provide customized HTML code for the library menu.

        After taking the name of this portion of a form and an optional
        prompt, the cached HTML code (or XHTML) is reformatted to include
        the extra information. The results are then returned to the caller."""
        return self.__cache.format(
            name,
            '' if default is None else
            f'    <option selected="selected">{default}</option>\n'
        )

    def __contains__(self, item):
        """Verify if the item is contained in the groups.

        A string is taken in and parsed to see if it could possibly
        refer to a file contained in one of the groups. Various checks
        are run, and the result is returned via the "X in Y" syntax."""
        if item is None or item.count('.') > 1:
            return False
        if '.' in item:
            group, file = item.split('.', 1)
        else:
            group, file = None, item
        if group is None:
            group = self.__option
        else:
            try:
                group = self.__groups[int(group)]
            except (ValueError, IndexError):
                return False
        try:
            return 0 <= int(file) < len(group)
        except ValueError:
            return False

    def __getitem__(self, key):
        """Retrieve a file from one of the contained groups.

        The key is verified in order to make certain assumptions later on.
        If it is valid, the correct file is returned after parsing the key."""
        if key not in self:
            raise KeyError(f'{key!r} not found in library')
        if '.' in key:
            group, file = list(map(int, key.split('.')))
        else:
            group, file = None, int(key)
        if group is None:
            group = self.__option
        else:
            group = self.__groups[group]
        return group[file]


class _VerseGroup:
    """Collect and store a verse file index from a directory.

    Instances of this class are automatically built when they are needed
    by VerseLibrary. This an intermediate storage system for _VerseFiles."""

    def __init__(self, path, title=None):
        """Initialize instance's variables from information in directory."""
        self.__title = title
        self.__files = [_VerseFile(text)
                        for text in path.glob('*.txt') if text.is_file()]
        self.__files.sort(key=VerseLibrary.TITLE)

    def __len__(self):
        """Provide the total number of files this _VerseGroup contains."""
        return len(self.__files)

    def __getitem__(self, key):
        """Index into the files and retrieve the one specified by the key."""
        return self.__files[key]

    def __iter__(self):
        """Generate an iterator over the files found in this _VerseGroup."""
        return iter(self.__files)

    @property
    def title(self):
        """Read-only title property identifying the group's contents."""
        return self.__title


class _VerseFile:
    """Cache the contents of a files along with its name.

    This class is used by _VerseGroup as needed. It stores the contents
    of text files and will automatically parse references when indexing."""

    def __init__(self, path):
        """Initialize using non-empty lines from file on path."""
        self.__title = path.stem
        self.__lines = list(filter(None, map(str.strip, path.open())))

    def __getitem__(self, key):
        """Index into file's lines and get a parsed reference back."""
        if key not in self:
            raise KeyError(f'{type(self).__name__}[{key!r}]')
        line = self.__lines[int(key)]
        return self.__parse(line)

    @staticmethod
    def __parse(line):
        """Split the reference into book, chapter, and verse range."""
        # noinspection PyBroadException,PyPep8
        try:
            book, addr = line.rsplit(' ', 1)
            book = database.BibleServer.BOOKS.index(book) + 1
            if ':' not in addr:
                return book, int(addr), None, None
            chapter, verse = addr.split(':')
            chapter = int(chapter)
            if '-' in verse:
                verse_a, verse_b = list(map(int, verse.split('-')))
            else:
                verse_a = verse_b = int(verse)
            return book, chapter, verse_a, verse_b
        except:
            return None, None, None, None

    def __delitem__(self, key):
        """Delete the verse reference line specified by the key."""
        if key not in self:
            raise KeyError(f'{type(self).__name__}[{key!r}]')
        del self.__lines[int(key)]

    def __iter__(self):
        """Create an iterator over the references in the verse file."""
        return iter(self.__lines)

    def __contains__(self, item):
        """Verify if the given string can refer to a line in the file."""
        try:
            index = int(item)
        except ValueError:
            return False
        else:
            return 0 <= index < len(self.__lines)

    @property
    def title(self):
        """Read-only title property identifying this file's contents."""
        return self.__title
