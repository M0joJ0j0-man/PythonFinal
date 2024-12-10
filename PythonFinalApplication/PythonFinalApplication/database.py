#! /usr/bin/env python3
"""Serve verses from the Bible in response to SQL queries.

Pulling Bible verses out of a database allows query details to be
abstracted away and powerful Verse objects returned to the caller."""

__author__ = 'Stephen Paul Chappell'
__date__ = '5 September 2019'
__version__ = '$Revision: 5 $'

import threading
import queue
import sqlite3
import sys

import async_exc
import bible_verse


class BibleServer:
    """Execute a protected SQLite3 database on a singular thread.

    Since a SQLite3 database can only accept queries on the thread that it
    was created on, this server receives requests through a queue and sends
    back the result through a list and mutex mechanism. The verses returned
    from queries are automatically wrapped in their own Verse objects."""

    BOOKS = ('Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
             'Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings',
             '2 Kings', '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah',
             'Esther', 'Job', 'Psalm', 'Proverbs', 'Ecclesiastes',
             'Song of Solomon', 'Isaiah', 'Jeremiah', 'Lamentations',
             'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah',
             'Jonah', 'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai',
             'Zechariah', 'Malachi', 'Matthew', 'Mark', 'Luke', 'John',
             'Acts', 'Romans', '1 Corinthians', '2 Corinthians', 'Galatians',
             'Ephesians', 'Philippians', 'Colossians', '1 Thessalonians',
             '2 Thessalonians', '1 Timothy', '2 Timothy', 'Titus', 'Philemon',
             'Hebrews', 'James', '1 Peter', '2 Peter', '1 John', '2 John',
             '3 John', 'Jude', 'Revelation')

    def __init__(self, *args):
        """Initialize the BibleServer with a SQLite3 database thread."""
        self.__ready = threading.Event()
        self.__thread = async_exc.Thread(target=self.__serve, args=args)
        self.__thread.start()
        self.__ready.wait()
        del self.__ready
        if self.__error is not None:
            raise self.__error
        del self.__error

    def __serve(self, *args):
        """Run a server continuously to answer SQL queries.

        A SQLite3 connection is made in this thread with errors being raised
        again for the instantiating caller. If the connection was made
        successfully, then the server goes into a continuous loop, processing
        SQL queries."""
        database = None
        # noinspection PyBroadException,PyPep8
        try:
            database = sqlite3.connect(*args)
        except:
            self.__error = error = sys.exc_info()[1]
        else:
            self.__error = error = None
        self.__ready.set()
        if error is None:
            self.__queue = queue.Queue()
            while True:
                ready, one, sql, parameters, ret = self.__queue.get()
                # noinspection PyBroadException,PyPep8
                try:
                    cursor = database.cursor()
                    cursor.execute(sql, parameters)
                    data = cursor.fetchone() if one else cursor.fetchall()
                    ret[:] = True, data
                except:
                    ret[:] = False, sys.exc_info()[1]
                ready.set()

    def fetch_chapter(self, book, chapter):
        """Fetch all verses from chapter and wrap in Verse objects."""
        rows = self.__fetch(False, '''\
SELECT verse,
       content
  FROM bible
 WHERE book = ?
   AND chapter = ?
 ORDER BY verse ASC''', book, chapter)
        return self.__verses(book, chapter, rows)

    def fetch_verse(self, book, chapter, verse):
        """Fetch one verse as specified and wrap in a Verse object."""
        row = self.__fetch(True, '''\
SELECT content
  FROM bible
 WHERE book = ?
   AND chapter = ?
   AND verse = ?''', book, chapter, verse)
        if row is not None:
            reference = self.__addr(book, chapter, verse)
            return [bible_verse.Verse(reference, row[0])]

    def fetch_range(self, book, chapter, verse_a, verse_b):
        """Fetch all verses in the range and wrap in Verse objects."""
        rows = self.__fetch(False, '''\
SELECT verse,
       content
  FROM bible
 WHERE book = ?
   AND chapter = ?
   AND verse BETWEEN ? AND ?
 ORDER BY verse ASC''', book, chapter, verse_a, verse_b)
        return self.__verses(book, chapter, rows)

    def __fetch(self, one, sql, *parameters):
        """Execute the specified SQL query and return the results.

        This is a powerful shortcut method that is the closest connection
        other threads will have with the SQL server. The parameters for the
        query are dumped into a queue, and the answer is retrieved when it
        becomes available. This prevents SQLite3 from throwing exceptions."""
        ready, ret = threading.Event(), []
        self.__queue.put((ready, one, sql, parameters, ret))
        ready.wait()
        valid, value = ret
        if valid:
            return value
        raise value

    def __verses(self, book, chapter, rows):
        """Wrap the text from the rows in Verse objects.

        If the query matched any verses, the verse reference is deduced and
        used with the verse text to construct an array of Verse objects."""
        if rows:
            return [bible_verse.Verse(self.__addr(book, chapter, verse), text)
                    for verse, text in rows]

    def __addr(self, book, chapter, verse):
        """Construct a verse reference from the final three parameters."""
        return f'{self.BOOKS[book - 1]} {chapter}:{verse}'

    def __del__(self):
        """Terminates the internal thread when the database is deleted."""
        if self.__thread.is_alive():
            self.__thread.exit()
