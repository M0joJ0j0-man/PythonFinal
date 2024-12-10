#! /usr/bin/env python3
import mmap
import pathlib
import sqlite3


def main():
    bible = parse_bible(pathlib.Path('pg30.txt'))
    # Create the database being built.
    connection = sqlite3.connect('pg30.db')
    cursor = connection.cursor()
    cursor.execute('''\
CREATE TABLE bible (
  book    INTEGER,
  chapter INTEGER,
  verse   INTEGER,
  content TEXT
)''')
    # Add all verses to database.
    for book_i, book in enumerate(bible):
        for chapter_i, chapter in enumerate(book):
            for verse_i, verse in enumerate(chapter):
                cursor.execute('''\
INSERT INTO bible (
  book,
  chapter,
  verse,
  content
) VALUES (?, ?, ?, ?)''',
                               (book_i + 1, chapter_i + 1, verse_i + 1, verse))
    # Commit changes and close database.
    connection.commit()
    connection.close()


def parse_bible(path):
    """Take a specially formatted file and extract the Bible's structure."""
    book = chapter = verse = 1
    book_list, chapter_list, verse_list = [], [], []
    start = 0
    with path.open('rb') as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as bible:
            for next_line in b'\r\n', b'\r', b'\n':
                if bible.find(next_line) >= 0:
                    break
            else:
                raise EOFError('could not find any line delimiters')
            while True:
                sub = f'{book:02}:{chapter:03}:{verse:03} '.encode()
                index = bible.find(sub, start)
                if index >= 0:
                    start = index + len(sub)
                    end = bible.find(next_line * 2, start)
                    if end < 0:
                        raise EOFError('could not find the end of the verse')
                    bible.seek(start)
                    verse_text = bible.read(end - start).decode()
                    verse_list.append(' '.join(verse_text.split()))
                    start = end
                    verse += 1
                elif verse != 1:
                    chapter_list.append(tuple(verse_list))
                    verse_list.clear()
                    chapter += 1
                    verse = 1
                elif chapter != 1:
                    book_list.append(tuple(chapter_list))
                    chapter_list.clear()
                    book += 1
                    chapter = 1
                elif book != 1:
                    return tuple(book_list)
                else:
                    raise EOFError('could not find any of the expected data')


if __name__ == '__main__':
    main()
