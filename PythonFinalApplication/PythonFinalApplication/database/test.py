#! /usr/bin/env python3
import sqlite3


def main(book, chapter, verse):
    connection = sqlite3.connect('pg30.db')
    cursor = connection.cursor()
    cursor.execute('''\
SELECT content
  FROM bible
 WHERE book = ?
   AND chapter = ?
   AND verse = ?''', (book, chapter, verse))
    text = cursor.fetchone()[0]
    input(text)


if __name__ == '__main__':
    main(17, 8, 9)
