#! /usr/bin/env python3
"""Run a Bible-verse-matching servlet on the network.

This program is a port of the SwordTest program written
in CPS 110 at BJU during the Autumn Semester of 2003."""

__author__ = 'Stephen Paul Chappell'
__date__ = '5 September 2019'
__version__ = '$Revision: 5 $'

# File: SwordTest.py
# Description: A simple "Sword Test" servlet

import mimetypes
import os
import pathlib
import sys

import database
import html_source
import library
import manager
import servlet
import bible_verse
from state import State, Options


def indent(text, level):
    return '\n'.join(' ' * level + line for line in text.splitlines())


class FileHandler(servlet.HttpServlet):
    """Try to serve file requests from the static path directory.

    The FileHandler class is meant as a patch to HttpServlet to allow
    automatic handling of file requests that the handler might received."""

    block_favicon_request = False
    static_path = pathlib.Path(sys.argv[0]).parent / 'include'

    def service(self, request, response):
        """Check and handle any valid file requests from the client."""
        path = request.path[1:]
        if path and not (request.params or request.query or request.fragment):
            path = self.static_path / path
            if path.is_file():
                kind, encoding = mimetypes.guess_type(str(path))
                response.setContentType(kind or 'application/octet-stream')
                response.setBinaryPayload(path.open('rb'))
                return True
        return False


class SwordTest(FileHandler):
    """Create a handler responsible for the application.

    This is a port of the SwordTest class in the first Java program.
    The service method gets called each time a HTTP request is made."""

    __status = None     # Create a default value.
    __init = False      # Tracks if SwordTest was initialized.

    @classmethod
    def init(cls, lib_path, db_path):
        """Initialize static variables so this class can be used.

        The session manager cleans memory of old sessions not in use.
        The library keeps Bible references and generates related HTML.
        The Bible server responds to verse queries with Verse objects."""
        assert not cls.__init, 'SwordTest is already initialized!'
        # Session Manager closes old sessions each hour.
        cls.SESSION_MANAGER = manager.SessionManager(3600)
        cls.SESSION_MANAGER.daemon = True
        cls.SESSION_MANAGER.start()
        cls.LIBRARY = library.VerseLibrary(lib_path)
        cls.BIBLE_SERVER = database.BibleServer(db_path)
        cls.__init = True
        # "verse.Verse.init_manager" should be called
        # somewhere to ensure that verse checks are killed.
        bible_verse.Verse.init_manager(5)  # Runs every 5 seconds.

    def service(self, request, response):
        """Handle GET and POST requests from client's browser."""
        assert self.__init, 'SwordTest was never initialized!'
        # If the parent handler cannot service the request ...
        if not super().service(request, response):
            state = self.get_state()
            # Handle action desired by the client.
            action = request.getParameter('action')
            state = self.exe_action(action, state, request)
            # Render HTML specified by current state.
            response.setContentType('text/html')
            response.getWriter().print(self.render_html(state))

    def get_state(self):
        """Get state of client's specific application instance.

        The client's IP address is used for identity purposes.
        If there is a session associated with the client, the
        session's state is returned for further processing.
        Otherwise, a new session is created with a new state
        object being added to it, and the new state is returned."""
        ip = self.client_address[0]
        with self.SESSION_MANAGER:
            if ip in self.SESSION_MANAGER:
                session = self.SESSION_MANAGER[ip]
            else:
                # Sessions may live for up to 24 hours.
                session = manager.Session(60 * 60 * 24)
                session.state = State(session, self.LIBRARY, self.BIBLE_SERVER)
                session.ip = ip
                self.SESSION_MANAGER[ip] = session
        return session.state

    def exe_action(self, action, state, request):
        """Execute the action specified by the caller.

        This application recognizes several actions that the
        client may freely attempt to invoke. If the action is
        recognized, relevant methods are called on the state
        object with needed parameter being queried as needed."""
        if action == 'Go Back':
            state.go_back()
        elif action == 'Reset Session':
            # This session is being deleted and a new one made.
            state.reset_session()
            state = self.get_state()
        elif action == 'Choose Quiz':
            # Form created by LIBRARY should be called "quiz."
            state.load_quiz(request.getParameter('quiz'))
        elif action == 'pick_verse':
            # Query: "action=pick_verse&id=X" where X is a number.
            state.pick_verse(request.getParameter('id'))
        elif action == 'Check Your Answer':
            state.check_text([request.getParameter(f'verse{verse_id}')
                              for verse_id in range(state.verse_total)])
            self.__status = 0
        elif action == 'check_status':
            self.__status = state.check_status()
        return state

    def render_html(self, state):
        """Render the XHTML of the current state.

        Depending on the state of a client's session,
        one of several different pages may be sent to
        the browser. The correct page is selected and
        rendered here along with XHTML code that may
        need to be dynamically generated on the fly."""
        if state.current is Options.GET_QUIZ:
            select = indent(self.LIBRARY.html('quiz', 'Options:'), 16)
            get_quiz = html_source.GET_QUIZ.format(select)
            template = html_source.TEMPLATE.format('', get_quiz)
        elif state.current is Options.GET_VERSE:
            title, menu = self.render_menu(state)
            get_verse = html_source.GET_VERSE.format(title, menu)
            template = html_source.TEMPLATE.format('', get_verse)
        elif state.current is Options.TEACH:
            area = self.render_area(state)
            teach = html_source.TEACH.format(area)
            template = html_source.TEMPLATE.format('', teach)
        elif state.current is Options.CHECK:
            check = html_source.CHECK.format(
                (self.__status if self.__status else 'No'),
                (' has' if self.__status == 1 else 's have')
            )
            template = html_source.TEMPLATE.format(html_source.REFRESH, check)
        else:
            raise ValueError(f'{state.current!r} is not a valid state')
        return template

    @staticmethod
    def render_menu(state):
        """Create a Bible verse selection menu.

        The second page of the application (GET_VERSE)
        displays a menu for selecting verses from the
        currently selected quiz set. The XHTML code for
        that menu is dynamically generated in this method."""
        file = state.verse_file
        ul = ['<ul class="hug">']
        li = '<li>\n    <a href="?action=pick_verse&id={}">{}</a>\n</li>'
        ul.extend(indent(li.format(*args), 4) for args in enumerate(file))
        ul.append('</ul>')
        return file.title, indent('\n'.join(ul), 16)

    def render_area(self, state):
        """Create code for verse entry boxes.

        The third page of the application (called TEACH)
        requires textarea XHTML code for individual verses
        to be entered in. The relevant fieldset, legend, and
        status codes are added to the template and returned."""
        return '\n'.join(html_source.VERSE.format(
            verse_obj.addr,
            f'verse{index}',
            *self.render_status(verse_obj)
        ) for index, verse_obj in enumerate(state.verse_list))

    @staticmethod
    def render_status(verse_obj):
        """Compute the status of verse in question.

        Each verse has an optional status to show on the
        TEACH page. This method figures out what to show
        (if anything) and decides what the verse hint is."""
        if not verse_obj.show_hint:
            return '', ''
        if verse_obj.ready is True:
            right, total, text = verse_obj.value
            if right == total:
                color = '#0A0'
                status = 'You know this verse perfectly!'
            elif right > 0:
                color = '#00A'
                status = f'You already know {right / total:.0%} of this verse.'
            else:
                color = '#00A'
                status = 'Here is a hint for this verse.'
        else:
            text = verse_obj.hint
            color = '#A00'
            status = 'Your answer took too long to check!'
        html = f'''
                <span style="color: {color};">{status}</span><br />'''
        return html, text


def main():
    """Initialize program variables and start the server."""
    # Initialize verse database and library.
    os.chdir(pathlib.Path(sys.argv[0]).parent)
    SwordTest.init(pathlib.Path('quizzes'), r'database\pg30.db')
    # Start servlet with debugging enabled.
    servlet.HttpServlet.debug(True)
    servlet.HttpServer.main(SwordTest, 8080)


if __name__ == '__main__':
    main()
