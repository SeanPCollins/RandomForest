#!/usr/bin/env python

"""
configuration for Health Canada jobs

Provides the Options class that will transparently handle the different option
sources through the .get() method. Pulls in defaults, site and job options plus
command line customisation. Instantiating Options will set up the logging for
the particular job.

"""
import os
import sys
import re
import copy
import functools
import logging
from logging import debug, error, info
from optparse import OptionParser
import textwrap
from io import StringIO

# Third-party libraries
import configparser

import __main__

CRITICAL_COLOR = '\033[30;41m'
ERROR_COLOR = '\033[30;41m'
WARNING_COLOR = '\033[30;43m'
DEBUG_COLOR = '\033[30;46m'
INFO_COLOR = '\033[30;42m'
NORMAL_COLOR = '\033[0m'


class Options(object):
    """
    Transparent options handling.

    A single unified way of dealing with input files and command line options
    delivering sensible defaults for unspecified values. Access options with
    the .get() method, or the method that specifies the expected type. It is
    recommended to replace with a new instance each time the script is run,
    otherwise commandline options or changed input files will not be picked up.

    """
    __all__ = ['Options']
    OPTION_SOURCES = {
    'A': ('Instance attributes', lambda self, item: object.__getattribute__(self, item)),
    'C': ('Commandline options', lambda self, item: self.options.__dict__.get(item)),
    'O': ('Custom -o options', lambda self, item: self.cmdopts.get(item)),
    'F': ('Jobname.hc per-job settings', lambda self, item: self.job_ini.get('job_config', item)),
    'S': ('Site_config.ini settings', lambda self, item: self.site_ini.get('site_config', item)),
    'D': ('Defaults', lambda self, item: self.defaults.get('defaults', item))
    }

    def __init__(self, job_name=None, code=None):
        """Initialize options from all .ini files and the commandline."""
        # use .get{type}() to read attributes, only access args directly
        initial_attributes = {
            'job_dir': '',
            'script_dir': '',
            'job_name': job_name,
            'code': code,
            'args': [],
            'options': {},
            'cmdopts': {},
            '_used_options': set(),
            'defaults': configparser.SafeConfigParser(),
            'site_ini': configparser.SafeConfigParser(),
            'job_ini': configparser.SafeConfigParser()
        }

        # Use a loop to set the attributes based on the dictionary
        for attr, value in initial_attributes.items():
            setattr(self, attr, value)

        # populate options
        self._init_paths()
        self.commandline()
        self._init_logging()
        self.load_defaults()
        self.load_site_defaults()
        self.load_job_defaults()

    def get(self, item):
        """Map values from different sources based on priorities."""
        # Define a default value in case the option is not found
        default_value = None
        # Iterate through the sources in the specified order
        for source, source_func_info in self.OPTION_SOURCES.items():
            source_func = source_func_info[1]
            try:
                value = source_func(self, item)
                # If a value is found in the source, return it, if not None
                if value is not None:
                    return value
            except (AttributeError,  configparser.NoOptionError) as e:
                # Option not found in this source, continue to the next
                continue
    
        # If the option is not found in any source, return the default value
        return default_value

    def getbool(self, item):
        """Parse option and return a boolean value."""
        value = self.get(item)
        if isinstance(value, bool):
            return value
        elif hasattr(value, 'lower'):
            return value.lower() in ["1", "yes", "true", "on"]
        elif hasattr(value, 'lower'):
            return value.lower() not in ["0", "no", "false", "off"]
        else:
            return bool(item)

    def getint(self, item):
        """Return item's value as an integer."""
        return int(self.get(item))

    def getfloat(self, item):
        """Return item's value as a float."""
        return float(self.get(item))

    def gettuple(self, item, dtype=None):
        """Return item's value interpreted as a tuple of 'dtype' [strings]."""
        value = self.get(item)
        value = [x for x in re.split('[\s,\(\)\[\]]+', value) if x]
        if dtype is not None:
            return tuple([dtype(x) for x in value])
        else:
            return tuple(value)

    def _init_paths(self):
        """Find the script directory and set up working directory"""
        # Where the script is has the config defaults.
        if __name__ != '__main__':
            self.script_dir = os.path.dirname(__file__)
        else:
            self.script_dir = os.path.abspath(sys.path[0])
        # Where we run the job.
        self.job_dir = os.getcwd()

    def _init_logging(self):
        """
        Setup the logging to terminal and .flog file, with levels as required.
        Must run before any logging calls so we need to access attributes
        rather than using self.get()!
    
        """
        # Define log level mappings
        log_levels = {
            'silent': (logging.CRITICAL, logging.INFO),
            'quiet': (logging.ERROR, logging.INFO),
            'verbose': (logging.DEBUG, logging.DEBUG),
            'default': (logging.INFO, logging.INFO)
        }
    
        # Determine log levels based on options
        if self.options.silent:
            stdout_level, file_level = log_levels['silent']
        elif self.options.quiet:
            stdout_level, file_level = log_levels['quiet']
        elif self.options.verbose:
            stdout_level, file_level = log_levels['verbose']
        else:
            stdout_level, file_level = log_levels['default']
    
        # Configure file logging
        logging.basicConfig(level=file_level,
                            format='[%(asctime)s] %(levelname)s %(message)s',
                            datefmt='%Y%m%d %H:%M:%S',
                            filename=self.job_name + '.flog',
                            filemode='a')
    
        # Define custom log level names
        log_level_names = {10: '--', 20: '>>', 30: '**', 40: '!!', 50: 'XX'}
        for level, name in log_level_names.items():
            logging.addLevelName(level, name)
    
        # Create a console handler for colored output or plain output
        console = (logging.StreamHandler(sys.stdout) if self.options.plain
                   else ColouredConsoleHandler(sys.stdout))
        console.setLevel(stdout_level)
    
        # Set the formatter for console output
        formatter = logging.Formatter('%(levelname)s %(message)s')
        console.setFormatter(formatter)
    
        # Add the console handler to the root logger
        logging.getLogger('').addHandler(console)

    def commandline(self):
        """Specified options, highest priority."""
        usage = "usage: %prog [options] [COMMAND] JOB_NAME"
        # use description for the script, not for this module
        parser = OptionParser(usage=usage, version="%prog 0.1", description=__main__.__doc__)
        
        options = [
            ("-v", "--verbose", "output extra debugging information"),
            ("-q", "--quiet", "only output warnings and errors"),
            ("-s", "--silent", "no terminal output"),
            ("-p", "--plain", "do not colorize or wrap output"),
            ("-i", "--interactive", "enter interactive mode"),
            ("-m", "--import", "try and import old data"),
            ("-n", "--no-submit", "create input files only, do not run any jobs"),
        ]
    
        for short_opt, long_opt, help_text in options:
            parser.add_option(short_opt, long_opt, action="store_true", dest=long_opt.lstrip('-'), help=help_text)
        parser.add_option("-o", "--option", action="append", dest="option", default=[], metavar="key=value",
                  help="set custom options as key=value pairs")
    
        (local_options, local_args) = parser.parse_args()
    
        # job_name may or may not be passed or set initially
        if self.job_name:
            if self.job_name in local_args:
                local_args.remove(self.job_name)
        elif len(local_args) == 0:
            parser.error("No arguments given (try %prog --help)")
        else:
            # Take the last argument as the job name
            self.job_name = local_args.pop()
    
        # key-value options from the command line
        if local_options.option is not None:
            for pair in local_options.option:
                if '=' in pair:
                    pair = pair.split('=', 1)  # maximum of one split
                    self.cmdopts[pair[0]] = pair[1]
                else:
                    self.cmdopts[pair] = True
        self.options = local_options
        # Args are only the COMMANDS for the run
        self.args = [arg.lower() for arg in local_args]

    def load_config_file(self, file_path, section_name):
        """
        Load configuration from a file and section, handling defaults.

        :param file_path: The path to the configuration file.
        :param section_name: The name of the section in the configuration file.
        """
        try:
            with open(file_path, 'r') as filetemp:
                config_data = filetemp.read()
                if not f'[{section_name}]' in config_data.lower():
                    config_data = f'[{section_name}]\n' + config_data
                config_data = StringIO(config_data)
        except IOError:
            # Handle the case when the file does not exist
            if section_name == 'defaults':
                logging.debug('Default options not found! Something is very wrong.')
            else:
                logging.debug(f"No {section_name} options found; using defaults")
            config_data = StringIO(f'[{section_name}]\n')

        if section_name == 'defaults':
            self.defaults.readfp(config_data)
        elif section_name == 'site_config':
            self.site_ini.readfp(config_data)
        elif section_name == 'job_config':
            self.job_ini.readfp(config_data)

    def load_defaults(self):
        """Load program defaults."""
        home_dir = os.path.expanduser('~')
        default_ini_path = os.path.join(f'{home_dir}/.defaults/', f'{self.code}_defaults.ini')
        self.load_config_file(default_ini_path, 'defaults')

    def load_site_defaults(self):
        """Find where the script is and load defaults"""
        home_dir = os.path.expanduser('~')
        site_ini_path = os.path.join(f'{home_dir}/.defaults/', f'{self.code}_site.ini')
        self.load_config_file(site_ini_path, 'site_config')

    def load_job_defaults(self):
        """Find where the job is running and load defaults"""
        job_ini_path = os.path.join(self.job_dir, f'{self.job_name}.hc')
        self.load_config_file(job_ini_path, 'job_config')

def options_test():
    """Try and read a few options from different sources."""
    testopts = Options()
    option_names = ['job_name', 'cmdopts', 'args', 'verbose', 'script_dir', 'interactive']

    for option_name in option_names:
        option_value = testopts.get(option_name)
        print(f"{option_name}: {option_value}")

        if option_name == 'args':
            for arg in option_value:
                print(f'{arg}: {option_value}')
                try:
                    print(f"{arg} as bool: {testopts.getbool(arg)}")
                except ValueError:
                    print(f"{arg} is not a bool")
                try:
                    print(f"{arg} as int: {testopts.getint(arg)}")
                except ValueError:
                    print(f"{arg} is not an int")
                try:
                    print(f"{arg} as float: {testopts.getfloat(arg)}")
                except ValueError:
                    print(f"{arg} is not a float")
                try:
                    print(f"{arg} as tuple: {testopts.gettuple(arg)}")
                except ValueError:
                    print(f"{arg} is not a tuple")

    print(testopts.get('not an option'))


class ColouredConsoleHandler(logging.StreamHandler):
    """Makes colorized and wrapped output for the console."""
    COLOR_MAP = {
        50: CRITICAL_COLOR,  # CRITICAL / FATAL
        40: ERROR_COLOR,     # ERROR
        30: WARNING_COLOR,   # WARNING
        20: INFO_COLOR,      # INFO
        10: DEBUG_COLOR,     # DEBUG
    }

    def emit(self, record):
        """Colorize and emit a record."""
        # Make a copy of the record to prevent altering the message for other loggers
        myrecord = copy.copy(record)
        levelno = myrecord.levelno
        front = self.COLOR_MAP.get(levelno, NORMAL_COLOR)
        myrecord.levelname = f'{front}{myrecord.levelname}{NORMAL_COLOR}'
        logging.StreamHandler.emit(self, myrecord)

class NullConfigParser(configparser.SafeConfigParser):
    """Use in place of a blank ConfigParser that has no options."""
    
    def __init__(self):
        super().__init__()

    def has_option(self, section, option):
        """Always return False as there are no options."""
        return False

if __name__ == '__main__':
    options_test()
