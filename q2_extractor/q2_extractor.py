#!/usr/bin/env python3
"""
Main module for command-line interaction with q2-extractor
"""

import argparse

import q2_extractor.Extractor as Extractor
import q2_extractor.PipelineExtractor as PipelineExtractor

def main():
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    sp = parser.add_subparsers()
    art_sp = sp.add_parser('artifact', help='Inspect an artifact file')
    art_sp.add_argument('artifact', metavar='a', help='Artifact file inspect')
    dir_sp = sp.add_parser('directory', help='Inspect a directory to ' \
                                             'reconstruct what was done')
    dir_sp.add_argument('directory', metavar='d', help='Directory to inspect')

    # Required positional argument
#    parser.add_argument("arg", help="Required positional argument")

    # Optional argument flag which defaults to False
#    parser.add_argument("-f", "--flag", action="store_true", default=False)

    # Optional argument which requires a parameter (eg. -d test)
#    parser.add_argument("-n", "--name", action="store", dest="name")

    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, etc)")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    if 'artifact' in args:
        print(Extractor.q2Extractor(args.artifact))
    elif 'directory' in args:
        print(PipelineExtractor.DirectoryInspector(args.directory).print_pipeline())
