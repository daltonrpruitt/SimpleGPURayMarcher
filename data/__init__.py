import os


def find(name):
    return os.path.normpath(os.path.abspath(os.path.join(__file__, '..', name)))


def open(name, mode):
    return open(find(name), mode)
