import re

URL_FRAGMENT = re.compile(r"\b(?:https?\w*|www\w*)\S*")
SPACES = re.compile(r"\s+")


def light_clean(title):
    title = URL_FRAGMENT.sub(" ", title)
    return SPACES.sub(" ", title).strip()
