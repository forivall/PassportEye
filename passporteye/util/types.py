
import types
from typing import Iterable, Sized, Container

# numpy 1.20 will have types, but i'm just doing this for now to help me annotate things in vscode
ndarray = types.new_class('ndarray', (Iterable, Sized, Container))
