"""
A very naive implementation of parsing type formats from PEP-3118.
"""
from dataclasses import dataclass
from typing import Union, Optional

import pyparsing as pp


c_types = {
    "c": "char",
    "b": "signed char",
    "B": "unsigned char",
    "h": "short",
    "H": "unsigned short",
    "i": "int",
    "I": "unsigned int",
    "l": "long",
    "L": "unsigned long",
    "q": "long long",
    "Q": "unsigned long long",
    "f": "float",
    "d": "double",
    "g": "long double",
    "?": "unsigned char",
    "s": "char",
    "u": "short",
    'w': "int",
}


class Type:
    def format(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class AtomicType(Type):
    typecode: str
    byte_order: str
    z: bool = False

    def format(self) -> str:
        return f"{self.byte_order if self.byte_order != '@' else ''}{self.typecode}"

    @property
    def c(self) -> str:
        if self.z:
            raise ValueError("complex values not supported")
        if self.byte_order != "@":
            raise ValueError(f"non-native byte order '{self.byte_order}' is not supported")
        return c_types[self.typecode]


@dataclass(frozen=True)
class StructField:
    type: Type
    shape: Union[tuple[int, ...], int, None]
    caption: Optional[str] = None

    def format(self) -> str:
        shape = self.shape
        if shape is None:
            shape = ""
        elif len(shape) == 1:
            shape = str(shape[0])
        else:
            shape = "(" + ','.join(map(str, self.shape)) + ")"
        return f"{shape}{self.type.format()}{':' + self.caption + ':' if self.caption is not None else ''}"


@dataclass(frozen=True)
class StructType(Type):
    fields: tuple[StructField, ...]

    def format(self) -> str:
        return "T{" + ''.join(i.format() for i in self.fields) + "}"


p_typecode = pp.Char("xbB?hHiIlLqQnNefdspPuwOgG")
p_byteorder = pp.Char("@=<>!")
p_complex = pp.Literal("Z")
p_atomic_type = (
        pp.Optional(p_byteorder, default="@") +
        pp.Optional(p_complex, default=False).set_parse_action(lambda x: x[0] == 'Z') +
        p_typecode
).set_parse_action(lambda x: AtomicType(typecode=x[2], byte_order=x[0], z=x[1]))
p_struct_type = pp.Forward().set_parse_action(lambda x: StructType(tuple(x)))
p_either_type = (p_atomic_type | p_struct_type).set_parse_action(lambda x: x[0])

p_number = pp.Word(pp.nums).set_parse_action(lambda x: int(x[0]))
p_shape = pp.Suppress("(") + pp.delimited_list(p_number).set_parse_action(lambda x: tuple(x)) + pp.Suppress(")")
p_any_shape = p_number | p_shape
p_caption = pp.Suppress(":") + pp.Word(pp.printables, exclude_chars=":") + pp.Suppress(":")

p_struct_field = (pp.Optional(p_any_shape, default=None) + p_either_type + pp.Optional(p_caption, default=None)).set_parse_action(lambda x: StructField(type=x[1], shape=x[0], caption=x[2]))
p_struct_type << pp.Suppress("T{") + pp.ZeroOrMore(p_struct_field) + pp.Suppress("}")


def parse_3118(s: str) -> Type:
    """
    Parse the PEP-3118 struct format string.

    Parameters
    ----------
    s
        The format string.

    Returns
    -------
    Struct description.
    """
    if not s.startswith("T{"):
        result = p_struct_field.parse_string(s)[0]
        if result.shape is not None and result.shape != 1:
            raise ValueError(f"unsupported struct field {s}")
        return result.type
    return p_struct_type.parse_string(s)[0]
