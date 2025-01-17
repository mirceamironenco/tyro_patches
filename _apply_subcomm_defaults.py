from __future__ import annotations

import dataclasses
import pathlib
import sys
import warnings
from collections import deque
from collections.abc import Sequence
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import shtab
import tyro.conf as conf
from tyro import _argparse as argparse
from tyro import (
    _argparse_formatter,
    _arguments,
    _calling,
    _docstrings,
    _fields,
    _resolver,
    _singleton,
    _strings,
    _unsafe_cache,
)
from tyro._parsers import (
    ParserSpecification,
    SubparsersSpecification,
    add_subparsers_to_leaves,
    handle_field,
)
from tyro._typing import TypeForm
from tyro.conf import _markers
from tyro.constructors._primitive_spec import UnsupportedTypeAnnotationError
from tyro.constructors._struct_spec import UnsupportedStructTypeMessage


class DSU(_singleton.Singleton):
    _parent: dict[str, str]
    _size: dict[str, int]

    def init(self, *args, **kwds) -> None:
        self._parent = {}
        self._size = {}
        self._components = 0

    def is_node(self, node: str) -> bool:
        return node in self._parent

    def make(self, node: str) -> bool:
        if not self.is_node(node):
            self._parent[node] = node
            self._size[node] = 1
            self._components += 1

            return True

        return False

    def find(self, node: str) -> str:
        if not self.is_node(node):
            self.make(node)
            return node

        while self._parent[node] != node:
            self._parent[node] = self._parent[self._parent[node]]
            node = self._parent[node]

        return self._parent[node]

    def union(self, node_x: str, node_y: str) -> bool:
        x, y = self.find(node_x), self.find(node_y)

        if x == y:
            return False

        if self._size[x] < self._size[y]:
            x, y = y, x

        self._parent[y] = x
        self._size[x] += self._size[y]
        self._components -= 1
        return True

    def connected(self, node_x: str, node_y: str) -> bool:
        return self.find(node_x) == self.find(node_y)

    def num_components(self) -> int:
        return self._components

    def component_size(self, node_x: str) -> int:
        return self._size[self.find(node_x)]


T = TypeVar("T")


class _Node:
    DEFAULT_PARSER_NAME = "<#_#>"
    default_name: str
    edges: list[_Node]
    is_subparser: bool

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        is_subparser: bool = False,
    ) -> None:
        if is_subparser:
            if name is None:
                raise ValueError("Subparsers must have default name provided.")
        self.default_name = name if name else _Node.DEFAULT_PARSER_NAME
        self.edges = []
        self.is_subparser = is_subparser

    def add_edge(self, other: _Node) -> None:
        self.edges.append(other)


@dataclasses.dataclass(frozen=True)
class _ParserSpecification(ParserSpecification):
    _node: _Node


# Patch ParserSpecification.from_callable to retain default subcommand names
def from_callable_or_type(
    f: Callable[..., T],
    markers: Set[_markers._Marker],
    description: str | None,
    parent_classes: Set[Type[Any]],
    default_instance: Union[
        T, _singleton.PropagatingMissingType, _singleton.NonpropagatingMissingType
    ],
    intern_prefix: str,
    extern_prefix: str,
    subcommand_prefix: str = "",
    support_single_arg_types: bool = False,
) -> ParserSpecification:
    """Create a parser definition from a callable or type."""

    # Consolidate subcommand types.
    markers = markers | set(_resolver.unwrap_annotated(f, _markers._Marker)[1])
    consolidate_subcommand_args = _markers.ConsolidateSubcommandArgs in markers

    # Cycle detection.
    #
    # 'parent' here refers to in the nesting hierarchy, not the superclass.
    if f in parent_classes and f is not dict:
        raise UnsupportedTypeAnnotationError(
            f"Found a cyclic dependency with type {f}."
        )

    # TODO: we are abusing the (minor) distinctions between types, classes, and
    # callables throughout the code. This is mostly for legacy reasons, could be
    # cleaned up.
    parent_classes = parent_classes | {cast(Type, f)}

    # Resolve the type of `f`, generate a field list.
    with _fields.FieldDefinition.marker_context(tuple(markers)):
        out = _fields.field_list_from_type_or_callable(
            f=f,
            default_instance=default_instance,
            support_single_arg_types=support_single_arg_types,
        )
        assert not isinstance(out, UnsupportedStructTypeMessage)
        f, field_list = out

    has_required_args = False
    args = []
    helptext_from_intern_prefixed_field_name: Dict[str, str | None] = {}

    child_from_prefix: Dict[str, ParserSpecification] = {}

    subparsers = None
    subparsers_from_prefix = {}

    ### Patch
    node = _Node()
    subcommand_dsu = DSU()
    ### Patch

    for field in field_list:
        field_out = handle_field(
            field,
            parent_classes=parent_classes,
            intern_prefix=intern_prefix,
            extern_prefix=extern_prefix,
            subcommand_prefix=subcommand_prefix,
        )
        if isinstance(field_out, _arguments.ArgumentDefinition):
            # Handle single arguments.
            args.append(field_out)
            if field_out.lowered.required:
                has_required_args = True
            continue

        if isinstance(field_out, SubparsersSpecification):
            # Handle subparsers.
            subparsers_from_prefix[field_out.intern_prefix] = field_out
            subparsers = add_subparsers_to_leaves(subparsers, field_out)

            ### Patch
            if field_out.default_name is not None:
                assert field_out.default_name in field_out.parser_from_name

                subcommand_dsu.make(field_out.default_name)

                subparser_node = _Node(name=field_out.default_name, is_subparser=True)
                node.add_edge(subparser_node)

                for choice, choice_parser in field_out.parser_from_name.items():
                    subcommand_dsu.union(field_out.default_name, choice)
                    if isinstance(choice_parser, _ParserSpecification):
                        choice_parser._node.default_name = choice
                        subparser_node.add_edge(choice_parser._node)
            ### Patch

        elif isinstance(field_out, ParserSpecification):
            # Handle nested parsers.
            nested_parser = field_out
            child_from_prefix[field_out.intern_prefix] = nested_parser

            if nested_parser.has_required_args:
                has_required_args = True

            # Include nested subparsers.
            if nested_parser.subparsers is not None:
                subparsers_from_prefix.update(
                    nested_parser.subparsers_from_intern_prefix
                )
                subparsers = add_subparsers_to_leaves(
                    subparsers, nested_parser.subparsers
                )

            # Helptext for this field; used as description for grouping arguments.
            class_field_name = _strings.make_field_name(
                [intern_prefix, field.intern_name]
            )
            if field.helptext is not None:
                helptext_from_intern_prefixed_field_name[class_field_name] = (
                    field.helptext
                )
            else:
                helptext_from_intern_prefixed_field_name[class_field_name] = (
                    _docstrings.get_callable_description(nested_parser.f)
                )

            # If arguments are in an optional group, it indicates that the default_instance
            # will be used if none of the arguments are passed in.
            if (
                len(nested_parser.args) >= 1
                and _markers._OPTIONAL_GROUP in nested_parser.args[0].field.markers
            ):
                current_helptext = helptext_from_intern_prefixed_field_name[
                    class_field_name
                ]
                helptext_from_intern_prefixed_field_name[class_field_name] = (
                    ("" if current_helptext is None else current_helptext + "\n\n")
                    + "Default: "
                    + str(field.default)
                )

            ### Patch
            if isinstance(field_out, _ParserSpecification):
                node.add_edge(field_out._node)
            ### Patch

    return _ParserSpecification(
        f=f,
        markers=markers,
        description=_strings.remove_single_line_breaks(
            description
            if description is not None
            else _docstrings.get_callable_description(f)
        ),
        args=args,
        field_list=field_list,
        child_from_prefix=child_from_prefix,
        helptext_from_intern_prefixed_field_name=helptext_from_intern_prefixed_field_name,
        subparsers=subparsers,
        subparsers_from_intern_prefix=subparsers_from_prefix,
        intern_prefix=intern_prefix,
        extern_prefix=extern_prefix,
        has_required_args=has_required_args,
        consolidate_subcommand_args=consolidate_subcommand_args,
        _node=node,
    )


OutT = TypeVar("OutT")


@overload
def cli(
    f: TypeForm[OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    default: None | OutT = None,
    return_unknown_args: Literal[False] = False,
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    apply_subcomm_defaults: bool = False,
) -> OutT: ...


@overload
def cli(
    f: TypeForm[OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    default: None | OutT = None,
    return_unknown_args: Literal[True],
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    apply_subcomm_defaults: bool = False,
) -> tuple[OutT, list[str]]: ...


@overload
def cli(
    f: Callable[..., OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    # Passing a default makes sense for things like dataclasses, but are not
    # supported for general callables. These can, however, be specified in the
    # signature of the callable itself.
    default: None = None,
    return_unknown_args: Literal[False] = False,
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    apply_subcomm_defaults: bool = False,
) -> OutT: ...


@overload
def cli(
    f: Callable[..., OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    # Passing a default makes sense for things like dataclasses, but are not
    # supported for general callables. These can, however, be specified in the
    # signature of the callable itself.
    default: None = None,
    return_unknown_args: Literal[True],
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    apply_subcomm_defaults: bool = False,
) -> tuple[OutT, list[str]]: ...


def cli(
    f: TypeForm[OutT] | Callable[..., OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    default: None | OutT = None,
    return_unknown_args: bool = False,
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    apply_subcomm_defaults: bool = False,
    **deprecated_kwargs,
) -> OutT | tuple[OutT, list[str]]:
    """Instantiate or call ``f``, with inputs populated from an automatically
    generated CLI interface.

    `f` should have type-annotated inputs, and can be a function or type. If
    ``f`` is a type, ``tyro.cli()`` returns an instance. If ``f`` is a
    function, ``tyro.cli()`` returns the output of calling the function.

    Args:
        f: Function or type.
        prog: The name of the program printed in helptext. Mirrors argument from
            :py:class:`argparse.ArgumentParser()`.
        description: Description text for the parser, displayed when the --help flag is
            passed in. If not specified, ``f``'s docstring is used. Mirrors argument from
            :py:class:`argparse.ArgumentParser()`.
        args: If set, parse arguments from a sequence of strings instead of the
            commandline. Mirrors argument from :py:meth:`argparse.ArgumentParser.parse_args()`.
        default: An instance of ``OutT`` to use for default values; supported if ``f`` is a
            type like a dataclass or dictionary, but not if ``f`` is a general callable
            like a function or standard class. Helpful for merging CLI arguments with
            values loaded from elsewhere. (for example, a config object loaded from a
            yaml file)
        return_unknown_args: If True, return a tuple of the output of ``f`` and a list of
            unknown arguments. Mirrors the unknown arguments returned from
            :py:meth:`argparse.ArgumentParser.parse_known_args()`.
        use_underscores: If True, use underscores as a word delimeter instead of hyphens.
            This primarily impacts helptext; underscores and hyphens are treated
            equivalently when parsing happens. We default helptext to hyphens to follow
            the GNU style guide.
            https://www.gnu.org/software/libc/manual/html_node/Argument-Syntax.html
        console_outputs: If set to ``False``, parsing errors and help messages will be
            supressed. This can be useful for distributed settings, where ``tyro.cli()``
            is called from multiple workers but we only want console outputs from the
            main one.
        config: Sequence of config marker objects, from :mod:`tyro.conf`. As an
            alternative to using them locally in annotations
            (``FlagConversionOff[bool]``), we can also pass in a sequence of
            them here to apply globally.

    Returns:
        The output of ``f(...)`` or an instance ``f``. If ``f`` is a class, the two are
        equivalent. If ``return_unknown_args`` is True, returns a tuple of the output of
        ``f(...)`` and a list of unknown arguments.
    """

    # Make sure we start on a clean slate. Some tests may fail without this due to
    # memory address conflicts.
    _unsafe_cache.clear_cache()

    with _strings.delimeter_context("_" if use_underscores else "-"):
        output = _cli_impl(
            f,
            prog=prog,
            description=description,
            args=args,
            default=default,
            return_parser=False,
            return_unknown_args=return_unknown_args,
            use_underscores=use_underscores,
            console_outputs=console_outputs,
            config=config,
            apply_subcomm_defaults=apply_subcomm_defaults,
            **deprecated_kwargs,
        )

    # Prevent unnecessary memory usage.
    _unsafe_cache.clear_cache()

    if return_unknown_args:
        assert isinstance(output, tuple)
        run_with_args_from_cli = output[0]
        return run_with_args_from_cli(), output[1]
    else:
        run_with_args_from_cli = cast(Callable[[], OutT], output)
        return run_with_args_from_cli()


def get_parser(
    f: TypeForm[OutT] | Callable[..., OutT],
    *,
    # We have no `args` argument, since this is only used when
    # parser.parse_args() is called.
    prog: None | str = None,
    description: None | str = None,
    default: None | OutT = None,
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    apply_subcomm_defaults: bool = False,
) -> argparse.ArgumentParser:
    """Get the ``argparse.ArgumentParser`` object generated under-the-hood by
    :func:`tyro.cli()`. Useful for tools like ``sphinx-argparse``, ``argcomplete``, etc.

    For tab completion, we recommend using :func:`tyro.cli()`'s built-in
    ``--tyro-write-completion`` flag."""
    with _strings.delimeter_context("_" if use_underscores else "-"):
        return cast(
            argparse.ArgumentParser,
            _cli_impl(
                f,
                prog=prog,
                description=description,
                args=None,
                default=default,
                return_parser=True,
                return_unknown_args=False,
                use_underscores=use_underscores,
                console_outputs=console_outputs,
                config=config,
                apply_subcomm_defaults=apply_subcomm_defaults,
            ),
        )


def _cli_impl(
    f: TypeForm[OutT] | Callable[..., OutT],
    *,
    prog: None | str = None,
    description: None | str,
    args: None | Sequence[str],
    default: None | OutT,
    return_parser: bool,
    return_unknown_args: bool,
    console_outputs: bool,
    config: None | Sequence[conf._markers.Marker],
    apply_subcomm_defaults: bool = False,
    **deprecated_kwargs,
) -> (
    OutT
    | argparse.ArgumentParser
    | tuple[
        Callable[[], OutT],
        list[str],
    ]
):
    """Helper for stitching the `tyro` pipeline together."""

    assert args is None
    args = list(sys.argv[1:])

    if config is not None:
        f = Annotated[(f, *config)]  # type: ignore

    if "default_instance" in deprecated_kwargs:
        warnings.warn(
            "`default_instance=` is deprecated! use `default=` instead.", stacklevel=2
        )
        default = deprecated_kwargs["default_instance"]
    if deprecated_kwargs.get("avoid_subparsers", False):
        f = conf.AvoidSubcommands[f]  # type: ignore
        warnings.warn(
            "`avoid_subparsers=` is deprecated! use `tyro.conf.AvoidSubcommands[]`"
            " instead.",
            stacklevel=2,
        )

    # Internally, we distinguish between two concepts:
    # - "default", which is used for individual arguments.
    # - "default_instance", which is used for _fields_ (which may be broken down into
    #   one or many arguments, depending on various factors).
    #
    # This could be revisited.
    default_instance_internal: _singleton.NonpropagatingMissingType | OutT = (
        _singleton.MISSING_NONPROP if default is None else default
    )

    # We wrap our type with a dummy dataclass if it can't be treated as a nested type.
    # For example: passing in f=int will result in a dataclass with a single field
    # typed as int.
    if not _fields.is_struct_type(cast(type, f), default_instance_internal):
        dummy_field = cast(
            dataclasses.Field,
            dataclasses.field(),
        )
        f = dataclasses.make_dataclass(
            cls_name="dummy",
            fields=[(_strings.dummy_field_name, cast(type, f), dummy_field)],
            frozen=True,
        )
        default_instance_internal = f(default_instance_internal)  # type: ignore
        dummy_wrapped = True
    else:
        dummy_wrapped = False

    # Read and fix arguments. If the user passes in --field_name instead of
    # --field-name, correct for them.
    args = list(sys.argv[1:]) if args is None else list(args)

    # Fix arguments. This will modify all option-style arguments replacing
    # underscores with hyphens, or vice versa if use_underscores=True.
    # If two options are ambiguous, e.g., --a_b and --a-b, raise a runtime error.
    modified_args: dict[str, str] = {}
    for index, arg in enumerate(args):
        if not arg.startswith("--"):
            continue

        if "=" in arg:
            arg, _, val = arg.partition("=")
            fixed = "--" + _strings.swap_delimeters(arg[2:]) + "=" + val
        else:
            fixed = "--" + _strings.swap_delimeters(arg[2:])
        if (
            return_unknown_args
            and fixed in modified_args
            and modified_args[fixed] != arg
        ):
            raise RuntimeError(
                "Ambiguous arguments: " + modified_args[fixed] + " and " + arg
            )
        modified_args[fixed] = arg
        args[index] = fixed

    # If we pass in the --tyro-print-completion or --tyro-write-completion flags: turn
    # formatting tags, and get the shell we want to generate a completion script for
    # (bash/zsh/tcsh).
    #
    # shtab also offers an add_argument_to() functions that fulfills a similar goal, but
    # manual parsing of argv is convenient for turning off formatting.
    #
    # Note: --tyro-print-completion is deprecated! --tyro-write-completion is less prone
    # to errors from accidental logging, print statements, etc.
    print_completion = False
    write_completion = False
    if len(args) >= 2:
        # We replace underscores with hyphens to accomodate for `use_undercores`.
        print_completion = args[0].replace("_", "-") == "--tyro-print-completion"
        write_completion = (
            len(args) >= 3 and args[0].replace("_", "-") == "--tyro-write-completion"
        )

    # Note: setting USE_RICH must happen before the parser specification is generated.
    # TODO: revisit this. Ideally we should be able to eliminate the global state
    # changes.
    completion_shell = None
    completion_target_path = None
    if print_completion or write_completion:
        completion_shell = args[1]
    if write_completion:
        completion_target_path = pathlib.Path(args[2])
    if print_completion or write_completion or return_parser:
        _arguments.USE_RICH = False
    else:
        _arguments.USE_RICH = True

    # Patch the parser cosntructor here!
    ParserSpecification.from_callable_or_type = staticmethod(from_callable_or_type)

    # Map a callable to the relevant CLI arguments + subparsers.
    parser_spec = ParserSpecification.from_callable_or_type(
        f,
        markers=set(),
        description=description,
        parent_classes=set(),  # Used for recursive calls.
        default_instance=default_instance_internal,  # Overrides for default values.
        intern_prefix="",  # Used for recursive calls.
        extern_prefix="",  # Used for recursive calls.
    )

    consolidate_subcomm = parser_spec.consolidate_subcommand_args

    ### Patch
    if apply_subcomm_defaults:
        if not consolidate_subcomm:
            raise ValueError(
                "Defaults instantiation enabled only for tyro.conf.ConsolidateSubcommandArgs"
            )

        # TODO: It's possible it works with this marker, needs more testing.
        if config is not None and conf.OmitSubcommandPrefixes in tuple(config):
            raise ValueError(
                "OmitSubcommandPrefix not compatible with apply_subcomm_defaults."
            )

        # TODO: Does it work with OmitArgPrefixes?

    if consolidate_subcomm and apply_subcomm_defaults:
        assert isinstance(parser_spec, _ParserSpecification)

        root_to_user_option = {}
        dsu = DSU()

        first_non_choice = None
        for index, arg_item in enumerate(args):
            if dsu.is_node(arg_item):
                root = dsu.find(arg_item)

                if root in root_to_user_option:
                    raise ValueError(
                        "Cannot specify multiple choices for the same subcommand,"
                        f"got {arg_item} and {root_to_user_option[root]}"
                    )

                root_to_user_option[root] = arg_item
            else:
                first_non_choice = index
                break

        found_defaults = False
        new_args = []
        queue: deque[_Node] = deque([parser_spec._node])
        while queue:
            node = queue.pop()
            if node.is_subparser:
                choice = node.default_name

                if (choice_root := dsu.find(choice)) in root_to_user_option:
                    choice = root_to_user_option[choice_root]

                for nei in node.edges:
                    if nei.default_name == choice:
                        queue.append(nei)
                        break
            else:
                if node.default_name != _Node.DEFAULT_PARSER_NAME:
                    new_args.append(node.default_name)
                    found_defaults = True
                    queue.extend(node.edges)
                else:
                    for nei in node.edges:
                        queue.appendleft(nei)

        if first_non_choice is not None:
            new_args.extend(args[first_non_choice:])

        args = new_args

        if not found_defaults:
            warnings.warn("apply_subcomm_defaults set to True but no defaults found.")
    ### Patch

    # Generate parser!
    with _argparse_formatter.ansi_context():
        parser = _argparse_formatter.TyroArgumentParser(
            prog=prog,
            formatter_class=_argparse_formatter.TyroArgparseHelpFormatter,
            allow_abbrev=False,
        )
        parser._parser_specification = parser_spec
        parser._parsing_known_args = return_unknown_args
        parser._console_outputs = console_outputs
        parser._args = args
        parser_spec.apply(parser, force_required_subparsers=False)

        # Print help message when no arguments are passed in. (but arguments are
        # expected)
        # if len(args) == 0 and parser_spec.has_required_args:
        #     args = ["--help"]

        if return_parser:
            _arguments.USE_RICH = True
            return parser

        if print_completion or write_completion:
            _arguments.USE_RICH = True
            assert completion_shell in (
                "bash",
                "zsh",
                "tcsh",
            ), (
                "Shell should be one `bash`, `zsh`, or `tcsh`, but got"
                f" {completion_shell}"
            )

            if write_completion and completion_target_path != pathlib.Path("-"):
                assert completion_target_path is not None
                completion_target_path.write_text(
                    shtab.complete(
                        parser=parser,
                        shell=completion_shell,
                        root_prefix=f"tyro_{parser.prog}",
                    )
                )
            else:
                print(
                    shtab.complete(
                        parser=parser,
                        shell=completion_shell,
                        root_prefix=f"tyro_{parser.prog}",
                    )
                )
            sys.exit()

        if return_unknown_args:
            namespace, unknown_args = parser.parse_known_args(args=args)
        else:
            unknown_args = None
            namespace = parser.parse_args(args=args)
        value_from_prefixed_field_name = vars(namespace)

    if dummy_wrapped:
        value_from_prefixed_field_name = {
            k.replace(_strings.dummy_field_name, ""): v
            for k, v in value_from_prefixed_field_name.items()
        }

    try:
        # Attempt to call `f` using whatever was passed in.
        get_out, consumed_keywords = _calling.callable_with_args(
            f,
            parser_spec,
            default_instance_internal,
            value_from_prefixed_field_name,
            field_name_prefix="",
        )
    except _calling.InstantiationError as e:
        # Print prettier errors.
        # This doesn't catch errors raised directly by get_out(), since that's
        # called later! This is intentional, because we do less error handling
        # for the root callable. Relevant: the `field_name_prefix == ""`
        # condition in `callable_with_args()`!

        # Emulate argparse's error behavior when invalid arguments are passed in.
        from rich.console import Console, Group
        from rich.padding import Padding
        from rich.panel import Panel
        from rich.rule import Rule
        from rich.style import Style
        from tyro._argparse_formatter import THEME

        if console_outputs:
            console = Console(theme=THEME.as_rich_theme(), stderr=True)
            console.print(
                Panel(
                    Group(
                        "[bright_red][bold]Error parsing"
                        f" {'/'.join(e.arg.lowered.name_or_flags) if isinstance(e.arg, _arguments.ArgumentDefinition) else e.arg}[/bold]:[/bright_red] {e.message}",
                        *cast(  # Cast to appease mypy...
                            list,
                            (
                                []
                                if not isinstance(e.arg, _arguments.ArgumentDefinition)
                                or e.arg.lowered.help is None
                                else [
                                    Rule(style=Style(color="red")),
                                    "Argument helptext:",
                                    Padding(
                                        Group(
                                            f"{'/'.join(e.arg.lowered.name_or_flags)} [bold]{e.arg.lowered.metavar}[/bold]",
                                            e.arg.lowered.help,
                                        ),
                                        pad=(0, 0, 0, 4),
                                    ),
                                    Rule(style=Style(color="red")),
                                    f"For full helptext, see [bold]{parser.prog} --help[/bold]",
                                ]
                            ),
                        ),
                    ),
                    title="[bold]Value error[/bold]",
                    title_align="left",
                    border_style=Style(color="red"),
                )
            )
        sys.exit(2)

    assert len(value_from_prefixed_field_name.keys() - consumed_keywords) == 0, (
        f"Parsed {value_from_prefixed_field_name.keys()}, but only consumed"
        f" {consumed_keywords}"
    )

    if dummy_wrapped:
        get_wrapped_out = get_out
        get_out = lambda: getattr(get_wrapped_out(), _strings.dummy_field_name)  # noqa

    if return_unknown_args:
        assert unknown_args is not None, "Should have parsed with `parse_known_args()`"
        # If we're parsed unknown args, we should return the original args, not
        # the fixed ones.
        unknown_args = [modified_args.get(arg, arg) for arg in unknown_args]
        return get_out, unknown_args  # type: ignore
    else:
        assert unknown_args is None, "Should have parsed with `parse_args()`"
        return get_out  # type: ignore
