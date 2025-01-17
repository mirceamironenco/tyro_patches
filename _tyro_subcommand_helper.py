from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import List

from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tyro import _argparse, _argparse_formatter, _strings


def is_subparser_action(obj) -> bool:
    return isinstance(obj, _argparse._SubParsersAction)


def is_help_action(obj) -> bool:
    return isinstance(obj, _argparse._HelpAction)


class TyroFlatSubcommandHelpFormatter(_argparse_formatter.TyroArgparseHelpFormatter):
    """
    A custom formatter that ensures --help prints arguments for all subcommands
    (tyro's default is to print for current / next subcommand).
    """

    def add_usage(self, usage, actions, groups, prefix=None):
        aggregated_subcommand_group = []
        for _, sub_parser in self.collect_subcommands_parsers(actions).items():
            for sub_action_group in sub_parser._action_groups:
                sub_group_actions = sub_action_group._group_actions
                if len(sub_group_actions) > 0:
                    if any(
                        [
                            is_subparser_action(a) and not is_help_action(a)
                            for a in sub_group_actions
                        ]
                    ):
                        aggregated_subcommand_group.append(sub_action_group)

        # Remove duplicate subcommand parsers
        aggregated_subcommand_group = list(
            {
                a._group_actions[0].metavar: a for a in aggregated_subcommand_group
            }.values()
        )
        next_actions = [g._group_actions[0] for g in aggregated_subcommand_group]
        actions.extend(next_actions)
        super().add_usage(usage, actions, groups, prefix)

    def add_arguments(self, action_group):
        if len(action_group) > 0 and action_group[0].container.title in (
            "subcommands",
            "optional subcommands",
        ):
            # If a subcommands action group - rename first subcommand (for which this function was invoked)
            choices_header = next(iter(action_group[0].choices))
            choices_title = choices_header.split(":")[0] + " choices"
            action_group[0].container.title = choices_title
            self._current_section.heading = (
                choices_title  # Formatter have already set a section, override heading
            )

        # Invoke default
        super().add_arguments(action_group)

        aggregated_action_group = []
        aggregated_subcommand_group = []
        for action in action_group:
            if not is_subparser_action(action):
                continue

            subcommands_parsers = self.collect_subcommands_parsers([action])

            for _, sub_parser in subcommands_parsers.items():
                sub_parser.formatter_class = self
                for sub_action_group in sub_parser._action_groups:
                    sub_group_actions = sub_action_group._group_actions
                    if len(sub_group_actions) > 0:
                        if any(
                            [
                                not is_subparser_action(a) and not is_help_action(a)
                                for a in sub_group_actions
                            ]
                        ):
                            continue
                        elif any([not is_help_action(a) for a in sub_group_actions]):
                            for a in sub_group_actions:
                                choices_header = next(
                                    iter(sub_group_actions[0].choices)
                                )
                                a.container.title = (
                                    choices_header.split(":")[0] + " choices"
                                )
                            aggregated_subcommand_group.append(sub_action_group)

        # Remove duplicate subcommand parsers
        aggregated_subcommand_group = list(
            {
                a._group_actions[0].metavar: a for a in aggregated_subcommand_group
            }.values()
        )
        for aggregated_group in (aggregated_subcommand_group, aggregated_action_group):
            for next_action_group in aggregated_group:
                self.end_section()
                self.start_section(next_action_group.title)
                self.add_text(next_action_group.description)
                super().add_arguments(next_action_group._group_actions)

    def collect_subcommands_parsers(self, actions):
        collected_titles = list()
        collected_subparsers = list()
        parsers = list()

        def _handle_actions(_actions):
            action_choices = [
                action.choices for action in _actions if is_subparser_action(action)
            ]
            for choices in action_choices:
                for subcommand, subcommand_parser in choices.items():
                    collected_titles.append(subcommand)
                    collected_subparsers.append(subcommand_parser)
                    parsers.append(subcommand_parser)

        _handle_actions(actions)
        while parsers:
            parser = parsers.pop(0)
            _handle_actions(parser._actions)

        # Eliminate duplicates and preserve order (dicts are guaranteed to preserve insertion order from python >=3.7)
        return dict(zip(collected_titles, collected_subparsers))

    class _Section(object):  # type: ignore
        def __init__(self, formatter, parent, heading=None):
            self.formatter = formatter
            self.parent = parent
            self.heading = heading
            self.items = []
            self.formatter._tyro_rule = None

        def format_help(self):
            if self.parent is None:
                return self._tyro_format_root()
            else:
                return self._tyro_format_nonroot()

        def _tyro_format_root(self):
            dummy_console = Console(
                width=self.formatter._width,
                theme=_argparse_formatter.THEME.as_rich_theme(),
            )
            with dummy_console.capture() as capture:
                # Get rich renderables from items.
                top_parts = []
                column_parts = []
                column_parts_lines = []
                for func, args in self.items:
                    item_content = func(*args)
                    if item_content is None:
                        pass

                    # Add strings. (usage, description, etc)
                    elif isinstance(item_content, str):
                        if item_content.strip() == "":
                            continue
                        top_parts.append(Text.from_ansi(item_content))

                    # Add panels. (argument groups, subcommands, etc)
                    else:
                        assert isinstance(item_content, Panel)
                        column_parts.append(item_content)
                        # Estimate line count. This won't correctly account for
                        # wrapping, as we don't know the column layout yet.
                        column_parts_lines.append(
                            _argparse_formatter.str_from_rich(item_content, width=65)
                            .strip()
                            .count("\n")
                            + 1
                        )

                # Split into columns.
                min_column_width = 65
                height_breakpoint = 50
                column_count = max(
                    1,
                    min(
                        sum(column_parts_lines) // height_breakpoint + 1,
                        self.formatter._width // min_column_width,
                        len(column_parts),
                    ),
                )
                done = False
                column_parts_grouped = None
                column_width = None
                while not done:
                    if column_count > 1:  # pragma: no cover
                        column_width = self.formatter._width // column_count - 1
                        # Correct the line count for each panel using the known column
                        # width. This will account for word wrap.
                        column_parts_lines = [
                            _argparse_formatter.str_from_rich(p, width=column_width)
                            .strip()
                            .count("\n")
                            + 1
                            for p in column_parts
                        ]
                    else:
                        column_width = None

                    column_lines = [0 for i in range(column_count)]
                    column_parts_grouped = [[] for i in range(column_count)]
                    for p, l in zip(column_parts, column_parts_lines):
                        chosen_column = column_lines.index(min(column_lines))
                        column_parts_grouped[chosen_column].append(p)
                        column_lines[chosen_column] += l

                    column_lines_max = max(*column_lines, 1)  # Prevent divide-by-zero.
                    column_lines_ratio = [l / column_lines_max for l in column_lines]

                    # Done if we're down to one column or all columns are
                    # within 60% of the maximum height.
                    #
                    # We use these ratios to prevent large hanging columns: https://github.com/brentyi/tyro/issues/222
                    if column_count == 1 or all(
                        [ratio > 0.3 for ratio in column_lines_ratio]
                    ):
                        break
                    column_count -= 1  # pragma: no cover

                assert column_parts_grouped is not None
                columns = Columns(
                    [Group(*g) for g in column_parts_grouped],
                    column_first=True,
                    width=column_width,
                )

                dummy_console.print(Group(*top_parts))
                dummy_console.print(columns)
            return capture.get()

        def _format_action(self, action: _argparse_formatter.argparse.Action):
            invocation = self.formatter._format_action_invocation(action)
            indent = self.formatter._current_indent
            help_position = min(
                self.formatter._action_max_length + 4,
                self.formatter._max_help_position,
            )
            if self.formatter._fixed_help_position:
                help_position = 4

            item_parts: List[RenderableType] = []

            # Put invocation and help side-by-side.
            if action.option_strings == ["-h", "--help"]:
                # Darken helptext for --help flag. This makes it visually consistent
                # with the helptext strings defined via docstrings and set by
                # _arguments.py.
                assert action.help is not None
                action.help = _argparse_formatter.str_from_rich(
                    Text.from_markup("[helptext]" + action.help + "[/helptext]")
                )

            # Unescape % signs, which need special handling in argparse.
            if action.help is not None:
                assert isinstance(action.help, str)
                helptext = (
                    Text.from_ansi(action.help.replace("%%", "%"))
                    if _strings.strip_ansi_sequences(action.help) != action.help
                    else Text.from_markup(action.help.replace("%%", "%"))
                )
            else:
                helptext = Text("")

            if (
                action.help
                and len(_strings.strip_ansi_sequences(invocation)) + indent
                < help_position - 1
                and not self.formatter._fixed_help_position
            ):
                table = Table(show_header=False, box=None, padding=0)
                table.add_column(width=help_position - indent)
                table.add_column()
                table.add_row(
                    Text.from_ansi(
                        invocation,
                        style=_argparse_formatter.THEME.invocation,
                    ),
                    helptext,
                )
                item_parts.append(table)

            # Put invocation and help on separate lines.
            else:
                item_parts.append(
                    Text.from_ansi(
                        invocation + "\n",
                        style=_argparse_formatter.THEME.invocation,
                    )
                )
                if action.help:
                    item_parts.append(
                        Padding(
                            # Unescape % signs, which need special handling in argparse.
                            helptext,
                            pad=(0, 0, 0, help_position - indent),
                        )
                    )

            # Add subactions, indented.
            try:
                subaction: _argparse_formatter.argparse.Action
                for subaction in action._get_subactions():  # type: ignore
                    self.formatter._indent()
                    item_parts.append(
                        Padding(
                            Group(*self._format_action(subaction)),
                            pad=(0, 0, 0, self.formatter._indent_increment),
                        )
                    )
                    self.formatter._dedent()
            except AttributeError:
                pass

            return item_parts

        def _tyro_format_nonroot(self):
            # Add each child item as a rich renderable.
            description_part = None
            item_parts = []
            for func, args in self.items:
                if (
                    getattr(func, "__func__", None)
                    is _argparse_formatter.TyroArgparseHelpFormatter._format_action
                ):
                    (action,) = args
                    assert isinstance(action, _argparse_formatter.argparse.Action)
                    item_parts.extend(self._format_action(action))

                else:
                    item_content = func(*args)
                    assert isinstance(item_content, str)
                    if item_content.strip() != "":
                        assert (
                            description_part is None
                        )  # Should only have one description part.
                        description_part = Text.from_ansi(
                            item_content.strip() + "\n",
                            style=_argparse_formatter.THEME.description,
                        )

            if len(item_parts) == 0:
                return None

            # Get heading.
            if (
                self.heading is not _argparse_formatter.argparse.SUPPRESS
                and self.heading is not None
            ):
                current_indent = self.formatter._current_indent
                heading = "%*s%s:\n" % (current_indent, "", self.heading)
                # Remove colon from heading.
                heading = heading.strip()[:-1]
            else:
                heading = ""

            # Determine width for divider below description text. This is shared across
            # all sections in a particular formatter.
            lines = list(
                itertools.chain(
                    *map(
                        lambda p: _strings.strip_ansi_sequences(
                            _argparse_formatter.str_from_rich(
                                p, width=self.formatter._width, soft_wrap=True
                            )
                        )
                        .rstrip()
                        .split("\n"),
                        (
                            item_parts + [description_part]
                            if description_part is not None
                            else item_parts
                        ),
                    )
                )
            )
            max_width = max(map(len, lines))

            if self.formatter._tyro_rule is None:
                # We don't use rich.rule.Rule() because this will make all of the panels
                # expand to fill the full width of the console. This only impacts
                # single-column layouts.
                self.formatter._tyro_rule = Text.from_ansi(
                    "─" * max_width,
                    style=_argparse_formatter.THEME.border,
                    overflow="crop",
                )
            elif len(self.formatter._tyro_rule._text[0]) < max_width:
                self.formatter._tyro_rule._text = ["─" * max_width]

            # Add description text if needed.
            if description_part is not None:
                item_parts = [
                    description_part,
                    self.formatter._tyro_rule,
                ] + item_parts

            return Panel(
                Group(*item_parts),
                title=heading,
                title_align="left",
                border_style=_argparse_formatter.THEME.border,
                # padding=(1, 1, 0, 1),
            )

    def _format_usage(
        self,
        usage,
        actions: Iterable[_argparse_formatter.argparse.Action],
        groups,
        prefix,
    ) -> str:
        assert isinstance(actions, list)
        if len(actions) > 4:
            new_actions = []
            # prog_parts = shlex.split(self._prog)
            added_options = False
            for action in actions:
                if action.dest == "help" or len(action.option_strings) == 0:
                    new_actions.append(action)
                elif not added_options:
                    added_options = True
                    new_actions.append(
                        _argparse_formatter.argparse.Action(
                            [
                                (
                                    "OPTIONS"
                                    # if len(prog_parts) == 1
                                    # else prog_parts[-1].upper() + " OPTIONS"
                                )
                            ],
                            dest="",
                        )
                    )
            actions = new_actions

        # Format the usage label.
        if prefix is None:
            prefix = _argparse_formatter.str_from_rich("[bold]usage[/bold]: ")
        usage = super()._format_usage(
            usage,
            actions,
            groups,
            prefix,
        )
        return "\n\n" + usage
