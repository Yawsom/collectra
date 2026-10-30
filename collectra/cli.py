"""Generic, recursive Cappa adapter for runtime Python objects."""

from __future__ import annotations

import inspect
from dataclasses import field, fields, make_dataclass, replace
from typing import (
    Annotated,
    Any,
    Callable,
    Mapping,
    get_args,
    get_origin,
    get_type_hints,
)

import cappa

CLI_KIND = "__collectra_cli_kind__"
CLI_NAME = "__collectra_cli_name__"
CLI_AVAILABLE = "__collectra_cli_available__"


def _member(
    kind: str,
    name: str | None = None,
    available: Callable[[], Any] | None = None,
):
    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, CLI_KIND, kind)
        setattr(func, CLI_NAME, name)
        setattr(func, CLI_AVAILABLE, available)
        return func

    return decorate


def command(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    available: Callable[[], Any] | None = None,
):
    """Mark a method as a terminal CLI command.

    ``available`` is checked each time the CLI is built; a command whose check
    is falsy (for example, one that needs an optional program) is left out.
    """
    decorator = _member("command", name, available)
    return decorator(func) if func is not None else decorator


def group(
    func: Callable[..., Mapping[str, Any]] | None = None, *, name: str | None = None
):
    """Mark a method as a mapping of named child CLI objects."""
    decorator = _member("group", name)
    return decorator(func) if func is not None else decorator


def _cli_members(obj: Any):
    for member_name in dir(obj):
        # Properties can load models or optional backends. Only bind members
        # after confirming that their static definition is a CLI command.
        definition = inspect.getattr_static(obj, member_name)
        if isinstance(definition, (staticmethod, classmethod)):
            definition = definition.__func__
        if not getattr(definition, CLI_KIND, None):
            continue
        available = getattr(definition, CLI_AVAILABLE, None)
        if available is not None and not available():
            continue
        member = getattr(obj, member_name)
        kind = getattr(member, CLI_KIND, None)
        if callable(member) and kind:
            cli_name = getattr(member, CLI_NAME, None) or member_name
            yield cli_name.replace("_", "-"), kind, member


def _callable_schema(name: str, func: Callable[..., Any]) -> type:
    signature = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    schema_fields = []

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            raise TypeError(
                f"CLI command {func.__qualname__} cannot expose **{parameter.name}; "
                "declare typed parameters instead"
            )

        annotation = hints.get(parameter.name, str)
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            annotation = list[annotation]
            schema_fields.append(
                (parameter.name, annotation, field(default_factory=list))
            )
        elif parameter.default is inspect.Parameter.empty:
            schema_fields.append((parameter.name, annotation))
        else:
            base_type = annotation
            metadata = []
            if get_origin(annotation) is Annotated:
                base_type, *metadata = get_args(annotation)
            if base_type is bool:
                cli_name = parameter.name.replace("_", "-")
                long = f"--{cli_name}/--no-{cli_name}"
            else:
                long = True
            if any(isinstance(item, cappa.Arg) for item in metadata):
                metadata = [
                    (
                        replace(item, long=long)
                        if isinstance(item, cappa.Arg)
                        and not item.long
                        and not item.short
                        else item
                    )
                    for item in metadata
                ]
            else:
                metadata.append(cappa.Arg(long=long))
            option = Annotated[(base_type, *metadata)]
            schema_fields.append((parameter.name, option, parameter.default))

    return make_dataclass(f"{name.title().replace('-', '')}Arguments", schema_fields)


def command_for_callable(name: str, func: Callable[..., Any]) -> cappa.Command:
    """Build a terminal Cappa command from a bound callable."""
    schema = _callable_schema(name, func)
    signature = inspect.signature(func)

    def invoke(parsed):
        values = {item.name: getattr(parsed, item.name) for item in fields(parsed)}
        positional = []
        keyword = {}
        for parameter in signature.parameters.values():
            value = values[parameter.name]
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                positional.extend(value)
            elif parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                positional.append(value)
            else:
                keyword[parameter.name] = value
        return func(*positional, **keyword)

    invoke.__annotations__ = {"parsed": schema}
    invoke.__name__ = f"invoke_{name.replace('-', '_')}"

    return cappa.Command(
        schema,
        name=name,
        help=inspect.getdoc(func),
        invoke=invoke,
    )


def command_for_object(
    obj: Any, *, name: str | None = None, help: str | None = None
) -> cappa.Command:
    """Recursively convert a decorated runtime object into a Cappa command."""
    options: dict[str, cappa.Command] = {}

    for cli_name, kind, member in _cli_members(obj):
        if kind == "command":
            options[cli_name] = command_for_callable(cli_name, member)
            continue

        children = member()
        if not isinstance(children, Mapping):
            raise TypeError(f"CLI group {member.__qualname__} must return a mapping")
        group_options = {
            str(child_name): command_for_object(
                child,
                name=str(child_name),
                help=inspect.getdoc(child.__class__),
            )
            for child_name, child in children.items()
        }
        options[cli_name] = _group_command(
            cli_name, group_options, inspect.getdoc(member)
        )

    command_name = name or obj.__class__.__name__.lower()
    return _group_command(command_name, options, help or inspect.getdoc(obj.__class__))


def _group_command(
    name: str, options: Mapping[str, cappa.Command], help: str | None
) -> cappa.Command:
    schema = make_dataclass(
        f"{name.title().replace('-', '')}Command",
        [("selected", Any, field(default=None))],
    )
    arguments = []
    if options:
        arguments.append(cappa.Subcommand(field_name="selected", options=options))
    return cappa.Command(schema, name=name, arguments=arguments, help=help)


def invoke(obj: Any, argv: list[str] | None = None, *, name: str = "collectra"):
    """Build and invoke the recursive command tree for ``obj``."""
    return cappa.invoke(command_for_object(obj, name=name), argv=argv)
