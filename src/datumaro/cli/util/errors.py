# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from attr import attrib, attrs

from datumaro.components.errors import DatumaroError


class CliException(DatumaroError):
    pass


@attrs(frozen=True)
class RevpathParseProblem:
    description = attrib()
    error = attrib()

    def format(self, indent="  "):
        lines = [f"{indent}{self.description}:"]
        cause = self.error
        seen = set()
        cause_indent = indent * 2

        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            message = str(cause).strip()
            details = type(cause).__name__
            if message:
                details += f": {message}"

            prefix = "" if len(lines) == 1 else "Caused by: "
            lines.append(f"{cause_indent}{prefix}{details}")
            cause = cause.__cause__

        return "\n".join(lines)


@attrs
class WrongRevpathError(CliException):
    revpath = attrib()
    problems = attrib()

    def __str__(self):
        details = "\n\n".join(problem.format() for problem in self.problems)
        return f"Failed to parse revpath {self.revpath!r}:\n\n{details}"
