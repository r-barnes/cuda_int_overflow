#!/usr/bin/env python3

import itertools
import subprocess
import tempfile

template = """#include <cxxabi.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>

#include "RichardSafeInt.hpp"

using namespace rsi;

std::string demangle_helper(const char* mangled)
{{
  int status;
  std::unique_ptr<char[], void (*)(void*)> result(
  abi::__cxa_demangle(mangled, 0, 0, &status), std::free);
  return result.get() ? std::string(result.get()) : "error occurred";
}}

template<class T>
std::string demangle(T t) {{
  return demangle_helper(typeid(t).name());
}}

int main() {{
{entries}
}}
"""

cpp_types: list[str] = [
    "bool",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    # "float",
    # "double",
    # "long double",
]

entry_template = '  std::cout << "{a} + {b} = " << demangle(static_cast<{a}>(1) + static_cast<{b}>(1)) << " pod" << std::endl;'
entry_list = [
    entry_template.format(prefix="a", a=a, b=b) for a, b in itertools.product(cpp_types, cpp_types)
]

entry_template = '  std::cout << "{a} + {b} = " << demangle(SafeInt(static_cast<{a}>(1)) + SafeInt(static_cast<{b}>(1))) << std::endl;'
entry_list += [
    entry_template.format(prefix="b", a=a, b=b) for a, b in itertools.product(cpp_types, cpp_types)
]

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".cpp", delete_on_close=False, dir="."
) as source:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".exe", delete_on_close=False
    ) as exe:
        source.write(template.format(entries="\n".join(entry_list)))
        source.close()
        exe.close()
        subprocess.run(["clang++", "--std=c++20", "-o", exe.name, source.name], check=True)
        subprocess.run([exe.name], check=True)