#!/usr/bin/env python3

import argparse
from dataclasses import dataclass

OUTPUT_FILE = "RichardSafeInt.hpp"


@dataclass(frozen=True)
class Op:
    name: str
    op: str


arithmetic_operators = [
    Op("ADD", "+"),
    Op("SUB", "-"),
    Op("DIV", "/"),
    Op("MUL", "*"),
    Op("MOD", "%"),
]

FILE_TEMPLATE = """#pragma once

// Richard's CUDA Safe Int Class
{generated_code_disclaimer}

#include <cstdint>
#include <ostream>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define RSI_HOST_DEVICE __host__ __device__
#else
#define RSI_HOST_DEVICE
#endif

#ifndef RSI_CONVERSION_TYPE
#define RSI_CONVERSION_TYPE // blank means implicit
#endif

#ifndef RSI_ALLOW_CONVERSION
#define RSI_ALLOW_CONVERSION 1
#endif

#ifndef RSI_ASSIGN_BEHAVIOUR
#define RSI_ASSIGN_BEHAVIOUR rsi::rsi_assign_behaviour::STANDARD_BEHAVIOUR
#endif
{behaviour_defaults}

namespace rsi {{

namespace detail {{

template<class T> struct LargerType {{}};
template<> struct LargerType<bool> {{ using type = int32_t; }};
template<> struct LargerType<int8_t> {{ using type = int32_t; }};
template<> struct LargerType<uint8_t> {{ using type = int32_t; }};
template<> struct LargerType<int16_t> {{ using type = int32_t; }};
template<> struct LargerType<uint16_t> {{ using type = int32_t; }};
template<> struct LargerType<int32_t> {{ using type = int64_t; }};
template<> struct LargerType<uint32_t> {{ using type = int64_t; }};

// Needed until C++23, see https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
template <typename...> inline constexpr bool always_false = false;

}}

enum class rsi_arith_behaviour {{
    STANDARD_BEHAVIOUR,
    STANDARD_BUT_NOT_SAME_SIZE_UNLESS_64,
    SAME_SIZE_PROMOTE_TO_LARGER_SIGNED_UNLESS_64,
    NONE
}};

enum class rsi_arith_accum_behaviour {{
    STANDARD_BEHAVIOUR,
    SMALLER_OR_EQUAL,
    ONLY_SMALLER,
    NONE
}};

enum class rsi_assign_behaviour {{
    STANDARD_BEHAVIOUR,
    SMALLER_OR_EQUAL_SAME_SIGN,
    SMALLER_OR_EQUAL,
    NONE
}};

// NOT YET USED
// enum class rsi_bitwise_behaviour {{
//     STANDARD_BEHAVIOUR,
//     STANDARD_BUT_ONLY_UNSIGNED,
//     UNSIGNED_AND_SAME_SIZE,
//     NONE
// }};

template <typename T>
class SafeInt {{
  private:
    static_assert(std::is_integral_v<T>, "SafeInt can only hold integers");
    T m_value = 0;

  public:

    constexpr SafeInt() = default;
    RSI_HOST_DEVICE constexpr SafeInt(T v) : m_value(v) {{}}

    #if RSI_ALLOW_CONVERSION
    RSI_CONVERSION_TYPE RSI_HOST_DEVICE constexpr operator T() const {{ return m_value; }}
    #endif

    RSI_HOST_DEVICE T value() const {{ return m_value; }}
    RSI_HOST_DEVICE T& value() {{ return m_value; }}

    template<typename U>
    RSI_HOST_DEVICE SafeInt<T>(const SafeInt<U> &o){{
        if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::STANDARD_BEHAVIOUR) {{
            m_value = o.m_value;
        }} else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::SMALLER_OR_EQUAL) {{
            if constexpr (sizeof(T) >= sizeof(U)) {{
                m_value = o.m_value;
            }} else {{
                static_assert(detail::always_false<T>, "Assignment not allowed: rhs type is larger than lhs type.");
            }}
        }} else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::SMALLER_OR_EQUAL_SAME_SIGN) {{
            if constexpr (sizeof(T) >= sizeof(U) && std::is_signed_v<T> == std::is_signed_v<U>) {{
                m_value = o.m_value;
            }} else {{
                static_assert(detail::always_false<T>, "Assignment not allowed: rhs type is larger than lhs type.");
            }}
        }} else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::NONE){{
            static_assert(detail::always_false<T>, "Performing assignment is not allowed.");
        }} else {{
            static_assert(detail::always_false<T>, "Unrecognized behaviour.");
        }}
    }}

    template<typename U>
    RSI_HOST_DEVICE SafeInt<T>& operator=(const SafeInt<U> &o){{
        if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::STANDARD_BEHAVIOUR) {{
            m_value = o.m_value;
            return *this;
        }} else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::SMALLER_OR_EQUAL) {{
            if constexpr (sizeof(T) >= sizeof(U)) {{
                m_value = o.m_value;
                return *this;
            }} else {{
                static_assert(detail::always_false<T>, "Assignment not allowed: rhs type is larger than lhs type.");
            }}
        }} else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::SMALLER_OR_EQUAL_SAME_SIGN) {{
            if constexpr (sizeof(T) >= sizeof(U) && std::is_signed_v<T> == std::is_signed_v<U>) {{
                m_value = o.m_value;
                return *this;
            }} else {{
                static_assert(detail::always_false<T>, "Assignment not allowed: rhs type is larger than lhs type.");
            }}
        }} else if constexpr (RSI_ASSIGN_BEHAVIOUR == rsi::rsi_assign_behaviour::NONE){{
            static_assert(detail::always_false<T>, "Performing assignment is not allowed.");
        }} else {{
            static_assert(detail::always_false<T>, "Unrecognized behaviour.");
        }}
    }}

    template<typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
    RSI_HOST_DEVICE SafeInt<T>& operator=(const U &rhs){{
        return (*this = SafeInt(rhs));
    }}

    // prefix increment
    RSI_HOST_DEVICE SafeInt<T>& operator++(){{
        ++m_value;
        return *this;
    }}
 
    // postfix increment
    RSI_HOST_DEVICE SafeInt<T> operator++(int){{
        auto old = *this;  // copy old value
        operator++();  // prefix increment
        return old;    // return old value
    }}
}};

}}

// template<typename T, typename U>
// RSI_HOST_DEVICE constexpr auto operator<=>(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {{
//     return lhs.value() <=> rhs.value();
// }}
// 
// template<typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
// RSI_HOST_DEVICE constexpr auto operator<=>(const rsi::SafeInt<T>& lhs, const U& rhs) {{
//     return lhs.value() <=> rhs;
// }}

template<typename T>
std::ostream& operator<<(std::ostream& out, const rsi::SafeInt<T>& x) {{
   return out << x.value();
}}

{non_member_operations}
"""

safe_arith = """
// {op_name} arithmetic operations
template<typename T, typename U>
RSI_HOST_DEVICE constexpr auto operator{op}(const rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {{
    if constexpr (RSI_ARITH_BEHAVIOUR_{op_name} == rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR) {{
        return rsi::SafeInt(lhs.value() * rhs.value());
    }} else if constexpr (RSI_ARITH_BEHAVIOUR_{op_name} == rsi::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE_UNLESS_64) {{
        if constexpr (sizeof(T) == sizeof(U) && sizeof(T) != 8) {{
            static_assert(rsi::detail::always_false<T>, "Performing {op_name} ({op}) on types of the same size is not allowed.");
        }} else {{
            return rsi::SafeInt(lhs.value() * rhs.value());
        }}
    }} else if constexpr (RSI_ARITH_BEHAVIOUR_{op_name} == rsi::rsi_arith_behaviour::SAME_SIZE_PROMOTE_TO_LARGER_SIGNED_UNLESS_64){{
        if constexpr (sizeof(T) == sizeof(U) && sizeof(T) != 8) {{
            using larger = typename rsi::detail::LargerType<T>::type;
            return rsi::SafeInt<larger>(static_cast<larger>(lhs.value()) {op} static_cast<larger>(rhs.value()));
        }} else {{
            return rsi::SafeInt(lhs.value() * rhs.value());
        }}
    }} else if constexpr (RSI_ARITH_BEHAVIOUR_{op_name} == rsi::rsi_arith_behaviour::NONE){{
        static_assert(rsi::detail::always_false<T>, "Performing {op_name} ({op}) is not allowed.");
    }} else {{
        static_assert(rsi::detail::always_false<T>, "Unrecognized behaviour.");
    }}
}}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_HOST_DEVICE constexpr auto operator{op}(const rsi::SafeInt<T>& lhs, const U& rhs) {{
    return lhs {op} rsi::SafeInt(rhs);
}}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_HOST_DEVICE constexpr auto operator{op}(const U& lhs, const rsi::SafeInt<T>& rhs) {{
    return rhs {op} rsi::SafeInt(lhs);
}}

// {op_name} accumulation operators
template<typename T, typename U>
RSI_HOST_DEVICE constexpr auto& operator{op}=(rsi::SafeInt<T>& lhs, const rsi::SafeInt<U>& rhs) {{
    if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_{op_name} == rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR) {{
        return (lhs.value() {op}= rhs.value());
    }} else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_{op_name} == rsi::rsi_arith_accum_behaviour::SMALLER_OR_EQUAL) {{
        if constexpr (sizeof(T) >= sizeof(U)) {{
            return lhs.value() {op}= rhs.value();
        }} else {{
            static_assert(rsi::detail::always_false<T>, "Performing {op_name} accumulation ({op}=) with the right-hand side of a larger type is not allowed.");
        }}
    }} else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_{op_name} == rsi::rsi_arith_accum_behaviour::ONLY_SMALLER){{
        if constexpr (sizeof(T) > sizeof(U)) {{
            return lhs.value() {op}= rhs.value();
        }} else {{
            static_assert(rsi::detail::always_false<T>, "Performing {op_name} accumulation ({op}=) with the right-hand side of a larger or equal type is not allowed.");
        }}
    }} else if constexpr (RSI_ARITH_ACCUM_BEHAVIOUR_{op_name} == rsi::rsi_arith_accum_behaviour::NONE){{
        static_assert(rsi::detail::always_false<T>, "Performing {op_name} accumulation ({op}=) is not allowed.");
        // Return is never called, but this suppresses
        // error: cannot deduce return type 'auto &' for function with no return statements
        return lhs;
    }} else {{
        static_assert(rsi::detail::always_false<T>, "Unrecognized behaviour.");
    }}
}}
template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
RSI_HOST_DEVICE constexpr auto& operator{op}=(rsi::SafeInt<T>& lhs, const U& rhs) {{
    lhs {op}= rsi::SafeInt<U>(rhs);
    return lhs;
}}
"""

arith_behaviour_default_template = """#ifndef RSI_ARITH_BEHAVIOUR_{op_name}
#define RSI_ARITH_BEHAVIOUR_{op_name} rsi::rsi_arith_behaviour::STANDARD_BEHAVIOUR
#endif
"""
arith_accum_behaviour_default_template = """#ifndef RSI_ARITH_ACCUM_BEHAVIOUR_{op_name}
#define RSI_ARITH_ACCUM_BEHAVIOUR_{op_name} rsi::rsi_arith_accum_behaviour::STANDARD_BEHAVIOUR
#endif
"""

# Argparse to have a mandatory flag for cuda ON or off
parser = argparse.ArgumentParser()
# parser.add_argument("--cuda", action="store_true", help="Generate CUDA code")
args = parser.parse_args()

# if args.cuda:
#     rsi_host_device = "__host__ __device__"
# else:
#     rsi_host_device = ""

non_member_operations = []
behaviour_defaults = []
for op in arithmetic_operators:
    non_member_operations.append(safe_arith.format(op_name=op.name, op=op.op))
    behaviour_defaults.append(arith_behaviour_default_template.format(op_name=op.name))
    behaviour_defaults.append(
        arith_accum_behaviour_default_template.format(op_name=op.name)
    )

generated_code_disclaimer = "// This file was @" + "generated by TODO"

with open(OUTPUT_FILE, "w") as fout:
    fout.write(
        FILE_TEMPLATE.format(
            generated_code_disclaimer=generated_code_disclaimer,
            behaviour_defaults="\n".join(behaviour_defaults),
            non_member_operations="\n".join(non_member_operations),
        )
    )
